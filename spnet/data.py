"""ID-aligned ADNI CN-versus-MCI four-modality dataset.

The ADNI ROI MAT file stores T1, fMRI and DTI features as named MATLAB
struct fields, while ASL and diagnostic labels live in TSV/CSV tables.  This
loader treats the exported subject table as the canonical order and validates
every modality against it before any train/validation/test split is made.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.io import loadmat
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler


def _safe_standardize_3d(train, val, test):
    """Fit one scaler per feature channel using train subjects only."""
    shape = train.shape
    scaler = StandardScaler().fit(train.reshape(-1, shape[-1]))
    return tuple(
        scaler.transform(values.reshape(-1, shape[-1])).reshape(values.shape).astype(np.float32)
        for values in (train, val, test)
    )


def _sparsify_nonnegative(matrix, keep_ratio):
    """Keep the strongest non-negative undirected edges and clear self-loops."""
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)
    matrix = (matrix + matrix.T) / 2.0
    np.fill_diagonal(matrix, 0.0)
    matrix[matrix < 0.0] = 0.0
    upper = matrix[np.triu_indices_from(matrix, k=1)]
    positive = upper[upper > 0.0]
    if positive.size == 0:
        return matrix.astype(np.float32)
    cutoff = np.quantile(positive, max(0.0, 1.0 - keep_ratio))
    matrix[matrix < cutoff] = 0.0
    return matrix.astype(np.float32)


class ADNIFourModalDataset:
    """Five-fold, leakage-free loader for complete ADNI CN/MCI data.

    Functional node features are each subject's BOLD FC matrix. The positive
    variant sets negative correlations to zero. The signed variant retains
    them in node features and uses FC magnitude for GCN edge weights, because
    standard GCN normalization requires non-negative graph weights. The
    supplied DTI regional metrics are node features.  Structural graph edges
    are each subject's measured 90-by-90 fiber-number (FN) matrix; no
    FA-similarity or cosine-similarity surrogate graph is constructed.
    """

    dti_fields = ("FA", "L1", "L23m", "MD", "LDHs", "LDHk")

    def __init__(self, mat_path, asl_feature_path, subject_table_path,
                 random_seed=40, keep_ratio=0.5, dti_feature_mode="fa",
                 fc_mode="positive", fn_dir=None, expected_subject_count=708,
                 split_mode="single", test_size=0.15, validation_size=0.15,
                 fn_keep_ratio=1.0):
        self.mat_path = Path(mat_path)
        self.asl_feature_path = Path(asl_feature_path)
        self.subject_table_path = Path(subject_table_path)
        self.random_seed = int(random_seed)
        self.keep_ratio = float(keep_ratio)
        self.fn_keep_ratio = float(fn_keep_ratio)
        self.fn_dir = Path(fn_dir) if fn_dir is not None else None
        self.expected_subject_count = int(expected_subject_count)
        if split_mode not in {"single", "five_fold"}:
            raise ValueError("split_mode must be 'single' or 'five_fold'.")
        if not 0.0 < test_size < 1.0 or not 0.0 < validation_size < 1.0 - test_size:
            raise ValueError("test_size and validation_size must be positive and sum to less than 1.")
        self.split_mode = split_mode
        self.test_size = float(test_size)
        self.validation_size = float(validation_size)
        if self.fn_dir is None:
            raise ValueError(
                "fn_dir is required: use subject-specific FN matrices as DTI graph edges; "
                "FA/cosine surrogate adjacency is no longer supported."
            )
        if not self.fn_dir.is_dir():
            raise FileNotFoundError(self.fn_dir)
        if not 0.0 < self.keep_ratio <= 1.0 or not 0.0 < self.fn_keep_ratio <= 1.0:
            raise ValueError("FC and FN keep ratios must both be in (0, 1].")
        # ``dti_feature_mode`` remains the public argument for backwards
        # compatibility.  In addition to ``fa`` and ``all``, it accepts a
        # comma-separated subset, e.g. ``FA,L1,MD``.  This makes the DTI
        # ablation use exactly the requested regional measures throughout
        # node features and graph construction.
        requested_dti = str(dti_feature_mode).strip()
        if requested_dti.lower() == "fa":
            selected_dti_fields = ("FA",)
        elif requested_dti.lower() == "all":
            selected_dti_fields = self.dti_fields
        else:
            selected_dti_fields = tuple(
                item.strip() for item in requested_dti.split(",") if item.strip()
            )
            invalid = set(selected_dti_fields) - set(self.dti_fields)
            if not selected_dti_fields or invalid or len(set(selected_dti_fields)) != len(selected_dti_fields):
                raise ValueError(
                    "dti_feature_mode must be 'fa', 'all', or a unique comma-separated "
                    f"subset of {self.dti_fields}; got {dti_feature_mode!r}."
                )
        if fc_mode not in {"positive", "signed"}:
            raise ValueError("fc_mode must be 'positive' or 'signed'.")
        self.dti_feature_mode = requested_dti
        self.dti_feature_fields = selected_dti_fields
        self.dti_feature_indices = tuple(self.dti_fields.index(name) for name in selected_dti_fields)
        self.fc_mode = fc_mode
        self._load_and_validate()
        self._build_splits()

    @staticmethod
    def _mat_field(feature_subject):
        if not str(feature_subject).startswith("sub-"):
            raise ValueError(f"Unexpected ADNI feature ID: {feature_subject}")
        return "sub_" + str(feature_subject)[4:]

    @staticmethod
    def _mat_subjects(struct):
        return {"sub-" + field[4:] for field in struct._fieldnames}

    def _load_and_validate(self):
        if not self.mat_path.is_file():
            raise FileNotFoundError(self.mat_path)
        if not self.asl_feature_path.is_file():
            raise FileNotFoundError(self.asl_feature_path)
        if not self.subject_table_path.is_file():
            raise FileNotFoundError(self.subject_table_path)

        table = pd.read_csv(self.subject_table_path)
        required_columns = {
            "ASL_feature_subject", "Subject", "Group", "T1_available",
            "fMRI_available", "DTI_available", "ASL_available",
            "ASL_nan_roi_count",
        }
        missing_columns = required_columns - set(table.columns)
        if missing_columns:
            raise ValueError(f"Subject table is missing columns: {sorted(missing_columns)}")
        if table["ASL_feature_subject"].duplicated().any():
            raise ValueError("Subject table has duplicate feature-subject IDs.")
        if set(table["Group"]) - {"CN", "MCI"}:
            raise ValueError("This ADNI trainer accepts only exact CN/MCI labels.")
        if not table[["T1_available", "fMRI_available", "DTI_available", "ASL_available"]].all().all():
            raise ValueError("Subject table contains a modality marked unavailable.")
        if not table["ASL_nan_roi_count"].eq(0).all():
            raise ValueError("Subject table contains incomplete ASL ROI data.")

        self.subject_ids = table["ASL_feature_subject"].astype(str).to_numpy()
        self.original_indices = np.arange(len(table), dtype=np.int64)
        self.y = table["Group"].map({"CN": 0, "MCI": 1}).to_numpy(dtype=np.int64)
        self.subject_table = table.copy()

        asl = pd.read_csv(self.asl_feature_path, sep="\t")
        if "subject" not in asl.columns or asl.shape[1] != 91:
            raise ValueError("ASL feature TSV must have one subject column and 90 AAL ROI columns.")
        if asl["subject"].duplicated().any():
            raise ValueError("ASL feature TSV has duplicate subjects.")
        asl = asl.set_index("subject").reindex(self.subject_ids)
        if asl.isnull().any().any():
            missing = asl.index[asl.isnull().any(axis=1)].tolist()
            raise ValueError(f"ASL feature TSV is missing selected subjects: {missing[:5]}")
        self.cbf = asl.to_numpy(dtype=np.float32)[..., None]

        mat = loadmat(self.mat_path, struct_as_record=False, squeeze_me=False)
        bold_struct = mat["fMRI"][0, 0].BOLD[0, 0]
        gm_struct = mat["T1"][0, 0].GM[0, 0]
        dti_structs = {field: getattr(mat["DTI"][0, 0], field)[0, 0]
                       for field in self.dti_fields}
        expected = set(self.subject_ids)
        modality_structs = {"BOLD": bold_struct, "GM": gm_struct, **dti_structs}
        for name, struct in modality_structs.items():
            observed = self._mat_subjects(struct)
            missing = expected - observed
            if missing:
                raise ValueError(
                    f"{name} MAT is missing selected cohort subjects: {sorted(missing)[:5]}"
                )

        fc_features, func_adjacencies = [], []
        gm, dti_values, fn_matrices = [], [], []
        for subject in self.subject_ids:
            field = self._mat_field(subject)
            bold = np.asarray(getattr(bold_struct, field), dtype=np.float32)
            if bold.shape[0] != 90 or bold.shape[1] < 3:
                raise ValueError(f"{subject}: BOLD must have shape [90, time>=3], got {bold.shape}")
            bold = np.nan_to_num(bold, nan=0.0, posinf=0.0, neginf=0.0)
            # BOLD FC is the functional node feature: every ROI carries its
            # 90-edge connectivity profile. Signed FC cannot be fed directly
            # as GCN edge weights, so the signed condition uses magnitude only
            # for graph edges while retaining signed values in node features.
            fc = np.corrcoef(bold)
            fc = np.nan_to_num(fc, nan=0.0, posinf=0.0, neginf=0.0)
            fc = np.clip(fc, -1.0, 1.0)
            np.fill_diagonal(fc, 0.0)
            if self.fc_mode == "positive":
                fc_features.append(np.maximum(fc, 0.0).astype(np.float32))
                adjacency_source = np.maximum(fc, 0.0)
            else:
                fc_features.append(fc.astype(np.float32))
                adjacency_source = np.abs(fc)
            func_adjacencies.append(_sparsify_nonnegative(adjacency_source, self.keep_ratio))

            gm_value = np.asarray(getattr(gm_struct, field), dtype=np.float32).reshape(-1)
            if gm_value.size != 90:
                raise ValueError(f"{subject}: T1 GM must have 90 ROIs, got {gm_value.size}")
            gm.append(np.nan_to_num(gm_value, nan=0.0, posinf=0.0, neginf=0.0))

            subject_dti = []
            for dti_field in self.dti_fields:
                value = np.asarray(getattr(dti_structs[dti_field], field), dtype=np.float32).reshape(-1)
                if value.size != 90:
                    raise ValueError(f"{subject}: {dti_field} must have 90 ROIs, got {value.size}")
                subject_dti.append(np.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0))
            subject_dti = np.stack(subject_dti, axis=1)  # [90, 6]
            selected_dti = subject_dti[:, self.dti_feature_indices]
            dti_values.append(selected_dti)

            fn_path = self.fn_dir / f"{subject}_FN_AAL90.txt"
            if not fn_path.is_file():
                raise FileNotFoundError(f"{subject}: missing FN matrix {fn_path}")
            fn_matrix = np.loadtxt(fn_path, dtype=np.float32)
            if fn_matrix.shape != (90, 90) or not np.isfinite(fn_matrix).all():
                raise ValueError(f"{subject}: FN matrix must be finite [90, 90], got {fn_matrix.shape}")
            if (fn_matrix < 0).any():
                raise ValueError(f"{subject}: FN matrix contains negative fiber counts")
            fn_matrix = (fn_matrix + fn_matrix.T) / 2.0
            np.fill_diagonal(fn_matrix, 0.0)
            fn_matrices.append(_sparsify_nonnegative(fn_matrix, self.fn_keep_ratio))

        self.bold_mean = np.stack(fc_features, axis=0).astype(np.float32)
        self.A_funcs = np.stack(func_adjacencies, axis=0)
        self.gm = np.stack(gm, axis=0)[..., None].astype(np.float32)
        self.fa = np.stack(dti_values, axis=0).astype(np.float32)
        self.fn_mats = np.stack(fn_matrices, axis=0)
        self._validate_arrays()

    def _validate_arrays(self):
        n = len(self.y)
        arrays = {
            "ASL": self.cbf, "fMRI features": self.bold_mean,
            "functional adjacency": self.A_funcs, "T1": self.gm,
            "DTI features": self.fa, "DTI adjacency": self.fn_mats,
        }
        if n != self.expected_subject_count:
            raise ValueError(
                f"Expected {self.expected_subject_count} selected CN/MCI subjects with FN, got {n}."
            )
        for name, values in arrays.items():
            if len(values) != n or not np.isfinite(values).all():
                raise ValueError(f"Invalid {name} array after ADNI loading.")
        expected_dti_dim = len(self.dti_feature_fields)
        if self.bold_mean.shape[1:] != (90, 90) or self.fa.shape[1:] != (90, expected_dti_dim):
            raise ValueError("Unexpected ADNI node-feature dimensions.")
        if self.A_funcs.shape[1:] != (90, 90) or self.fn_mats.shape[1:] != (90, 90):
            raise ValueError("Unexpected ADNI graph dimensions.")

    def _build_splits(self):
        self.dataset = {}
        def store_split(key, train_idx, val_idx, test_idx):
            xft, xfv, xfs = _safe_standardize_3d(
                self.bold_mean[train_idx], self.bold_mean[val_idx], self.bold_mean[test_idx]
            )
            xst, xsv, xss = _safe_standardize_3d(
                self.fa[train_idx], self.fa[val_idx], self.fa[test_idx]
            )
            xat, xav, xas = _safe_standardize_3d(
                self.cbf[train_idx], self.cbf[val_idx], self.cbf[test_idx]
            )
            xtt, xtv, xts = _safe_standardize_3d(
                self.gm[train_idx], self.gm[val_idx], self.gm[test_idx]
            )
            struct_train = self.fn_mats[train_idx].copy()
            struct_val = self.fn_mats[val_idx].copy()
            struct_test = self.fn_mats[test_idx].copy()
            for matrices in (struct_train, struct_val, struct_test):
                maxima = matrices.max(axis=(1, 2), keepdims=True)
                matrices /= np.maximum(maxima, 1e-8)
            self.dataset[key] = {
                "func_adj": (self.A_funcs[train_idx], self.A_funcs[val_idx], self.A_funcs[test_idx]),
                "func_feat": (xft, xfv, xfs),
                "struct_adj": (struct_train, struct_val, struct_test),
                "struct_feat": (xst, xsv, xss),
                "label": (self.y[train_idx], self.y[val_idx], self.y[test_idx]),
                "indices": (train_idx, val_idx, test_idx),
                "cbf": (xat, xav, xas),
                "gm": (xtt, xtv, xts),
            }

        if self.split_mode == "single":
            train_val_idx, test_idx = train_test_split(
                np.arange(len(self.y)), test_size=self.test_size, random_state=self.random_seed,
                stratify=self.y,
            )
            validation_relative_size = self.validation_size / (1.0 - self.test_size)
            train_idx, val_idx = train_test_split(
                train_val_idx, test_size=validation_relative_size, random_state=self.random_seed + 1,
                stratify=self.y[train_val_idx],
            )
            store_split("split", train_idx, val_idx, test_idx)
        else:
            outer = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_seed)
            for fold, (train_val_idx, test_idx) in enumerate(outer.split(np.zeros(len(self.y)), self.y), start=1):
                train_idx, val_idx = train_test_split(
                    train_val_idx, test_size=0.2, random_state=self.random_seed + fold,
                    stratify=self.y[train_val_idx],
                )
                store_split(f"cv{fold}", train_idx, val_idx, test_idx)

    def __call__(self, cv=None, tensor=True, device=None):
        key = "split" if self.split_mode == "single" else f"cv{cv}"
        ds = self.dataset[key]
        if not tensor:
            return ds
        to_tensor = lambda values: torch.tensor(values, dtype=torch.float32, device=device)
        return {
            "A_func": tuple(to_tensor(values) for values in ds["func_adj"]),
            "X_func": tuple(to_tensor(values) for values in ds["func_feat"]),
            "A_struct": tuple(to_tensor(values) for values in ds["struct_adj"]),
            "X_struct": tuple(to_tensor(values) for values in ds["struct_feat"]),
            "y": ds["label"],
            "indices": ds["indices"],
            "X_asl": tuple(to_tensor(values) for values in ds["cbf"]),
            "X_t1": tuple(to_tensor(values) for values in ds["gm"]),
        }

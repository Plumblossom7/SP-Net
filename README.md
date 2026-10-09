# SP-Net

SP-Net is a four-modality graph neural network for ADNI CN-versus-MCI
classification. It uses T1 gray-matter ROI features, BOLD functional
connectivity, DTI regional features with subject-specific fiber-number graphs,
and ASL cerebral-blood-flow ROI features.

## Repository layout

- `spnet/model.py` — SP-Net model and loss functions.
- `spnet/data.py` — ID-aligned, leakage-free ADNI four-modality data loader.
- `spnet/graph.py` — graph construction, reproducibility helpers, and metrics.
- `spnet/engine.py` — training, validation, test evaluation, and early stopping.
- `spnet/attribution.py` — ROI attribution utilities.
- `scripts/train.py` — five-fold CN-versus-MCI training entry point.
- `configs/adni_cn_mci.yaml` — documented default settings.

## Data access

ADNI data are access-controlled and are deliberately excluded from this
repository. Prepare the following layout from data obtained through ADNI:

```text
CN_MCI_4modal_complete/
├── ROI_T1_fMRI_DTI_AAL90_CN_MCI_4modal_complete.mat
├── CN_MCI_4modal_complete_FN_subjects.csv
├── ASL_CBF/
│   └── CBF_AAL90_features.tsv
└── FN/
    └── sub-*_FN_AAL90.txt
```

The participant table is the canonical subject order. The loader validates
that every selected participant has all four modalities, complete ASL ROI
features, and a valid 90-by-90 fiber-number matrix. The prepared cohort has
708 participants: 467 CN and 241 MCI.

## Installation

Install PyTorch appropriate for your CUDA/CPU environment first, then install
the remaining packages:

```bash
pip install -r requirements.txt
```

`torch-geometric` and `torch-scatter` wheels must be compatible with the
installed PyTorch and CUDA versions; consult the PyG installation guide if a
binary wheel is not available for your platform.

## Training

Run from the repository root:

```bash
python scripts/train.py --data-root /path/to/CN_MCI_4modal_complete
```

For a short smoke test:

```bash
python scripts/train.py --data-root /path/to/CN_MCI_4modal_complete --quick
```

Each outer fold keeps its test set untouched; a validation partition is drawn
only from the remaining training participants. Results, predictions, and model
checkpoints are written under `ra/`, which is ignored by Git.

## Not included

Raw/derived ADNI data, subject-level outputs, trained weights, TensorBoard
logs, and exploratory or historical experiment scripts are intentionally not
published. The code here is the focused CN-versus-MCI reproducibility release.

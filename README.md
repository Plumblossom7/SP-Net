# SP-Net

SP-Net is a four-modality graph neural network for ADNI CN-versus-MCI
classification. It uses T1 gray-matter ROI features, BOLD functional
connectivity, DTI regional features with subject-specific fiber-number graphs,
and ASL cerebral-blood-flow ROI features.

## Repository layout

- `spnet/data.py` — ID-aligned, leakage-free ADNI four-modality loader.
- `spnet/stable_model.py` — StableV3 model used by the final experiment.
- `spnet/stable_engine.py` — end-to-end training and evaluation helpers.
- `scripts/train.py` — single stratified CN-versus-MCI training entry point.
- `configs/adni_cn_mci.yaml` — executable final-experiment configuration.

## Data access

ADNI data are access-controlled and deliberately excluded from this repository.
Prepare the following layout from data obtained through ADNI:

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
installed PyTorch and CUDA versions.

## Training

Run from the repository root:

```bash
python scripts/train.py --data-root /path/to/CN_MCI_4modal_complete
```

For a short smoke test:

```bash
python scripts/train.py --data-root /path/to/CN_MCI_4modal_complete --quick
```

The default configuration is the final StableV3 setting: seed 42, a single
stratified 70%/15%/15% train/validation/test split, signed FC features with a
0.2 edge retention ratio, FA node features, full FN structural graphs, AdamW
at 5e-5, and 250 epochs. The validation split selects the checkpoint; the test
curve is diagnostic only. Results, predictions, and model checkpoints are
written under `ra/`, which is ignored by Git.

## Not included

Raw/derived ADNI data, subject-level outputs, trained weights, TensorBoard
logs, and exploratory or historical experiment scripts are intentionally not
published.

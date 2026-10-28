"""Reproducible single-split (70/15/15) ADNI CN-versus-MCI training."""

from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    ConfusionMatrixDisplay, accuracy_score, balanced_accuracy_score, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from torch.utils.data import DataLoader

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from spnet.data import ADNIFourModalDataset
from spnet.graph import IndexDataset, make_graph_lists, set_seed
from spnet.stable_engine import evaluate_one_epoch, train_one_epoch, validate_one_epoch
from spnet.stable_model import MultiModalGNNStableV3


def load_scalar_yaml(path):
    """Read this repository's deliberately flat, scalar-only YAML config.

    Keeping the parser dependency-free lets the quick validation run in the
    existing PyTorch environment; nested YAML is intentionally unsupported.
    """
    config = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.split("#", 1)[0].strip()
        if not line:
            continue
        key, value = (part.strip() for part in line.split(":", 1))
        lowered = value.lower()
        if lowered in {"true", "false"}:
            config[key] = lowered == "true"
        else:
            try:
                config[key] = int(value)
            except ValueError:
                try:
                    config[key] = float(value)
                except ValueError:
                    config[key] = value.strip('"\'')
    return config


def make_model(functional_graphs, structural_graphs, config, device):
    model = MultiModalGNNStableV3(
        in_dim1=functional_graphs[0].x.shape[1], in_dim2=structural_graphs[0].x.shape[1],
        aux_dim=config["aux_dim"], shared_dim=config["shared_dim"], private_dim=config["shared_dim"],
        num_classes=2, tau=config["tau"], lambda_param=config["lambda_param"], hidden=config["hidden"],
    ).to(device)
    model.unfreeze_all()
    for module in model.modules():
        if isinstance(module, nn.Dropout):
            module.p = config["dropout"]
    model.supcon_temperature = config["supcon_temperature"]
    model.dropedge_prob = config["dropedge"]
    model.graph_pooling = config["graph_pooling"]
    for encoder in (model.encoder_func, model.encoder_struct):
        encoder.num_layers = config["gcn_layers"]
    return model


def make_optimizer(model, config):
    if config["no_bias_decay"]:
        decay, no_decay = [], []
        for name, parameter in model.named_parameters():
            (no_decay if name.endswith(".bias") else decay).append(parameter)
        return torch.optim.AdamW(
            [{"params": decay, "weight_decay": config["weight_decay"]}, {"params": no_decay, "weight_decay": 0.0}],
            lr=config["lr"],
        )
    return torch.optim.AdamW(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])


def save_curve(history, path):
    figure, axes = plt.subplots(1, 3, figsize=(14, 4))
    axes[0].plot(history["epoch"], history["train_loss"], label="train total loss")
    axes[0].plot(history["epoch"], history["val_loss"], label="validation CE loss")
    axes[1].plot(history["epoch"], history["train_acc"], label="train balanced accuracy")
    axes[1].plot(history["epoch"], history["val_acc"], label="validation accuracy")
    axes[2].plot(history["epoch"], history["val_macro_f1"], label="validation Macro-F1")
    if "test_loss" in history:
        axes[0].plot(history["epoch"], history["test_loss"], label="test CE loss (diagnostic only)")
        axes[1].plot(history["epoch"], history["test_acc"], label="test accuracy (diagnostic only)")
        axes[2].plot(history["epoch"], history["test_macro_f1"], label="test Macro-F1 (diagnostic only)")
    for axis, title, ylabel in zip(axes, ("Loss", "Accuracy", "Checkpoint selection"), ("loss", "accuracy", "Macro-F1")):
        axis.set(xlabel="epoch", ylabel=ylabel, title=title)
        axis.grid(alpha=0.25)
        axis.legend()
    axes[1].set_ylim(0, 1.05)
    axes[2].set_ylim(0, 1.05)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=REPOSITORY_ROOT / "configs" / "adni_cn_mci.yaml")
    parser.add_argument("--data-root", type=Path, default=None, help="Overrides data_root in the YAML config.")
    parser.add_argument("--result-dir", type=Path, default=None)
    parser.add_argument("--quick", action="store_true", help="Run two epochs to verify the installation and data layout.")
    args = parser.parse_args()
    config = load_scalar_yaml(args.config)
    if args.data_root is not None:
        config["data_root"] = str(args.data_root)
    root = Path(config["data_root"])
    if not root.is_dir():
        raise FileNotFoundError(f"Set data_root to the complete ADNI cohort directory, got: {root}")
    if args.quick:
        config.update(epochs=2, disable_early_stopping=False, track_test_curve=False)
    output = args.result_dir or REPOSITORY_ROOT / "ra" / f"adni-cn-mci-holdout-s{config['seed']}-{datetime.now():%Y%m%d-%H%M%S}"
    output.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(config["seed"])

    dataset = ADNIFourModalDataset(
        mat_path=root / config["mat_file"], asl_feature_path=root / config["asl_feature_file"],
        subject_table_path=root / config["subject_table_file"], fn_dir=root / config["fn_directory"],
        random_seed=config["seed"], keep_ratio=config["fc_keep_ratio"], fn_keep_ratio=config["fn_keep_ratio"],
        dti_feature_mode=config["dti_features"], fc_mode=config["fc_mode"],
        expected_subject_count=config["expected_subject_count"], split_mode="single",
        test_size=config["test_size"], validation_size=config["validation_size"],
    )
    data = dataset(tensor=True, device=device)
    af_train, af_val, af_test = data["A_func"]
    xf_train, xf_val, xf_test = data["X_func"]
    as_train, as_val, as_test = data["A_struct"]
    xs_train, xs_val, xs_test = data["X_struct"]
    y_train, y_val, y_test = data["y"]
    asl_train, asl_val, asl_test = data["X_asl"]
    t1_train, t1_val, t1_test = data["X_t1"]
    train_idx, val_idx, test_idx = (np.asarray(indices, dtype=np.int64) for indices in data["indices"])
    fg_train, sg_train = make_graph_lists(af_train, xf_train, as_train, xs_train, y_train, device)
    fg_val, sg_val = make_graph_lists(af_val, xf_val, as_val, xs_val, y_val, device)
    fg_test, sg_test = make_graph_lists(af_test, xf_test, as_test, xs_test, y_test, device)

    rows = []
    for split, indices in (("train", train_idx), ("validation", val_idx), ("test", test_idx)):
        rows.extend({"split": split, "cohort_row": int(index), "subject_id": dataset.subject_ids[index],
                     "label": int(dataset.y[index]), "diagnosis": "MCI" if dataset.y[index] else "CN"} for index in indices)
    pd.DataFrame(rows).to_csv(output / "split_subjects.csv", index=False)

    counts = np.bincount(np.asarray(y_train, dtype=np.int64), minlength=2)
    weights = len(y_train) / (2.0 * counts.astype(np.float32))
    criterion = nn.CrossEntropyLoss(weight=torch.tensor(weights, dtype=torch.float32, device=device))
    eval_criterion = nn.CrossEntropyLoss()
    train_loader = DataLoader(IndexDataset(range(len(fg_train))), batch_size=config["batch_size"], shuffle=True)
    val_loader = DataLoader(IndexDataset(range(len(fg_val))), batch_size=config["batch_size"], shuffle=False)
    test_loader = DataLoader(IndexDataset(range(len(fg_test))), batch_size=config["batch_size"], shuffle=False)
    model, optimizer = make_model(fg_train, sg_train, config, device), None
    optimizer = make_optimizer(model, config)

    history = {key: [] for key in ("epoch", "train_loss", "train_cls_loss", "train_acc", "val_loss", "val_acc", "val_macro_f1")}
    if config["track_test_curve"]:
        history.update({"test_loss": [], "test_acc": [], "test_macro_f1": []})
    best, stale = {"score": -1.0, "epoch": None, "state": None}, 0
    for epoch in range(1, config["epochs"] + 1):
        train_loss, train_acc, _, _, train_cls_loss = train_one_epoch(
            config["laux"], config["lspe"], fg_train, sg_train, t1_train, asl_train, model, train_loader,
            optimizer, criterion, config["lcont"], config["lothro"], device, lsupcon=config["lsupcon"],
        )
        val_loss, val_acc, _, val_prediction = validate_one_epoch(fg_val, sg_val, t1_val, asl_val, model, val_loader, eval_criterion, device)
        val_f1 = f1_score(y_val, val_prediction, average="macro", zero_division=0)
        for key, value in (("epoch", epoch), ("train_loss", train_loss), ("train_cls_loss", train_cls_loss),
                           ("train_acc", train_acc), ("val_loss", val_loss), ("val_acc", val_acc), ("val_macro_f1", val_f1)):
            history[key].append(float(value))
        if config["track_test_curve"]:
            test_loss, test_prediction, _, _ = evaluate_one_epoch(fg_test, sg_test, t1_test, asl_test, model, test_loader, eval_criterion, device)
            history["test_loss"].append(float(test_loss))
            history["test_acc"].append(float(accuracy_score(y_test, test_prediction)))
            history["test_macro_f1"].append(float(f1_score(y_test, test_prediction, average="macro", zero_division=0)))
        if val_f1 > best["score"]:
            best, stale = {"score": float(val_f1), "epoch": epoch, "state": copy.deepcopy(model.state_dict())}, 0
        else:
            stale += 1
            if stale >= config["patience"] and not config["disable_early_stopping"]:
                break

    model.load_state_dict(best["state"])
    torch.save(best["state"], output / "best_validation_model.pth")
    pd.DataFrame(history).to_csv(output / "epoch_metrics.csv", index=False)
    save_curve(history, output / "training_validation_curve.png")
    val_true, _, val_score = evaluate_one_epoch(fg_val, sg_val, t1_val, asl_val, model, val_loader, eval_criterion, device)
    thresholds = pd.DataFrame({"threshold": np.linspace(0.25, 0.75, 101)})
    thresholds["validation_macro_f1"] = [f1_score(val_true, np.asarray(val_score) >= value, average="macro", zero_division=0) for value in thresholds.threshold]
    thresholds["validation_balanced_accuracy"] = [balanced_accuracy_score(val_true, np.asarray(val_score) >= value) for value in thresholds.threshold]
    thresholds["distance_to_0.5"] = (thresholds.threshold - 0.5).abs()
    threshold = float(thresholds.sort_values(["validation_macro_f1", "validation_balanced_accuracy", "distance_to_0.5"], ascending=[False, False, True]).iloc[0].threshold)
    thresholds.drop(columns="distance_to_0.5").to_csv(output / "validation_threshold_search.csv", index=False)

    y_true, _, y_score = evaluate_one_epoch(fg_test, sg_test, t1_test, asl_test, model, test_loader, eval_criterion, device)
    y_pred = (np.asarray(y_score) >= threshold).astype(np.int64)
    metrics = {
        "n_train": len(train_idx), "n_validation": len(val_idx), "n_test": len(test_idx),
        "train_counts_CN_MCI": counts.tolist(), "class_weights_CN_MCI": weights.tolist(),
        "best_validation_macro_f1": best["score"], "selected_epoch": best["epoch"], "validation_selected_threshold": threshold,
        "test_accuracy": accuracy_score(y_true, y_pred), "test_balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "test_macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "test_mci_recall": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "test_mci_precision": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "test_mci_f1": f1_score(y_true, y_pred, pos_label=1, zero_division=0), "test_auc": roc_auc_score(y_true, y_score),
    }
    pd.DataFrame([metrics]).to_csv(output / "test_metrics.csv", index=False)
    pd.DataFrame({"cohort_row": test_idx, "subject_id": dataset.subject_ids[test_idx], "true_label": y_true,
                  "true_diagnosis": np.where(y_true, "MCI", "CN"), "predicted_label": y_pred,
                  "predicted_diagnosis": np.where(y_pred, "MCI", "CN"), "probability_MCI": y_score,
                  "probability_CN": 1.0 - np.asarray(y_score)}).to_csv(output / "test_subject_predictions.csv", index=False)
    figure, axis = plt.subplots(figsize=(5, 4))
    ConfusionMatrixDisplay.from_predictions(y_true, y_pred, display_labels=("CN", "MCI"), ax=axis, cmap="Blues")
    axis.set_title("ADNI CN/MCI holdout test")
    figure.tight_layout()
    figure.savefig(output / "test_confusion_matrix.png", dpi=180)
    plt.close(figure)
    config.update(device=str(device), cohort="708 subjects: CN=467, MCI=241", split="stratified train/validation/test = 70%/15%/15%")
    (output / "run_parameters.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    print(f"Results written to: {output}")


if __name__ == "__main__":
    main()

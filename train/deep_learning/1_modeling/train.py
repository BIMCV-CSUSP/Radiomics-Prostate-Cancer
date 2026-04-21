#!/usr/bin/env python
"""
Train deep learning models for prostate cancer classification with grouped cross-validation.

This version improves the original pipeline by:
1. Persisting grouped cross-validation splits for downstream reuse.
2. Seeding Python, NumPy, and PyTorch for reproducibility.
3. Saving the real best checkpoint with a deep copy of the model state.
4. Writing comments, logs, and result files in English.
"""

from __future__ import annotations

import argparse
import copy
import importlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
COMMON_DIR = PROJECT_ROOT / "train" / "common"
if str(COMMON_DIR) not in sys.path:
    sys.path.append(str(COMMON_DIR))

import numpy as np
import pandas as pd
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.optim as optim
from monai.data import DataLoader, Dataset
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from runtime_utils import load_or_create_splits, resolve_project_path, set_global_seed, setup_logger

mp.set_sharing_strategy("file_system")


def dynamic_import(class_path: str):
    """Import a class from a fully qualified module path."""

    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def build_extra_transforms(config: dict, logger):
    """Instantiate optional MONAI transforms defined in the JSON configuration."""

    extra_transforms = []
    for transform_item in config.get("extra_transforms", []):
        if isinstance(transform_item, dict):
            transform_class_name = transform_item.get("class")
            transform_args = transform_item.get("args", {})
        else:
            transform_class_name = transform_item
            transform_args = config.get("extra_transform_args", {})

        TransformClass = dynamic_import(transform_class_name)
        extra_transforms.append(TransformClass(**transform_args))
        logger.info(
            "Added data augmentation transform %s with args=%s",
            transform_class_name,
            transform_args,
        )

    if not extra_transforms:
        logger.info("No extra augmentation transforms were defined in the configuration.")

    return extra_transforms


def compute_class_weights(train_subset: list[dict], num_classes: int) -> torch.Tensor:
    """Compute inverse-frequency class weights from the current training subset."""

    class_counts = torch.zeros(num_classes, dtype=torch.long)
    for item in train_subset:
        class_index = torch.argmax(item["label"]).item()
        class_counts[class_index] += 1

    train_size = len(train_subset)
    weights = [
        train_size / (class_counts[class_index].item() if class_counts[class_index] > 0 else 1e-6)
        for class_index in range(num_classes)
    ]
    weights_tensor = torch.tensor(weights, dtype=torch.float32)
    return weights_tensor / weights_tensor.sum()


def calculate_epoch_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> dict:
    """Compute classification metrics for one validation epoch."""

    try:
        auc_value = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc_value = np.nan

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    per_class_precision = precision_score(y_true, y_pred, average=None, zero_division=0)
    per_class_recall = recall_score(y_true, y_pred, average=None, zero_division=0)
    per_class_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    per_class_accuracy = []
    for row_index in range(len(cm)):
        row_sum = cm[row_index, :].sum()
        per_class_accuracy.append(cm[row_index, row_index] / row_sum if row_sum > 0 else np.nan)

    return {
        "auc": auc_value,
        "mcc": matthews_corrcoef(y_true, y_pred),
        "kappa": cohen_kappa_score(y_true, y_pred),
        "f1_binary": f1_score(y_true, y_pred, average="binary", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "sensitivity": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "specificity": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "ppv": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "npv": tn / (tn + fn) if (tn + fn) > 0 else np.nan,
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "per_class_precision": per_class_precision.tolist(),
        "per_class_recall": per_class_recall.tolist(),
        "per_class_f1": per_class_f1.tolist(),
        "per_class_accuracy": per_class_accuracy,
    }


def main():
    """Train all folds for the selected deep learning experiment configuration."""

    parser = argparse.ArgumentParser(description="Train a MONAI model with grouped cross-validation.")
    parser.add_argument("--config_key", type=str, required=True, help="Configuration key inside the JSON file.")
    parser.add_argument(
        "--config_file",
        type=str,
        default="train/deep_learning/1_modeling/config.json",
        help="Path to the JSON configuration file.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["full", "gland"],
        required=True,
        help="Input mode: full field of view or gland-only ROI.",
    )
    parser.add_argument(
        "--csv_path",
        type=str,
        default="artifacts/data.csv",
        help="Path to the dataset CSV file.",
    )
    parser.add_argument(
        "--input_shape",
        type=int,
        nargs=3,
        default=[128, 128, 32],
        help="Input tensor shape as three integers.",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Maximum number of training epochs.")
    parser.add_argument("--n_splits", type=int, default=5, help="Number of grouped validation folds.")
    parser.add_argument("--batch_size", type=int, default=2, help="Batch size used by the DataLoaders.")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of DataLoader workers.")
    parser.add_argument("--learning_rate", type=float, default=1e-5, help="Optimizer learning rate.")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience in epochs.")
    parser.add_argument("--seed", type=int, default=42, help="Global seed used for reproducibility.")
    args = parser.parse_args()

    set_global_seed(args.seed)

    config_path = resolve_project_path(PROJECT_ROOT, args.config_file)
    with config_path.open("r", encoding="utf-8") as file_handle:
        configs = json.load(file_handle)

    if args.config_key not in configs:
        raise ValueError(f"Configuration key '{args.config_key}' was not found in {config_path}.")

    config = configs[args.config_key]
    csv_path = resolve_project_path(PROJECT_ROOT, args.csv_path)
    input_shape = tuple(args.input_shape)

    if args.mode == "full":
        loader_module = "data_loaders.data_loader_for_cv_org"
    else:
        loader_module = "data_loaders.data_loader_for_cv_roi"

    MyDataLoader = dynamic_import(f"{loader_module}.MyDataLoader")

    base_dir = PROJECT_ROOT / "artifacts" / "deep_learning" / args.mode
    logs_dir = base_dir / "logs"
    models_dir = base_dir / "models" / args.config_key
    results_dir = base_dir / "results" / args.config_key
    splits_dir = base_dir / "splits"
    metadata_dir = base_dir / "metadata"
    for directory in [logs_dir, models_dir, results_dir, splits_dir, metadata_dir]:
        directory.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("deep_learning_train", logs_dir / f"{args.config_key}.log")
    logger.info("Loaded configuration key '%s': %s", args.config_key, config)
    logger.info("Project root: %s", PROJECT_ROOT)
    logger.info("Dataset CSV: %s", csv_path)
    logger.info("Input mode: %s", args.mode)

    extra_transforms = build_extra_transforms(config=config, logger=logger)

    data_loader = MyDataLoader(
        csv_path=str(csv_path),
        input_shape=input_shape,
        config={"batch_size": args.batch_size, "num_workers": args.num_workers},
        transformations=extra_transforms,
        num_classes=config.get("model_args", {}).get("num_classes", 2),
    )
    all_data = data_loader.get_all_data()
    all_labels = [int(torch.argmax(item["label"]).item()) for item in all_data]
    patient_ids = [item["patient_id"] for item in all_data]

    split_file = splits_dir / f"{csv_path.stem}_{args.mode}_{args.n_splits}fold_seed{args.seed}.json"
    split_bundle = load_or_create_splits(
        split_file=split_file,
        labels=all_labels,
        groups=patient_ids,
        n_splits=args.n_splits,
        seed=args.seed,
        metadata={
            "csv_path": str(csv_path),
            "mode": args.mode,
            "n_splits": args.n_splits,
            "seed": args.seed,
        },
    )
    logger.info("Using persisted split definition from %s", split_file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Training device: %s", device)

    best_overall_model = None
    best_overall_score = -np.inf
    best_split_info = {"split": None, "epoch": None}

    for split_info in split_bundle["splits"]:
        split_index = split_info["split_index"]
        train_idx = split_info["train_indices"]
        val_idx = split_info["validation_indices"]
        logger.info("=== Starting split %s/%s ===", split_index, args.n_splits)

        train_subset = [all_data[index] for index in train_idx]
        val_subset = [all_data[index] for index in val_idx]

        class_weights = compute_class_weights(
            train_subset=train_subset,
            num_classes=config.get("model_args", {}).get("num_classes", 2),
        )
        logger.info("Class weights for split %s: %s", split_index, class_weights.tolist())

        train_dataset = Dataset(data=train_subset, transform=data_loader.get_transforms(augment=True))
        val_dataset = Dataset(data=val_subset, transform=data_loader.get_transforms(augment=False))
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
        )

        ModelClass = dynamic_import(config["model"])
        model = ModelClass(**config.get("model_args", {})).to(device)
        logger.info("Instantiated model %s", config["model"])

        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)

        best_split_model_state = None
        best_split_val_auc = -np.inf
        best_split_epoch = 0
        no_improve_count = 0
        split_results = []

        for epoch in range(1, args.epochs + 1):
            model.train()
            train_loss_accumulator = 0.0
            train_predictions = []
            train_labels = []
            train_probabilities = []

            for batch in train_loader:
                inputs = batch["image"].to(device)
                label_hot = batch["label"].to(device)
                label_class = torch.argmax(label_hot, dim=1)

                optimizer.zero_grad()
                outputs = model(inputs)
                if isinstance(outputs, (tuple, list)):
                    outputs = outputs[0]

                loss = criterion(outputs, label_class)
                loss.backward()
                optimizer.step()

                train_loss_accumulator += loss.item()
                train_predictions.append(torch.argmax(outputs, dim=1).cpu())
                train_labels.append(label_class.cpu())
                train_probabilities.append(torch.softmax(outputs, dim=1)[:, 1].detach().cpu())

            train_loss = train_loss_accumulator / max(len(train_loader), 1)
            train_labels_np = torch.cat(train_labels).numpy()
            train_predictions_np = torch.cat(train_predictions).numpy()
            train_probabilities_np = torch.cat(train_probabilities).numpy()

            try:
                train_auc = roc_auc_score(train_labels_np, train_probabilities_np)
            except ValueError:
                train_auc = np.nan
            train_f1 = f1_score(train_labels_np, train_predictions_np, average="binary", zero_division=0)

            model.eval()
            val_loss_accumulator = 0.0
            val_predictions = []
            val_labels = []
            val_probabilities = []

            with torch.no_grad():
                for batch in val_loader:
                    inputs = batch["image"].to(device)
                    label_hot = batch["label"].to(device)
                    label_class = torch.argmax(label_hot, dim=1)

                    outputs = model(inputs)
                    if isinstance(outputs, (tuple, list)):
                        outputs = outputs[0]

                    loss = criterion(outputs, label_class)
                    val_loss_accumulator += loss.item()

                    val_probabilities.append(torch.softmax(outputs, dim=1)[:, 1].cpu())
                    val_predictions.append(torch.argmax(outputs, dim=1).cpu())
                    val_labels.append(label_class.cpu())

            val_loss = val_loss_accumulator / max(len(val_loader), 1)
            val_labels_np = torch.cat(val_labels).numpy()
            val_predictions_np = torch.cat(val_predictions).numpy()
            val_probabilities_np = torch.cat(val_probabilities).numpy()
            validation_metrics = calculate_epoch_metrics(
                y_true=val_labels_np,
                y_pred=val_predictions_np,
                y_prob=val_probabilities_np,
            )

            logger.info(
                "Split %s | Epoch %s/%s | Train loss=%.4f | Train AUC=%.4f | Train F1=%.4f | "
                "Validation loss=%.4f | Validation AUC=%.4f | Validation MCC=%.4f | "
                "Validation Kappa=%.4f | Validation F1(binary)=%.4f | Validation F1(macro)=%.4f | "
                "Validation accuracy=%.4f | Validation sensitivity=%.4f | Validation specificity=%.4f | "
                "Validation PPV=%.4f | Validation NPV=%.4f | Validation balanced accuracy=%.4f",
                split_index,
                epoch,
                args.epochs,
                train_loss,
                train_auc,
                train_f1,
                val_loss,
                validation_metrics["auc"],
                validation_metrics["mcc"],
                validation_metrics["kappa"],
                validation_metrics["f1_binary"],
                validation_metrics["f1_macro"],
                validation_metrics["accuracy"],
                validation_metrics["sensitivity"],
                validation_metrics["specificity"],
                validation_metrics["ppv"],
                validation_metrics["npv"],
                validation_metrics["balanced_accuracy"],
            )

            split_results.append(
                {
                    "split": split_index,
                    "epoch": epoch,
                    "train_loss": train_loss,
                    "train_auc": train_auc,
                    "train_f1": train_f1,
                    "val_loss": val_loss,
                    "val_auc": validation_metrics["auc"],
                    "val_mcc": validation_metrics["mcc"],
                    "val_kappa": validation_metrics["kappa"],
                    "val_f1_binary": validation_metrics["f1_binary"],
                    "val_f1_macro": validation_metrics["f1_macro"],
                    "val_accuracy": validation_metrics["accuracy"],
                    "val_sensitivity": validation_metrics["sensitivity"],
                    "val_specificity": validation_metrics["specificity"],
                    "val_ppv": validation_metrics["ppv"],
                    "val_npv": validation_metrics["npv"],
                    "val_balanced_accuracy": validation_metrics["balanced_accuracy"],
                    "per_class_precision": validation_metrics["per_class_precision"],
                    "per_class_recall": validation_metrics["per_class_recall"],
                    "per_class_f1": validation_metrics["per_class_f1"],
                    "per_class_accuracy": validation_metrics["per_class_accuracy"],
                }
            )

            if validation_metrics["auc"] > best_split_val_auc:
                best_split_val_auc = validation_metrics["auc"]
                best_split_epoch = epoch
                best_split_model_state = copy.deepcopy(model.state_dict())
                no_improve_count = 0
            else:
                no_improve_count += 1

            if no_improve_count >= args.patience:
                logger.info(
                    "Early stopping triggered for split %s at epoch %s after %s epochs without validation AUC improvement.",
                    split_index,
                    epoch,
                    args.patience,
                )
                break

        split_results_path = results_dir / f"split_{split_index}_results.csv"
        pd.DataFrame(split_results).to_csv(split_results_path, index=False)
        logger.info("Saved split metrics to %s", split_results_path)

        split_model_path = models_dir / f"best_model_split_{split_index}.pth"
        torch.save(best_split_model_state, split_model_path)
        logger.info(
            "Saved best checkpoint for split %s (epoch %s, validation AUC %.4f) to %s",
            split_index,
            best_split_epoch,
            best_split_val_auc,
            split_model_path,
        )

        if best_split_val_auc > best_overall_score:
            best_overall_score = best_split_val_auc
            best_split_info = {"split": split_index, "epoch": best_split_epoch}
            best_overall_model = copy.deepcopy(best_split_model_state)

    if best_overall_model is None:
        raise RuntimeError("Training finished without producing a valid checkpoint.")

    overall_model_path = models_dir / "best_overall_model.pth"
    torch.save(best_overall_model, overall_model_path)
    logger.info(
        "Saved the overall best checkpoint from split %s epoch %s with validation AUC %.4f to %s",
        best_split_info["split"],
        best_split_info["epoch"],
        best_overall_score,
        overall_model_path,
    )

    metadata_path = metadata_dir / f"{args.config_key}_{args.mode}_training_metadata.json"
    metadata = {
        "config_key": args.config_key,
        "config_file": str(config_path),
        "mode": args.mode,
        "csv_path": str(csv_path),
        "input_shape": input_shape,
        "epochs": args.epochs,
        "n_splits": args.n_splits,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "learning_rate": args.learning_rate,
        "patience": args.patience,
        "seed": args.seed,
        "split_file": str(split_file),
        "best_overall_validation_auc": best_overall_score,
        "best_overall_split": best_split_info["split"],
        "best_overall_epoch": best_split_info["epoch"],
    }
    with metadata_path.open("w", encoding="utf-8") as file_handle:
        json.dump(metadata, file_handle, indent=2)
    logger.info("Saved training metadata to %s", metadata_path)


if __name__ == "__main__":
    main()

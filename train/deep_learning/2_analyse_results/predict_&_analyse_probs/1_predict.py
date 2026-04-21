#!/usr/bin/env python
"""
Generate validation predictions for previously trained deep learning models.

This script reuses the persisted grouped validation splits created during training,
which guarantees that prediction and downstream analysis refer to the exact same
fold assignment used to train each checkpoint.
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
MODELING_DIR = PROJECT_ROOT / "train" / "deep_learning" / "1_modeling"
if str(MODELING_DIR) not in sys.path:
    sys.path.append(str(MODELING_DIR))
COMMON_DIR = PROJECT_ROOT / "train" / "common"
if str(COMMON_DIR) not in sys.path:
    sys.path.append(str(COMMON_DIR))

import pandas as pd
import torch
import torch.multiprocessing as mp
from monai.data import DataLoader, Dataset

from runtime_utils import load_or_create_splits, resolve_project_path, setup_logger

mp.set_sharing_strategy("file_system")


def dynamic_import(class_path: str):
    """Import a class from a fully qualified module path."""

    module_name, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def get_loader_class(mode: str):
    """Select the correct MONAI data loader for the requested input mode."""

    if mode == "full":
        loader_module = "data_loaders.data_loader_for_cv_org"
    else:
        loader_module = "data_loaders.data_loader_for_cv_roi"
    return dynamic_import(f"{loader_module}.MyDataLoader")


def main():
    """Generate validation predictions for every available trained checkpoint folder."""

    parser = argparse.ArgumentParser(description="Generate validation predictions for trained MONAI models.")
    parser.add_argument(
        "--mode",
        type=str,
        choices=["gland", "full"],
        default="gland",
        help="Input mode used during training.",
    )
    parser.add_argument(
        "--data_root",
        type=str,
        default="artifacts/deep_learning",
        help="Root directory that contains trained model folders.",
    )
    parser.add_argument(
        "--config_file",
        type=str,
        default="train/deep_learning/1_modeling/config.json",
        help="Path to the JSON configuration file.",
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
    parser.add_argument("--n_splits", type=int, default=5, help="Number of grouped validation folds.")
    parser.add_argument("--seed", type=int, default=42, help="Seed used to build persisted validation folds.")
    parser.add_argument(
        "--output_base",
        type=str,
        default="results/deep_learning/model_comparison/predict_and_analyse_probs",
        help="Base directory for the exported prediction CSV files.",
    )
    args = parser.parse_args()

    models_root = resolve_project_path(PROJECT_ROOT, Path(args.data_root) / args.mode / "models")
    output_dir = resolve_project_path(
        PROJECT_ROOT, Path(args.output_base) / f"{args.mode}_analysis" / "predictions"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger("deep_learning_predictions", output_dir / "generate_predictions.log")
    logger.info("Scanning model checkpoints in %s", models_root)

    config_path = resolve_project_path(PROJECT_ROOT, args.config_file)
    csv_path = resolve_project_path(PROJECT_ROOT, args.csv_path)
    with config_path.open("r", encoding="utf-8") as file_handle:
        configs = json.load(file_handle)

    LoaderClass = get_loader_class(args.mode)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_loader = LoaderClass(
        csv_path=str(csv_path),
        input_shape=tuple(args.input_shape),
        config={},
        transformations=[],
        num_classes=2,
    )
    all_data = data_loader.get_all_data()
    all_labels = [int(torch.argmax(item["label"]).item()) for item in all_data]
    patient_ids = [item["patient_id"] for item in all_data]

    split_file = resolve_project_path(
        PROJECT_ROOT,
        Path("artifacts") / "deep_learning" / args.mode / "splits" / f"{csv_path.stem}_{args.mode}_{args.n_splits}fold_seed{args.seed}.json",
    )
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
    logger.info("Using validation split definition from %s", split_file)

    model_folders = [folder for folder in models_root.iterdir() if folder.is_dir()]
    logger.info("Found %s model directories.", len(model_folders))

    for model_folder in model_folders:
        model_name = model_folder.name
        logger.info("Processing model directory %s", model_folder)

        if model_name not in configs:
            logger.warning("Skipping %s because it is missing from %s", model_name, config_path)
            continue

        config = configs[model_name]
        ModelClass = dynamic_import(config["model"])
        model_files = sorted(model_folder.glob("*.pth"))
        if not model_files:
            logger.warning("Skipping %s because no checkpoint files were found.", model_name)
            continue

        all_predictions = []
        for split_info in split_bundle["splits"]:
            split_index = split_info["split_index"]
            split_model_path = model_folder / f"best_model_split_{split_index}.pth"
            if not split_model_path.exists():
                logger.warning(
                    "Checkpoint for split %s was not found in %s. Skipping this split.",
                    split_index,
                    model_folder,
                )
                continue

            model = ModelClass(**config.get("model_args", {}))
            model.load_state_dict(torch.load(split_model_path, map_location=device))
            model = model.to(device)
            model.eval()

            validation_subset = [all_data[index] for index in split_info["validation_indices"]]
            validation_dataset = Dataset(
                data=validation_subset,
                transform=data_loader.get_transforms(augment=False),
            )
            validation_loader = DataLoader(validation_dataset, batch_size=2, num_workers=4, shuffle=False)

            split_predictions = []
            with torch.no_grad():
                for batch in validation_loader:
                    inputs = batch["image"].to(device)
                    labels = torch.argmax(batch["label"].to(device), dim=1)
                    outputs = model(inputs)
                    if isinstance(outputs, (tuple, list)):
                        outputs = outputs[0]

                    probabilities = torch.softmax(outputs, dim=1)
                    predictions = torch.argmax(outputs, dim=1)

                    patient_batch = batch["patient_id"]
                    study_batch = batch["study_id"]
                    for sample_index in range(inputs.size(0)):
                        prediction_entry = {
                            "split": split_index,
                            "model": model_name,
                            "patient_id": patient_batch[sample_index],
                            "study_id": study_batch[sample_index],
                            "true_label": labels[sample_index].item(),
                            "prediction": predictions[sample_index].item(),
                        }
                        for class_index in range(probabilities.size(1)):
                            prediction_entry[f"prob_class_{class_index}"] = probabilities[
                                sample_index, class_index
                            ].item()
                        split_predictions.append(prediction_entry)

            all_predictions.extend(split_predictions)
            logger.info(
                "Generated %s validation predictions for model %s on split %s.",
                len(split_predictions),
                model_name,
                split_index,
            )

        if all_predictions:
            output_file = output_dir / f"{model_name}_predictions.csv"
            pd.DataFrame(all_predictions).to_csv(output_file, index=False)
            logger.info("Saved prediction CSV to %s", output_file)

    logger.info("Prediction generation finished successfully.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""Build concatenated radiomics tables from already extracted modality-specific CSV files."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]


MODALITY_TO_FILENAME = {
    "t2": "features_t2_{mode}.csv",
    "adc": "features_adc_{mode}.csv",
    "dwi": "features_dwi_{mode}.csv",
}
METADATA_COLUMNS = ["patient_id", "study_id", "label"]


def ensure_directory(path: Path) -> Path:
    """Create a directory if it does not exist and return the path."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def setup_logger(logger_name: str, log_file: Path) -> logging.Logger:
    """Create a simple console/file logger for concatenation runs."""

    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    ensure_directory(log_file.parent)
    file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    return logger


def load_modality_table(
    radiomics_root: Path,
    modality: str,
    mode: str,
    keep_shape_from: str,
) -> pd.DataFrame:
    """Load one modality CSV, drop diagnostics, and prefix feature names."""

    input_path = radiomics_root / MODALITY_TO_FILENAME[modality].format(mode=mode)
    if not input_path.exists():
        raise FileNotFoundError(f"Missing modality table: {input_path}")

    df = pd.read_csv(input_path)
    missing_columns = [column for column in METADATA_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"Missing required columns in {input_path}: {missing_columns}")

    diagnostic_columns = [column for column in df.columns if column.startswith("diagnostics_")]
    optional_metadata = [column for column in ["mask_type"] if column in df.columns]
    radiomic_columns = [
        column
        for column in df.columns
        if column not in METADATA_COLUMNS + optional_metadata + diagnostic_columns
    ]

    if modality != keep_shape_from:
        radiomic_columns = [column for column in radiomic_columns if "_shape_" not in column]

    renamed_columns = {column: f"{modality}_{column}" for column in radiomic_columns}
    cleaned_df = df[METADATA_COLUMNS + radiomic_columns].rename(columns=renamed_columns)
    cleaned_df = cleaned_df.drop_duplicates(subset=METADATA_COLUMNS, keep="first")
    return cleaned_df


def build_concatenated_table(
    radiomics_root: Path,
    mode: str,
    keep_shape_from: str,
) -> pd.DataFrame:
    """Merge modality-specific tables into one clean modeling table."""

    merged_df: pd.DataFrame | None = None
    for modality in ["t2", "adc", "dwi"]:
        modality_df = load_modality_table(
            radiomics_root=radiomics_root,
            modality=modality,
            mode=mode,
            keep_shape_from=keep_shape_from,
        )
        if merged_df is None:
            merged_df = modality_df
        else:
            merged_df = merged_df.merge(
                modality_df,
                on=METADATA_COLUMNS,
                how="inner",
                validate="one_to_one",
            )

    if merged_df is None:
        raise RuntimeError("No modality tables were merged.")

    merged_df["sample_id"] = (
        merged_df["patient_id"].astype(str) + "_" + merged_df["study_id"].astype(str)
    )
    ordered_columns = ["sample_id", *METADATA_COLUMNS] + [
        column for column in merged_df.columns if column not in {"sample_id", *METADATA_COLUMNS}
    ]
    return merged_df[ordered_columns]


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description=(
            "Merge already extracted modality-specific radiomics CSV files into one modeling table."
        )
    )
    parser.add_argument(
        "--radiomics_root",
        type=str,
        default="artifacts/radiomics",
        help="Directory containing the extracted modality-specific radiomics CSV files.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["gland", "full"],
        default="gland",
        help="Spatial mode to concatenate.",
    )
    parser.add_argument(
        "--keep_shape_from",
        type=str,
        choices=["t2", "adc", "dwi"],
        default="t2",
        help="Reference modality used to keep shape features. Shape is dropped from the others.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="artifacts/radiomics/concatenated_data/features_all_gland.csv",
        help="Output CSV path for the concatenated feature table.",
    )
    return parser.parse_args()


def main() -> None:
    """Build and save the concatenated table."""

    args = parse_args()
    radiomics_root = (PROJECT_ROOT / args.radiomics_root).resolve()
    output_path = (PROJECT_ROOT / args.output).resolve()
    logger = setup_logger(
        logger_name="RadiomicsConcatenation",
        log_file=output_path.parent / f"build_{Path(args.output).stem}.log",
    )

    logger.info("Radiomics root: %s", radiomics_root)
    logger.info("Mode: %s", args.mode)
    logger.info("Shape reference modality: %s", args.keep_shape_from)

    concatenated_df = build_concatenated_table(
        radiomics_root=radiomics_root,
        mode=args.mode,
        keep_shape_from=args.keep_shape_from,
    )

    ensure_directory(output_path.parent)
    concatenated_df.to_csv(output_path, index=False)

    feature_columns = [
        column
        for column in concatenated_df.columns
        if column not in {"sample_id", *METADATA_COLUMNS}
    ]
    logger.info("Saved concatenated table to: %s", output_path)
    logger.info("Samples: %d", len(concatenated_df))
    logger.info("Feature columns: %d", len(feature_columns))


if __name__ == "__main__":
    main()

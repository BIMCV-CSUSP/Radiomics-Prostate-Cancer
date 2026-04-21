from __future__ import annotations

import json
import logging
import os
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import StratifiedGroupKFold


def find_project_root(start_path: str | Path) -> Path:
    """Return the repository root by walking upwards from a file or directory."""

    current_path = Path(start_path).resolve()
    if current_path.is_file():
        current_path = current_path.parent

    for candidate in [current_path, *current_path.parents]:
        if (candidate / "README.md").exists() and (candidate / "train").exists():
            return candidate

    raise FileNotFoundError(
        f"Could not infer the project root from '{start_path}'. "
        "Expected to find a parent containing both README.md and train/."
    )


def ensure_directory(path: str | Path) -> Path:
    """Create a directory if it does not exist and return it as a Path."""

    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def resolve_project_path(project_root: str | Path, path_value: str | Path) -> Path:
    """Resolve an absolute or project-relative path into an absolute Path."""

    candidate = Path(path_value)
    if candidate.is_absolute():
        return candidate
    return (Path(project_root) / candidate).resolve()


def infer_project_path_from_csv(csv_path: str | Path, relative_value: str | Path) -> Path:
    """Resolve a path stored inside a project CSV using the CSV location as anchor."""

    csv_path = Path(csv_path).resolve()
    project_root = find_project_root(csv_path)
    return resolve_project_path(project_root, relative_value)


def setup_logger(
    logger_name: str,
    log_file: str | Path | None = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Create a console/file logger with clean handlers and English formatting."""

    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    logger.propagate = False

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file is not None:
        log_file = Path(log_file)
        ensure_directory(log_file.parent)
        file_handler = logging.FileHandler(log_file, mode="w", encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def set_global_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy, and PyTorch for reproducible experiments."""

    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def make_grouped_splits(
    labels: list[int] | np.ndarray,
    groups: list[str] | np.ndarray,
    n_splits: int,
    seed: int,
) -> list[dict[str, list[int]]]:
    """Build grouped stratified splits and return JSON-serializable indices."""

    labels_array = np.asarray(labels)
    groups_array = np.asarray(groups)
    dummy_features = np.zeros(len(labels_array))

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )

    splits: list[dict[str, list[int]]] = []
    for split_index, (train_idx, val_idx) in enumerate(
        splitter.split(dummy_features, labels_array, groups=groups_array),
        start=1,
    ):
        splits.append(
            {
                "split_index": split_index,
                "train_indices": train_idx.tolist(),
                "validation_indices": val_idx.tolist(),
            }
        )
    return splits


def save_splits(split_file: str | Path, splits: list[dict[str, list[int]]], metadata: dict) -> None:
    """Persist split indices and metadata in a JSON file."""

    split_file = Path(split_file)
    ensure_directory(split_file.parent)
    payload = {"metadata": metadata, "splits": splits}
    with split_file.open("w", encoding="utf-8") as file_handle:
        json.dump(payload, file_handle, indent=2)


def load_splits(split_file: str | Path) -> dict:
    """Load persisted split information from disk."""

    with Path(split_file).open("r", encoding="utf-8") as file_handle:
        return json.load(file_handle)


def load_or_create_splits(
    split_file: str | Path,
    labels: list[int] | np.ndarray,
    groups: list[str] | np.ndarray,
    n_splits: int,
    seed: int,
    metadata: dict,
) -> dict:
    """Reuse persisted splits when available, otherwise create and save them."""

    split_file = Path(split_file)
    if split_file.exists():
        return load_splits(split_file)

    splits = make_grouped_splits(labels=labels, groups=groups, n_splits=n_splits, seed=seed)
    save_splits(split_file=split_file, splits=splits, metadata=metadata)
    return {"metadata": metadata, "splits": splits}

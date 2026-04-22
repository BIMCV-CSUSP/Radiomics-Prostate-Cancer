#!/usr/bin/env python
"""
Train and evaluate radiomics classifiers for clinically significant prostate cancer.

Key improvements in this version:
1. Feature selection is performed inside each training fold to avoid data leakage.
2. Feature selection now combines univariate ranking, FDR control, and correlation pruning.
3. Outputs, plots, comments, and log messages are written in English.
4. Paths and post-processing calls are resolved from the project root.
5. Fold-wise selected features are exported for reproducibility.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import joblib
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics
from sklearn.base import clone
from sklearn.ensemble import (
    AdaBoostClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
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
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from train.common.radiomics_utils import (
    prepare_numeric_radiomics_matrix,
    resolve_feature_table_path,
    select_radiomics_features,
)

mpl.use("Agg")
try:
    import scienceplots  # noqa: F401

    plt.style.use(["science", "grid"])
except ModuleNotFoundError:
    plt.style.use("default")

# SciencePlots may enable TeX rendering on some systems. Force Matplotlib's
# built-in text rendering so the pipeline works on clusters without LaTeX.
mpl.rcParams["text.usetex"] = False
plt.rcParams["text.usetex"] = False

DPI = 300
DEFAULT_BASE_RANDOM_STATE = 42
MODEL_NAME_TO_CLI_NAME = {
    "SVM": "SVM",
    "Logistic Regression": "LogisticRegression",
    "Random Forest": "RandomForest",
    "Extra Trees": "ExtraTrees",
    "Decision Tree": "DecisionTree",
    "Naive Bayes": "NaiveBayes",
    "KNN": "KNN",
    "Gradient Boosting": "GradientBoosting",
    "AdaBoost": "AdaBoost",
}


def configure_live_logging() -> None:
    """Enable line-buffered stdout/stderr so logs appear promptly in cluster job files."""

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)


def log_progress(message: str) -> None:
    """Print a timestamped progress message and flush immediately."""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} | INFO | {message}", flush=True)


def format_metric(value: float) -> str:
    """Format numeric metrics consistently for console logging."""

    return "nan" if pd.isna(value) else f"{value:.4f}"


def save_live_results_snapshot(df_results: pd.DataFrame, output_path: Path) -> None:
    """Persist an intermediate results table so progress is visible while the run is ongoing."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df_results.to_csv(output_path, index=False)


def parse_serialized_list(value):
    """Convert a CSV-stored Python-style list back into a Python list."""

    if isinstance(value, list):
        return value
    if pd.isna(value) or value == "":
        return []
    if isinstance(value, str):
        try:
            parsed_value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            return []
        return parsed_value if isinstance(parsed_value, list) else []
    return []


def make_safe_slug(value: str) -> str:
    """Convert a classifier name into a filesystem-friendly slug."""

    safe_value = "".join(character.lower() if character.isalnum() else "_" for character in value)
    while "__" in safe_value:
        safe_value = safe_value.replace("__", "_")
    return safe_value.strip("_")


def build_model_color_map(classifier_names: list[str]) -> dict[str, str]:
    """Assign a stable color to each classifier name."""

    ordered_names = list(dict.fromkeys(classifier_names))
    palette = sns.color_palette("colorblind", n_colors=max(1, len(ordered_names)))
    return {
        classifier_name: mpl.colors.to_hex(color_value)
        for classifier_name, color_value in zip(ordered_names, palette)
    }


def sanitize_experiment_name(value: str) -> str:
    """Normalize a user-provided experiment tag so it is safe for directory names."""

    safe_value = "".join(character.lower() if character.isalnum() else "_" for character in value)
    while "__" in safe_value:
        safe_value = safe_value.replace("__", "_")
    safe_value = safe_value.strip("_")
    return safe_value or "experiment"


def build_fold_plan_cache_key(
    *,
    data_path: Path,
    n_splits: int,
    n_repeats: int,
    base_random_state: int,
    feature_strategy: str,
    min_features: int,
    max_features_cap: int,
    samples_per_feature: int,
    minority_samples_per_feature: int,
    fdr_alpha: float,
    correlation_threshold: float,
) -> str:
    """Build a reproducible cache key for grouped folds and fold-wise feature selection."""

    stat_result = data_path.stat()
    payload = {
        "data_path": str(data_path.resolve()),
        "data_mtime_ns": stat_result.st_mtime_ns,
        "data_size": stat_result.st_size,
        "n_splits": n_splits,
        "n_repeats": n_repeats,
        "base_random_state": base_random_state,
        "feature_strategy": feature_strategy,
        "min_features": min_features,
        "max_features_cap": max_features_cap,
        "samples_per_feature": samples_per_feature,
        "minority_samples_per_feature": minority_samples_per_feature,
        "fdr_alpha": fdr_alpha,
        "correlation_threshold": correlation_threshold,
    }
    payload_json = json.dumps(payload, sort_keys=True)
    return hashlib.md5(payload_json.encode("utf-8")).hexdigest()


def load_cached_fold_plan(cache_path: Path) -> tuple[list[dict], list[dict]]:
    """Load a previously saved grouped fold plan and feature-selection payload."""

    cached_payload = joblib.load(cache_path)
    return cached_payload["fold_plan"], cached_payload["selection_records"]


def save_cached_fold_plan(cache_path: Path, fold_plan: list[dict], selection_records: list[dict]) -> None:
    """Persist a grouped fold plan so future runs can skip fold-wise feature selection."""

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "fold_plan": fold_plan,
            "selection_records": selection_records,
        },
        cache_path,
        compress=3,
    )


def summarize_fold_plan(fold_plan: list[dict], sample_ids: np.ndarray) -> pd.DataFrame:
    """Create a compact summary of the grouped folds, including validation-set signatures."""

    summary_rows = []
    for fold_info in fold_plan:
        train_sample_ids = sample_ids[fold_info["train_idx"]]
        val_sample_ids = sample_ids[fold_info["val_idx"]]
        summary_rows.append(
            {
                "Fold": fold_info["fold_index"],
                "Repeat": fold_info["Repeat"],
                "fold_in_repeat": fold_info["fold_in_repeat"],
                "n_train": len(fold_info["train_idx"]),
                "n_val": len(fold_info["val_idx"]),
                "num_selected_features": len(fold_info["selected_features"]),
                "train_signature": hashlib.md5(
                    "||".join(map(str, train_sample_ids)).encode("utf-8")
                ).hexdigest(),
                "val_signature": hashlib.md5(
                    "||".join(map(str, val_sample_ids)).encode("utf-8")
                ).hexdigest(),
            }
        )
    return pd.DataFrame(summary_rows)


def get_models(random_state: int = 42):
    """Build the classifier pipelines used in the radiomics comparison."""

    base_steps = [SimpleImputer(strategy="median"), StandardScaler(), VarianceThreshold()]

    models = [
        (
            "SVM",
            make_pipeline(
                *base_steps,
                SVC(random_state=random_state, class_weight="balanced", probability=True),
            ),
        ),
        (
            "Logistic Regression",
            make_pipeline(
                *base_steps,
                LogisticRegression(
                    penalty="elasticnet",
                    l1_ratio=0.5,
                    class_weight="balanced",
                    random_state=random_state,
                    solver="saga",
                    max_iter=10000,
                ),
            ),
        ),
        (
            "Random Forest",
            make_pipeline(
                *base_steps,
                RandomForestClassifier(
                    n_jobs=-1,
                    class_weight="balanced_subsample",
                    random_state=random_state,
                ),
            ),
        ),
        (
            "Extra Trees",
            make_pipeline(
                *base_steps,
                ExtraTreesClassifier(
                    n_jobs=-1,
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ),
        (
            "Decision Tree",
            make_pipeline(
                *base_steps,
                DecisionTreeClassifier(
                    class_weight="balanced",
                    random_state=random_state,
                ),
            ),
        ),
        ("Naive Bayes", make_pipeline(*base_steps, GaussianNB())),
        ("KNN", make_pipeline(*base_steps, KNeighborsClassifier(n_jobs=-1))),
        (
            "Gradient Boosting",
            make_pipeline(*base_steps, GradientBoostingClassifier(random_state=random_state)),
        ),
        (
            "AdaBoost",
            make_pipeline(*base_steps, AdaBoostClassifier(random_state=random_state)),
        ),
    ]
    return models


def build_cv_fold_plan(
    X: pd.DataFrame,
    y: np.ndarray,
    groups: np.ndarray,
    n_splits: int = 5,
    n_repeats: int = 1,
    base_random_state: int = 42,
    feature_strategy: str = "all",
    min_features: int = 10,
    max_features_cap: int = 60,
    samples_per_feature: int = 25,
    minority_samples_per_feature: int = 8,
    fdr_alpha: float = 0.05,
    correlation_threshold: float = 0.90,
    selection_n_jobs: int | None = None,
) -> tuple[list[dict], list[dict]]:
    """Precompute grouped CV splits and training-only feature subsets once for reuse across models."""

    fold_plan = []
    selection_records = []
    total_folds = n_splits * n_repeats
    global_fold_index = 0

    for repeat_index in range(1, n_repeats + 1):
        log_progress(
            f"Preparing fold plan | repeat {repeat_index}/{n_repeats} "
            f"with grouped {n_splits}-fold CV."
        )
        current_random_state = base_random_state + repeat_index - 1
        splitter = StratifiedGroupKFold(
            n_splits=n_splits,
            shuffle=True,
            random_state=current_random_state,
        )

        for train_idx, val_idx in splitter.split(X, y, groups=groups):
            global_fold_index += 1
            fold_in_repeat = ((global_fold_index - 1) % n_splits) + 1
            X_train_raw = X.iloc[train_idx].copy()
            y_train = y[train_idx]

            train_positive = int(np.sum(y_train == 1))
            val_positive = int(np.sum(y[val_idx] == 1))
            log_progress(
                f"Preparing fold {global_fold_index}/{total_folds} "
                f"(repeat {repeat_index}/{n_repeats}, local fold {fold_in_repeat}/{n_splits}) "
                f"| train n={len(train_idx)} pos={train_positive} "
                f"| val n={len(val_idx)} pos={val_positive}"
            )

            if feature_strategy == "most_discriminant":
                selected_features, selection_df, selection_metadata = select_radiomics_features(
                    X_train=X_train_raw,
                    y_train=y_train,
                    repeat_index=repeat_index,
                    fold_index=global_fold_index,
                    min_features=min_features,
                    max_features_cap=max_features_cap,
                    samples_per_feature=samples_per_feature,
                    minority_samples_per_feature=minority_samples_per_feature,
                    fdr_alpha=fdr_alpha,
                    correlation_threshold=correlation_threshold,
                    n_jobs=selection_n_jobs,
                )
                selection_records.extend(
                    {
                        **record,
                        **selection_metadata,
                    }
                    for record in selection_df.to_dict(orient="records")
                )
                log_progress(
                    f"Prepared fold {global_fold_index}/{total_folds} "
                    f"| selected {len(selected_features)} features "
                    f"(FDR candidates={selection_metadata['n_fdr_features']}, "
                    f"pruned pool={selection_metadata['n_pruned_features']}, "
                    f"cap={selection_metadata['feature_limit']})"
                )
            else:
                selected_features = X_train_raw.columns.tolist()
                selection_metadata = {
                    "feature_limit": len(selected_features),
                    "n_valid_features": len(selected_features),
                    "n_fdr_features": len(selected_features),
                    "n_candidate_features": len(selected_features),
                    "n_pruned_features": len(selected_features),
                    "correlation_threshold": correlation_threshold,
                    "fdr_alpha": fdr_alpha,
                    "selection_n_jobs": selection_n_jobs,
                }
                log_progress(
                    f"Prepared fold {global_fold_index}/{total_folds} "
                    f"| using all {len(selected_features)} features."
                )

            fold_plan.append(
                {
                    "fold_index": global_fold_index,
                    "Repeat": repeat_index,
                    "fold_in_repeat": fold_in_repeat,
                    "train_idx": train_idx,
                    "val_idx": val_idx,
                    "selected_features": selected_features,
                    "selection_metadata": selection_metadata,
                }
            )

    return fold_plan, selection_records


def evaluate_model(
    model,
    classifier_name: str,
    X: pd.DataFrame,
    y: np.ndarray,
    sample_ids: np.ndarray,
    patient_ids: np.ndarray,
    study_ids: np.ndarray,
    fold_plan: list[dict],
    on_fold_complete=None,
):
    """Run grouped repeated cross-validation over a precomputed fold plan."""

    fold_results = []
    folds_data = []
    total_folds = len(fold_plan)

    for fold_info in fold_plan:
        train_idx = fold_info["train_idx"]
        val_idx = fold_info["val_idx"]
        repeat_index = fold_info["Repeat"]
        global_fold_index = fold_info["fold_index"]
        fold_in_repeat = fold_info["fold_in_repeat"]
        selected_features = fold_info["selected_features"]

        fold_start_time = time.perf_counter()
        y_train, y_val = y[train_idx], y[val_idx]
        train_positive = int(np.sum(y_train == 1))
        val_positive = int(np.sum(y_val == 1))
        log_progress(
            f"{classifier_name} | Fold {global_fold_index}/{total_folds} "
            f"(repeat {repeat_index}, local fold {fold_in_repeat}) "
            f"| train n={len(train_idx)} pos={train_positive} "
            f"| val n={len(val_idx)} pos={val_positive} "
            f"| features={len(selected_features)}"
        )

        X_train = X.iloc[train_idx][selected_features].copy()
        X_val = X.iloc[val_idx][selected_features].copy()

        fold_model = clone(model)
        fold_model.fit(X_train, y_train)

        y_train_pred = fold_model.predict(X_train)
        y_val_pred = fold_model.predict(X_val)

        if hasattr(fold_model, "predict_proba"):
            y_train_prob = fold_model.predict_proba(X_train)[:, 1]
            y_val_prob = fold_model.predict_proba(X_val)[:, 1]
        elif hasattr(fold_model, "decision_function"):
            y_train_prob = fold_model.decision_function(X_train)
            y_val_prob = fold_model.decision_function(X_val)
        else:
            y_train_prob = None
            y_val_prob = None

        try:
            train_auc = roc_auc_score(y_train, y_train_prob) if y_train_prob is not None else np.nan
        except ValueError:
            train_auc = np.nan

        try:
            val_auc = roc_auc_score(y_val, y_val_prob) if y_val_prob is not None else np.nan
        except ValueError:
            val_auc = np.nan

        cm = confusion_matrix(y_val, y_val_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        per_class_precision = precision_score(y_val, y_val_pred, average=None, zero_division=0)
        per_class_recall = recall_score(y_val, y_val_pred, average=None, zero_division=0)
        per_class_f1 = f1_score(y_val, y_val_pred, average=None, zero_division=0)
        per_class_accuracy = []
        for row_index in range(len(cm)):
            row_sum = np.sum(cm[row_index, :])
            per_class_accuracy.append(cm[row_index, row_index] / row_sum if row_sum > 0 else np.nan)

        fold_results.append(
            {
                "Fold": global_fold_index,
                "Repeat": repeat_index,
                "train_auc": train_auc,
                "train_f1": f1_score(y_train, y_train_pred, average="binary", zero_division=0),
                "val_auc": val_auc,
                "val_mcc": matthews_corrcoef(y_val, y_val_pred),
                "val_kappa": cohen_kappa_score(y_val, y_val_pred),
                "val_f1_binary": f1_score(y_val, y_val_pred, average="binary", zero_division=0),
                "val_f1_macro": f1_score(y_val, y_val_pred, average="macro", zero_division=0),
                "val_accuracy": accuracy_score(y_val, y_val_pred),
                "val_sensitivity": recall_score(y_val, y_val_pred, pos_label=1, zero_division=0),
                "val_specificity": recall_score(y_val, y_val_pred, pos_label=0, zero_division=0),
                "val_ppv": precision_score(y_val, y_val_pred, pos_label=1, zero_division=0),
                "val_npv": tn / (tn + fn) if (tn + fn) > 0 else np.nan,
                "val_balanced_accuracy": balanced_accuracy_score(y_val, y_val_pred),
                "num_selected_features": len(selected_features),
                "selected_features": selected_features,
                "per_class_precision": per_class_precision.tolist(),
                "per_class_recall": per_class_recall.tolist(),
                "per_class_f1": per_class_f1.tolist(),
                "per_class_accuracy": per_class_accuracy,
            }
        )
        completed_fold_result = fold_results[-1]
        elapsed_seconds = time.perf_counter() - fold_start_time
        running_results_df = pd.DataFrame(fold_results)
        running_auc_mean = running_results_df["val_auc"].mean()
        running_auc_median = running_results_df["val_auc"].median()
        running_bal_acc_mean = running_results_df["val_balanced_accuracy"].mean()
        log_progress(
            f"{classifier_name} | Completed fold {global_fold_index}/{total_folds} "
            f"in {elapsed_seconds:.1f}s | "
            f"val_auc={format_metric(completed_fold_result['val_auc'])} | "
            f"val_f1={format_metric(completed_fold_result['val_f1_binary'])} | "
            f"val_bal_acc={format_metric(completed_fold_result['val_balanced_accuracy'])} | "
            f"running_mean_auc={format_metric(running_auc_mean)} | "
            f"running_median_auc={format_metric(running_auc_median)} | "
            f"running_mean_bal_acc={format_metric(running_bal_acc_mean)}"
        )

        folds_data.append(
            {
                "fold_index": global_fold_index,
                "Repeat": repeat_index,
                "sample_ids": sample_ids[val_idx].tolist(),
                "patient_ids": patient_ids[val_idx].tolist(),
                "study_ids": study_ids[val_idx].tolist(),
                "y_val": y_val,
                "y_val_pred": y_val_pred,
                "y_val_prob": y_val_prob,
                "selected_features": selected_features,
            }
        )
        if on_fold_complete is not None:
            on_fold_complete(
                classifier_name=classifier_name,
                fold_result=completed_fold_result,
                fold_predictions=folds_data[-1],
                all_fold_results=fold_results,
                selection_records=None,
            )

    return fold_results, {"folds": folds_data}


def build_flat_prediction_table(predictions_data: list[dict]) -> pd.DataFrame:
    """Expand fold prediction bundles into one row per validation sample."""

    flat_rows = []
    for classifier_bundle in predictions_data:
        classifier_name = classifier_bundle["Classifier"]
        for fold_info in classifier_bundle["folds"]:
            probabilities = fold_info["y_val_prob"]
            for row_index, sample_id in enumerate(fold_info["sample_ids"]):
                probability_positive = (
                    float(probabilities[row_index]) if probabilities is not None else np.nan
                )
                flat_rows.append(
                    {
                        "Classifier": classifier_name,
                        "Fold": fold_info["fold_index"],
                        "Repeat": fold_info["Repeat"],
                        "sample_id": sample_id,
                        "patient_id": fold_info["patient_ids"][row_index],
                        "study_id": fold_info["study_ids"][row_index],
                        "true_label": int(fold_info["y_val"][row_index]),
                        "predicted_label": int(fold_info["y_val_pred"][row_index]),
                        "prob_class_1": probability_positive,
                        "selected_features": fold_info["selected_features"],
                    }
                )

    flat_df = pd.DataFrame(flat_rows)
    if flat_df.empty:
        return flat_df

    flat_df["selected_features_hash"] = flat_df["selected_features"].apply(
        lambda feature_list: hashlib.md5("||".join(feature_list).encode("utf-8")).hexdigest()
        if isinstance(feature_list, list)
        else np.nan
    )
    return flat_df


def aggregate_oof_predictions(flat_predictions_df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    """Average repeated out-of-fold predictions into a single row per case and classifier."""

    if flat_predictions_df.empty:
        return flat_predictions_df.copy()

    aggregated_df = (
        flat_predictions_df.groupby(
            ["Classifier", "sample_id", "patient_id", "study_id", "true_label"], as_index=False
        )
        .agg(
            prob_class_1=("prob_class_1", "mean"),
            num_validation_predictions=("prob_class_1", "size"),
            mean_predicted_label=("predicted_label", "mean"),
        )
        .sort_values(by=["Classifier", "sample_id"])
    )
    aggregated_df["predicted_label"] = (aggregated_df["prob_class_1"] >= threshold).astype(int)
    aggregated_df["classification_threshold"] = threshold
    return aggregated_df


def compute_binary_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5) -> dict:
    """Compute threshold-based and threshold-free binary classification metrics."""

    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    try:
        auc_value = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc_value = np.nan

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()
    return {
        "auc": auc_value,
        "accuracy": accuracy_score(y_true, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "f1_binary": f1_score(y_true, y_pred, average="binary", zero_division=0),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "kappa": cohen_kappa_score(y_true, y_pred),
        "sensitivity": recall_score(y_true, y_pred, pos_label=1, zero_division=0),
        "specificity": recall_score(y_true, y_pred, pos_label=0, zero_division=0),
        "ppv": precision_score(y_true, y_pred, pos_label=1, zero_division=0),
        "npv": tn / (tn + fn) if (tn + fn) > 0 else np.nan,
    }


def bootstrap_patient_level_performance(
    aggregated_predictions_df: pd.DataFrame,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    threshold: float = 0.5,
    seed: int = 42,
    n_roc_points: int = 200,
) -> dict:
    """Estimate patient-level metric confidence intervals with stratified bootstrap resampling."""

    if aggregated_predictions_df.empty:
        raise ValueError("Cannot bootstrap an empty prediction table.")

    patient_rows = {
        patient_id: patient_df.copy()
        for patient_id, patient_df in aggregated_predictions_df.groupby("patient_id")
    }
    patient_labels = (
        aggregated_predictions_df.groupby("patient_id")["true_label"].agg(lambda labels: int(labels.iloc[0]))
    )
    strata = {
        class_label: patient_labels[patient_labels == class_label].index.to_numpy()
        for class_label in sorted(patient_labels.unique())
    }

    point_metrics = compute_binary_metrics(
        y_true=aggregated_predictions_df["true_label"].to_numpy(),
        y_prob=aggregated_predictions_df["prob_class_1"].to_numpy(),
        threshold=threshold,
    )

    bootstrap_distributions = {metric_name: [] for metric_name in point_metrics}
    mean_fpr = np.linspace(0.0, 1.0, n_roc_points)
    tpr_samples = []
    random_generator = np.random.default_rng(seed)

    for _ in range(n_bootstrap):
        sampled_frames = []
        for class_label, patient_ids in strata.items():
            if len(patient_ids) == 0:
                continue
            sampled_patient_ids = random_generator.choice(
                patient_ids,
                size=len(patient_ids),
                replace=True,
            )
            for patient_id in sampled_patient_ids:
                sampled_frames.append(patient_rows[patient_id])

        if not sampled_frames:
            continue

        bootstrap_df = pd.concat(sampled_frames, ignore_index=True)
        if bootstrap_df["true_label"].nunique() < 2:
            continue

        bootstrap_metrics = compute_binary_metrics(
            y_true=bootstrap_df["true_label"].to_numpy(),
            y_prob=bootstrap_df["prob_class_1"].to_numpy(),
            threshold=threshold,
        )
        for metric_name, metric_value in bootstrap_metrics.items():
            bootstrap_distributions[metric_name].append(metric_value)

        fpr, tpr, _ = metrics.roc_curve(
            bootstrap_df["true_label"].to_numpy(),
            bootstrap_df["prob_class_1"].to_numpy(),
            pos_label=1,
        )
        interpolated_tpr = np.interp(mean_fpr, fpr, tpr)
        interpolated_tpr[0] = 0.0
        interpolated_tpr[-1] = 1.0
        tpr_samples.append(interpolated_tpr)

    alpha = 1.0 - ci_level
    ci_summary = {}
    for metric_name, metric_values in bootstrap_distributions.items():
        if metric_values:
            ci_summary[metric_name] = {
                "point_estimate": point_metrics[metric_name],
                "ci_low": float(np.nanpercentile(metric_values, 100 * (alpha / 2))),
                "ci_high": float(np.nanpercentile(metric_values, 100 * (1 - alpha / 2))),
                "n_bootstrap_success": len(metric_values),
            }
        else:
            ci_summary[metric_name] = {
                "point_estimate": point_metrics[metric_name],
                "ci_low": np.nan,
                "ci_high": np.nan,
                "n_bootstrap_success": 0,
            }

    tpr_matrix = np.vstack(tpr_samples) if tpr_samples else None
    roc_point_fpr, roc_point_tpr, _ = metrics.roc_curve(
        aggregated_predictions_df["true_label"].to_numpy(),
        aggregated_predictions_df["prob_class_1"].to_numpy(),
        pos_label=1,
    )
    roc_point_auc = metrics.auc(roc_point_fpr, roc_point_tpr)

    return {
        "metrics": ci_summary,
        "roc": {
            "point_fpr": roc_point_fpr,
            "point_tpr": roc_point_tpr,
            "point_auc": roc_point_auc,
            "grid_fpr": mean_fpr,
            "tpr_ci_low": np.nanpercentile(tpr_matrix, 100 * (alpha / 2), axis=0)
            if tpr_matrix is not None
            else None,
            "tpr_ci_high": np.nanpercentile(tpr_matrix, 100 * (1 - alpha / 2), axis=0)
            if tpr_matrix is not None
            else None,
        },
    }


def summarize_classifier_performance(
    df_results: pd.DataFrame,
    aggregated_predictions_df: pd.DataFrame,
    bootstrap_results: dict,
) -> pd.DataFrame:
    """Build a classifier-level summary table with fold statistics and OOF confidence intervals."""

    metrics_to_summarize = [
        "val_auc",
        "val_accuracy",
        "val_balanced_accuracy",
        "val_f1_binary",
        "val_mcc",
        "val_kappa",
        "val_sensitivity",
        "val_specificity",
        "val_ppv",
        "val_npv",
    ]

    summary_rows = []
    for classifier_name in sorted(df_results["Classifier"].unique()):
        classifier_results = df_results[df_results["Classifier"] == classifier_name]
        classifier_oof = aggregated_predictions_df[aggregated_predictions_df["Classifier"] == classifier_name]
        classifier_bootstrap = bootstrap_results[classifier_name]["metrics"]

        summary_row = {
            "Classifier": classifier_name,
            "n_fold_evaluations": int(len(classifier_results)),
            "n_unique_cases": int(len(classifier_oof)),
            "n_unique_patients": int(classifier_oof["patient_id"].nunique()),
        }

        for metric_name in metrics_to_summarize:
            metric_values = classifier_results[metric_name].dropna()
            summary_row[f"{metric_name}_mean"] = float(metric_values.mean()) if len(metric_values) else np.nan
            summary_row[f"{metric_name}_std"] = float(metric_values.std(ddof=1)) if len(metric_values) > 1 else np.nan
            summary_row[f"{metric_name}_median"] = float(metric_values.median()) if len(metric_values) else np.nan
            summary_row[f"{metric_name}_iqr"] = (
                float(metric_values.quantile(0.75) - metric_values.quantile(0.25))
                if len(metric_values)
                else np.nan
            )

        for metric_name, metric_payload in classifier_bootstrap.items():
            summary_row[f"oof_{metric_name}"] = metric_payload["point_estimate"]
            summary_row[f"oof_{metric_name}_ci_low"] = metric_payload["ci_low"]
            summary_row[f"oof_{metric_name}_ci_high"] = metric_payload["ci_high"]

        summary_rows.append(summary_row)

    return pd.DataFrame(summary_rows).sort_values(
        by=["oof_auc", "oof_auc_ci_low", "val_auc_median"],
        ascending=[False, False, False],
    )


def choose_best_classifier(summary_df: pd.DataFrame) -> pd.Series:
    """Select the best classifier using aggregated patient-level OOF AUC."""

    if summary_df.empty:
        raise ValueError("Cannot choose the best classifier from an empty summary table.")

    ranking_df = summary_df.sort_values(
        by=["oof_auc", "oof_auc_ci_low", "val_auc_median"],
        ascending=[False, False, False],
    ).reset_index(drop=True)
    return ranking_df.iloc[0]


def save_aggregated_performance_outputs(
    aggregated_predictions_df: pd.DataFrame,
    summary_df: pd.DataFrame,
    bootstrap_results: dict,
    output_dir: Path,
    ci_level: float,
) -> None:
    """Write publication-style aggregated OOF summaries, confidence intervals, and ROC plots."""

    output_dir.mkdir(parents=True, exist_ok=True)
    roc_dir = output_dir / "roc_curves"
    roc_dir.mkdir(parents=True, exist_ok=True)

    summary_path = output_dir / "summary_metrics.csv"
    summary_df.to_csv(summary_path, index=False)

    auc_summary_df = summary_df[
        [
            "Classifier",
            "oof_auc",
            "oof_auc_ci_low",
            "oof_auc_ci_high",
            "n_unique_cases",
            "n_unique_patients",
        ]
    ].copy()
    auc_summary_path = output_dir / "auc_ci_summary.csv"
    auc_summary_df.to_csv(auc_summary_path, index=False)

    auc_summary_text_path = output_dir / "auc_ci_summary.txt"
    with auc_summary_text_path.open("w", encoding="utf-8") as file_handle:
        file_handle.write(f"Patient-level out-of-fold AUC summary with {int(ci_level * 100)}% confidence intervals\n\n")
        for _, row in auc_summary_df.iterrows():
            file_handle.write(
                f"{row['Classifier']}: AUC={row['oof_auc']:.4f} "
                f"[{row['oof_auc_ci_low']:.4f}, {row['oof_auc_ci_high']:.4f}] | "
                f"cases={int(row['n_unique_cases'])} | patients={int(row['n_unique_patients'])}\n"
            )

    color_palette = build_model_color_map(summary_df["Classifier"].tolist())

    fig, axis = plt.subplots(figsize=(8, 6))
    for _, summary_row in summary_df.iterrows():
        classifier_name = summary_row["Classifier"]
        roc_payload = bootstrap_results[classifier_name]["roc"]
        axis.plot(
            roc_payload["point_fpr"],
            roc_payload["point_tpr"],
            label=(
                f"{classifier_name} "
                f"(AUC={summary_row['oof_auc']:.3f} "
                f"[{summary_row['oof_auc_ci_low']:.3f}, {summary_row['oof_auc_ci_high']:.3f}])"
            ),
            color=color_palette.get(classifier_name, "#333333"),
        )

    axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="_nolegend_")
    axis.set_xlabel("False Positive Rate", fontsize=12, labelpad=10)
    axis.set_ylabel("True Positive Rate", fontsize=12, labelpad=10)
    axis.tick_params(axis="both", which="major", labelsize=10)
    axis.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(roc_dir / "roc_aggregated.png", dpi=DPI, bbox_inches="tight")
    plt.close(fig)

    ci_plot_dir = roc_dir / "aggregated_with_ci"
    ci_plot_dir.mkdir(parents=True, exist_ok=True)
    for _, summary_row in summary_df.iterrows():
        classifier_name = summary_row["Classifier"]
        roc_payload = bootstrap_results[classifier_name]["roc"]
        safe_name = make_safe_slug(classifier_name)

        fig, axis = plt.subplots(figsize=(7, 6))
        axis.plot(
            roc_payload["point_fpr"],
            roc_payload["point_tpr"],
            color=color_palette.get(classifier_name, "#333333"),
            label=(
                f"{classifier_name} "
                f"(AUC={summary_row['oof_auc']:.3f} "
                f"[{summary_row['oof_auc_ci_low']:.3f}, {summary_row['oof_auc_ci_high']:.3f}])"
            ),
        )
        if roc_payload["tpr_ci_low"] is not None and roc_payload["tpr_ci_high"] is not None:
            axis.fill_between(
                roc_payload["grid_fpr"],
                roc_payload["tpr_ci_low"],
                roc_payload["tpr_ci_high"],
                color=color_palette.get(classifier_name, "#333333"),
                alpha=0.2,
                label=f"{int(ci_level * 100)}% bootstrap CI",
            )

        axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="_nolegend_")
        axis.set_xlabel("False Positive Rate", fontsize=12, labelpad=10)
        axis.set_ylabel("True Positive Rate", fontsize=12, labelpad=10)
        axis.tick_params(axis="both", which="major", labelsize=10)
        axis.legend(fontsize=9)
        fig.tight_layout()
        fig.savefig(ci_plot_dir / f"{safe_name}_roc_aggregated_ci.png", dpi=DPI, bbox_inches="tight")
        plt.close(fig)

        confusion_matrix_path = output_dir / f"{safe_name}_aggregated_confusion_matrix.csv"
        confusion_values = confusion_matrix(
            aggregated_predictions_df.loc[
                aggregated_predictions_df["Classifier"] == classifier_name, "true_label"
            ],
            aggregated_predictions_df.loc[
                aggregated_predictions_df["Classifier"] == classifier_name, "predicted_label"
            ],
            labels=[0, 1],
        )
        pd.DataFrame(
            confusion_values,
            index=["true_0", "true_1"],
            columns=["pred_0", "pred_1"],
        ).to_csv(confusion_matrix_path)


def save_roc_plots(df_results: pd.DataFrame, df_predictions: pd.DataFrame, roc_dir: Path) -> None:
    """Generate ROC plots for the best and median fold of each classifier."""

    roc_dir.mkdir(parents=True, exist_ok=True)

    optimal_curves = []
    median_curves = []
    classifiers = df_results["Classifier"].unique()

    for classifier_name in classifiers:
        df_classifier = df_results[df_results["Classifier"] == classifier_name]
        best_fold_row = df_classifier["val_auc"].idxmax()
        best_fold_number = df_classifier.loc[best_fold_row, "Fold"]

        median_auc = df_classifier["val_auc"].median()
        median_fold_row = (df_classifier["val_auc"] - median_auc).abs().idxmin()
        median_fold_number = df_classifier.loc[median_fold_row, "Fold"]

        best_prediction_row = df_predictions[
            (df_predictions["Classifier"] == classifier_name)
            & (df_predictions["Fold"] == best_fold_number)
        ]
        if len(best_prediction_row) > 0 and best_prediction_row.iloc[0]["y_prob"]:
            fpr, tpr, _ = metrics.roc_curve(
                best_prediction_row.iloc[0]["y_val"],
                best_prediction_row.iloc[0]["y_prob"],
                pos_label=1,
            )
            optimal_curves.append(
                {
                    "classifier": classifier_name,
                    "fold": best_fold_number,
                    "fpr": fpr,
                    "tpr": tpr,
                    "auc": metrics.auc(fpr, tpr),
                }
            )

        median_prediction_row = df_predictions[
            (df_predictions["Classifier"] == classifier_name)
            & (df_predictions["Fold"] == median_fold_number)
        ]
        if len(median_prediction_row) > 0 and median_prediction_row.iloc[0]["y_prob"]:
            fpr, tpr, _ = metrics.roc_curve(
                median_prediction_row.iloc[0]["y_val"],
                median_prediction_row.iloc[0]["y_prob"],
                pos_label=1,
            )
            median_curves.append(
                {
                    "classifier": classifier_name,
                    "fold": median_fold_number,
                    "fpr": fpr,
                    "tpr": tpr,
                    "auc": metrics.auc(fpr, tpr),
                }
            )

    optimal_curves.sort(key=lambda item: item["auc"], reverse=True)
    median_curves.sort(key=lambda item: item["auc"], reverse=True)

    color_palette = build_model_color_map(df_results["Classifier"].tolist())

    for figure_name, curves in [
        ("roc_best_folds.png", optimal_curves),
        ("roc_median_folds.png", median_curves),
    ]:
        fig, axis = plt.subplots(figsize=(8, 6))
        for curve in curves:
            axis.plot(
                curve["fpr"],
                curve["tpr"],
                label=f"{curve['classifier']} (Fold={curve['fold']}, AUC={curve['auc']:.3f})",
                color=color_palette.get(curve["classifier"], "#333333"),
            )

        axis.plot([0, 1], [0, 1], linestyle="--", color="gray", label="_nolegend_")
        axis.set_xlabel("False Positive Rate", fontsize=12, labelpad=10)
        axis.set_ylabel("True Positive Rate", fontsize=12, labelpad=10)
        axis.tick_params(axis="both", which="major", labelsize=10)
        axis.legend(fontsize=10)

        legend = axis.get_legend()
        for legend_line in legend.get_lines():
            legend_line.set_linewidth(2.5)

        fig.tight_layout()
        fig.savefig(roc_dir / figure_name, dpi=DPI, bbox_inches="tight")
        plt.close(fig)


def main():
    """Entry point for fold-aware radiomics model training and evaluation."""

    configure_live_logging()

    parser = argparse.ArgumentParser(
        description="Evaluate radiomics classifiers with repeated grouped cross-validation."
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="features_all_gland.csv",
        help="Feature CSV to evaluate.",
    )
    parser.add_argument(
        "--data_pre",
        type=str,
        default="artifacts/radiomics",
        help="Root directory containing the radiomics feature tables.",
    )
    parser.add_argument(
        "--results_base",
        type=str,
        default="results/radiomics",
        help="Base directory where outputs will be written.",
    )
    parser.add_argument(
        "--experiment_name",
        type=str,
        default=None,
        help=(
            "Optional extra directory level used to keep a run separate from previous experiments. "
            "Useful when comparing configurations without overwriting prior results."
        ),
    )
    parser.add_argument(
        "--n_splits",
        type=int,
        default=5,
        help="Number of grouped folds per repeat.",
    )
    parser.add_argument(
        "--n_repeats",
        type=int,
        default=10,
        help="Number of repeated grouped cross-validation rounds.",
    )
    parser.add_argument(
        "--feature_strategy",
        type=str,
        choices=["all", "most_discriminant"],
        default="most_discriminant",
        help="Use all features or perform leakage-safe fold-wise feature selection.",
    )
    parser.add_argument(
        "--calculate_differences",
        action="store_true",
        help="Run the post-hoc classifier comparison script after training.",
    )
    parser.add_argument(
        "--fine_tune_best_model",
        action="store_true",
        help="Run the fine-tuning script for the best classifier after training.",
    )
    parser.add_argument(
        "--bootstrap_iterations",
        type=int,
        default=1000,
        help="Number of patient-level bootstrap iterations used for confidence intervals.",
    )
    parser.add_argument(
        "--ci_level",
        type=float,
        default=0.95,
        help="Confidence level used for bootstrap confidence intervals.",
    )
    parser.add_argument(
        "--classification_threshold",
        type=float,
        default=0.5,
        help="Probability threshold used for aggregated threshold-based OOF metrics.",
    )
    parser.add_argument(
        "--min_features",
        type=int,
        default=10,
        help="Minimum number of radiomics features kept after fold-wise feature selection.",
    )
    parser.add_argument(
        "--max_features_cap",
        type=int,
        default=60,
        help="Upper cap for the automatically inferred number of selected radiomics features.",
    )
    parser.add_argument(
        "--samples_per_feature",
        type=int,
        default=25,
        help="Target number of training samples per selected feature used to infer the feature cap.",
    )
    parser.add_argument(
        "--minority_samples_per_feature",
        type=int,
        default=8,
        help="Target number of minority-class samples per selected feature used to infer the feature cap.",
    )
    parser.add_argument(
        "--fdr_alpha",
        type=float,
        default=0.05,
        help="False-discovery-rate alpha used before correlation pruning.",
    )
    parser.add_argument(
        "--correlation_threshold",
        type=float,
        default=0.90,
        help="Absolute Pearson-correlation threshold used to prune redundant features.",
    )
    parser.add_argument(
        "--selection_n_jobs",
        type=int,
        default=-1,
        help=(
            "Number of parallel workers used during fold-wise univariate feature scoring. "
            "Increase this on multi-core servers to speed up feature selection."
        ),
    )
    parser.add_argument(
        "--selection_cache_dir",
        type=str,
        default="results/radiomics/shared_cache",
        help=(
            "Shared cache directory used to store grouped fold plans and fold-wise feature selection. "
            "Runs with the same dataset and selection settings can reuse this cache."
        ),
    )
    parser.add_argument(
        "--refresh_selection_cache",
        action="store_true",
        help="Ignore any saved grouped fold-plan cache and recompute fold-wise feature selection.",
    )
    parser.add_argument(
        "--search_iterations",
        type=int,
        default=50,
        help=(
            "Number of Bayesian optimization iterations to use when the automatic hold-out "
            "fine-tuning step is enabled."
        ),
    )
    parser.add_argument(
        "--search_n_jobs",
        type=int,
        default=-1,
        help=(
            "Number of parallel workers to use during the automatic hold-out fine-tuning step."
        ),
    )
    parser.add_argument(
        "--postprocess_only",
        action="store_true",
        help=(
            "Skip model training and resume from previously saved CSV outputs in the experiment "
            "directory. Useful after plot-only or post-processing failures."
        ),
    )
    args = parser.parse_args()

    data_root = (PROJECT_ROOT / args.data_pre).resolve()
    data_path = resolve_feature_table_path(
        project_root=PROJECT_ROOT,
        data_root=data_root,
        csv_argument=args.csv,
    )

    df = pd.read_csv(data_path)
    if "sample_id" in df.columns:
        df = df.set_index("sample_id")
    else:
        df["patient_id_study_id"] = df["patient_id"].astype(str) + "_" + df["study_id"].astype(str)
        df = df.set_index("patient_id_study_id")

    sample_ids = df.index.to_numpy()
    patient_ids = df["patient_id"].to_numpy()
    study_ids = df["study_id"].to_numpy()
    y = LabelEncoder().fit_transform(df["label"].values)
    groups = patient_ids
    X = prepare_numeric_radiomics_matrix(df)
    X = X.replace([np.inf, -np.inf], np.nan)

    base_dir = (PROJECT_ROOT / args.results_base / args.feature_strategy).resolve()
    csv_stem = data_path.stem
    mode = csv_stem.rsplit("_", 1)[-1]
    experiment_dir = base_dir / mode
    if args.experiment_name:
        experiment_dir = experiment_dir / sanitize_experiment_name(args.experiment_name)
    experiment_dir.mkdir(parents=True, exist_ok=True)
    results_filename = f"results_{csv_stem}_{args.feature_strategy}.csv"
    results_path = experiment_dir / results_filename
    predictions_filename = f"predictions_{csv_stem}_{args.feature_strategy}.csv"
    predictions_path = experiment_dir / predictions_filename
    flat_predictions_path = experiment_dir / f"oof_predictions_flat_{csv_stem}_{args.feature_strategy}.csv"
    aggregated_predictions_path = (
        experiment_dir / f"oof_predictions_aggregated_{csv_stem}_{args.feature_strategy}.csv"
    )

    log_progress(f"Results directory: {experiment_dir}")
    if args.postprocess_only:
        log_progress("Postprocess-only mode enabled. Loading saved outputs from disk...")
        required_paths = [
            results_path,
            predictions_path,
            flat_predictions_path,
            aggregated_predictions_path,
        ]
        missing_paths = [path for path in required_paths if not path.exists()]
        if missing_paths:
            missing_text = "\n".join(f"  - {path}" for path in missing_paths)
            raise FileNotFoundError(
                "Postprocess-only mode requires previously saved outputs, but the following files "
                f"are missing:\n{missing_text}"
            )

        df_results = pd.read_csv(results_path)
        df_predictions = pd.read_csv(
            predictions_path,
            converters={
                "sample_ids": parse_serialized_list,
                "patient_ids": parse_serialized_list,
                "study_ids": parse_serialized_list,
                "y_val": parse_serialized_list,
                "y_pred": parse_serialized_list,
                "y_prob": parse_serialized_list,
                "selected_features": parse_serialized_list,
            },
        )
        flat_predictions_df = pd.read_csv(
            flat_predictions_path,
            converters={"selected_features": parse_serialized_list},
        )
        aggregated_predictions_df = pd.read_csv(aggregated_predictions_path)
        log_progress(
            f"Loaded saved outputs | fold_metrics={len(df_results)} rows | "
            f"fold_predictions={len(df_predictions)} rows | "
            f"flat_oof={len(flat_predictions_df)} rows | "
            f"aggregated_oof={len(aggregated_predictions_df)} rows"
        )
    else:
        log_progress(f"Loaded {X.shape[0]} samples and {X.shape[1]} numeric features.")
        log_progress(
            f"Feature strategy: {args.feature_strategy} | "
            f"n_splits={args.n_splits} | n_repeats={args.n_repeats} | "
            f"bootstrap_iterations={args.bootstrap_iterations}"
        )
        log_progress(
            "Feature selection settings: "
            f"min_features={args.min_features}, max_features_cap={args.max_features_cap}, "
            f"samples_per_feature={args.samples_per_feature}, "
            f"minority_samples_per_feature={args.minority_samples_per_feature}, "
            f"fdr_alpha={args.fdr_alpha}, correlation_threshold={args.correlation_threshold}, "
            f"selection_n_jobs={args.selection_n_jobs}"
        )
        if args.experiment_name:
            log_progress(f"Experiment name: {sanitize_experiment_name(args.experiment_name)}")

        all_results = []
        predictions_data = []
        feature_selection_records = []
        models = get_models(random_state=DEFAULT_BASE_RANDOM_STATE)
        num_models = len(models)
        live_results_path = experiment_dir / f"results_live_{csv_stem}_{args.feature_strategy}.csv"
        log_progress(f"Live fold-metrics snapshot will be updated at: {live_results_path}")
        cache_root = (PROJECT_ROOT / args.selection_cache_dir).resolve()
        fold_plan_cache_key = build_fold_plan_cache_key(
            data_path=data_path,
            n_splits=args.n_splits,
            n_repeats=args.n_repeats,
            base_random_state=DEFAULT_BASE_RANDOM_STATE,
            feature_strategy=args.feature_strategy,
            min_features=args.min_features,
            max_features_cap=args.max_features_cap,
            samples_per_feature=args.samples_per_feature,
            minority_samples_per_feature=args.minority_samples_per_feature,
            fdr_alpha=args.fdr_alpha,
            correlation_threshold=args.correlation_threshold,
        )
        fold_plan_cache_path = cache_root / f"fold_plan_{fold_plan_cache_key}.joblib"
        if fold_plan_cache_path.exists() and not args.refresh_selection_cache:
            log_progress(f"Loading cached fold plan and feature selection from: {fold_plan_cache_path}")
            fold_plan, shared_selection_records = load_cached_fold_plan(fold_plan_cache_path)
        else:
            log_progress("Precomputing grouped CV folds and training-only feature subsets for reuse across models...")
            fold_plan, shared_selection_records = build_cv_fold_plan(
                X=X,
                y=y,
                groups=groups,
                n_splits=args.n_splits,
                n_repeats=args.n_repeats,
                base_random_state=DEFAULT_BASE_RANDOM_STATE,
                feature_strategy=args.feature_strategy,
                min_features=args.min_features,
                max_features_cap=args.max_features_cap,
                samples_per_feature=args.samples_per_feature,
                minority_samples_per_feature=args.minority_samples_per_feature,
                fdr_alpha=args.fdr_alpha,
                correlation_threshold=args.correlation_threshold,
                selection_n_jobs=args.selection_n_jobs,
            )
            save_cached_fold_plan(
                cache_path=fold_plan_cache_path,
                fold_plan=fold_plan,
                selection_records=shared_selection_records,
            )
            log_progress(f"Saved grouped fold-plan cache to: {fold_plan_cache_path}")

        fold_plan_summary_df = summarize_fold_plan(fold_plan=fold_plan, sample_ids=sample_ids)
        fold_plan_summary_path = experiment_dir / "fold_plan_summary.csv"
        fold_plan_summary_df.to_csv(fold_plan_summary_path, index=False)
        duplicate_validation_folds = int(
            len(fold_plan_summary_df) - fold_plan_summary_df["val_signature"].nunique()
        )
        log_progress(
            f"Prepared {len(fold_plan)} reusable folds. "
            f"Average selected features per fold: "
            f"{np.mean([len(item['selected_features']) for item in fold_plan]):.1f}"
        )
        log_progress(
            f"Fold-plan summary saved to: {fold_plan_summary_path} | "
            f"duplicate validation signatures across repeats={duplicate_validation_folds}"
        )

        for model_index, (model_name, model) in enumerate(models, start=1):
            log_progress(f"Starting model {model_index}/{num_models}: {model_name}")

            def on_fold_complete(**callback_payload):
                accumulated_rows = list(all_results) + [
                    {**row, "Classifier": model_name} for row in callback_payload["all_fold_results"]
                ]
                live_df = pd.DataFrame(accumulated_rows)
                save_live_results_snapshot(live_df, live_results_path)

            fold_metrics, prediction_bundle = evaluate_model(
                model=model,
                classifier_name=model_name,
                X=X,
                y=y,
                sample_ids=sample_ids,
                patient_ids=patient_ids,
                study_ids=study_ids,
                fold_plan=fold_plan,
                on_fold_complete=on_fold_complete,
            )

            for fold_metrics_row in fold_metrics:
                fold_metrics_row["Classifier"] = model_name
                all_results.append(fold_metrics_row)

            predictions_data.append({"Classifier": model_name, "folds": prediction_bundle["folds"]})
            for record in shared_selection_records:
                feature_selection_records.append({**record, "Classifier": model_name})

            model_results_df = pd.DataFrame(fold_metrics)
            log_progress(
                f"Completed model {model_index}/{num_models}: {model_name} | "
                f"mean_val_auc={format_metric(model_results_df['val_auc'].mean())} | "
                f"median_val_auc={format_metric(model_results_df['val_auc'].median())} | "
                f"mean_val_bal_acc={format_metric(model_results_df['val_balanced_accuracy'].mean())}"
            )

        df_results = pd.DataFrame(all_results)
        fixed_columns = ["Classifier", "Fold", "Repeat"]
        other_columns = [column for column in df_results.columns if column not in fixed_columns]
        df_results = df_results[fixed_columns + other_columns]
        df_results.sort_values(by=["Classifier", "Fold"], inplace=True)

        df_results.to_csv(results_path, index=False)
        log_progress(f"Saved fold metrics to: {results_path}")

        prediction_rows = []
        for item in predictions_data:
            classifier_name = item["Classifier"]
            for fold_info in item["folds"]:
                prediction_rows.append(
                    {
                        "Classifier": classifier_name,
                        "Fold": fold_info["fold_index"],
                        "Repeat": fold_info["Repeat"],
                        "sample_ids": fold_info["sample_ids"],
                        "patient_ids": fold_info["patient_ids"],
                        "study_ids": fold_info["study_ids"],
                        "y_val": fold_info["y_val"].tolist(),
                        "y_pred": fold_info["y_val_pred"].tolist(),
                        "y_prob": fold_info["y_val_prob"].tolist() if fold_info["y_val_prob"] is not None else [],
                        "selected_features": fold_info["selected_features"],
                    }
                )

        df_predictions = pd.DataFrame(prediction_rows)
        df_predictions.to_csv(predictions_path, index=False)
        log_progress(f"Saved fold predictions to: {predictions_path}")

        flat_predictions_df = build_flat_prediction_table(predictions_data)
        flat_predictions_df.to_csv(flat_predictions_path, index=False)
        log_progress(f"Saved flat OOF predictions to: {flat_predictions_path}")

        aggregated_predictions_df = aggregate_oof_predictions(
            flat_predictions_df=flat_predictions_df,
            threshold=args.classification_threshold,
        )
        aggregated_predictions_df.to_csv(aggregated_predictions_path, index=False)
        log_progress(f"Saved aggregated OOF predictions to: {aggregated_predictions_path}")

        if args.feature_strategy == "most_discriminant" and feature_selection_records:
            feature_selection_dir = experiment_dir / "feature_selection"
            feature_selection_dir.mkdir(parents=True, exist_ok=True)

            selection_df = pd.DataFrame(feature_selection_records)
            detailed_path = feature_selection_dir / "selected_features_by_fold.csv"
            selection_df.to_csv(detailed_path, index=False)

            summary_df = (
                selection_df[selection_df["is_selected"]]
                .groupby(["Classifier", "feature"])
                .agg(
                    times_selected=("is_selected", "sum"),
                    mean_auc=("auc", "mean"),
                    mean_p_value=("p_value", "mean"),
                    mean_q_value=("q_value", "mean"),
                )
                .reset_index()
                .sort_values(by=["Classifier", "times_selected", "mean_auc"], ascending=[True, False, False])
            )
            summary_path = feature_selection_dir / "feature_selection_frequency.csv"
            summary_df.to_csv(summary_path, index=False)

            top_features_path = feature_selection_dir / "top_selected_features.txt"
            with top_features_path.open("w", encoding="utf-8") as file_handle:
                for classifier_name, classifier_df in summary_df.groupby("Classifier"):
                    file_handle.write(f"{classifier_name}\n")
                    for _, row in classifier_df.head(20).iterrows():
                        file_handle.write(
                            f"  - {row['feature']} | selected {int(row['times_selected'])} times | "
                            f"mean AUC={row['mean_auc']:.4f} | mean p-value={row['mean_p_value']:.4e} | "
                            f"mean q-value={row['mean_q_value']:.4e}\n"
                        )
                    file_handle.write("\n")

            recommended_features_path = feature_selection_dir / "recommended_features_by_classifier.txt"
            with recommended_features_path.open("w", encoding="utf-8") as file_handle:
                for classifier_name, classifier_df in summary_df.groupby("Classifier"):
                    file_handle.write(f"{classifier_name}\n")
                    for feature_name in classifier_df.head(args.max_features_cap)["feature"].tolist():
                        file_handle.write(f"{feature_name}\n")
                    file_handle.write("\n")

            log_progress(f"Saved fold-wise feature selection details to: {feature_selection_dir}")
        else:
            variables_path = experiment_dir / "variables_used.txt"
            with variables_path.open("w", encoding="utf-8") as file_handle:
                for feature_name in X.columns:
                    file_handle.write(f"{feature_name}\n")
            log_progress(f"Saved feature list to: {variables_path}")

    roc_dir = experiment_dir / "roc_curves"
    save_roc_plots(df_results=df_results, df_predictions=df_predictions, roc_dir=roc_dir)
    log_progress(f"Saved ROC plots to: {roc_dir}")

    bootstrap_results = {}
    for classifier_name in aggregated_predictions_df["Classifier"].unique():
        log_progress(f"Bootstrapping aggregated patient-level confidence intervals for {classifier_name}...")
        classifier_df = aggregated_predictions_df[
            aggregated_predictions_df["Classifier"] == classifier_name
        ].copy()
        bootstrap_results[classifier_name] = bootstrap_patient_level_performance(
            aggregated_predictions_df=classifier_df,
            n_bootstrap=args.bootstrap_iterations,
            ci_level=args.ci_level,
            threshold=args.classification_threshold,
            seed=42,
        )
        auc_ci = bootstrap_results[classifier_name]["metrics"]["auc"]
        log_progress(
            f"{classifier_name} | aggregated patient-level AUC={format_metric(auc_ci['point_estimate'])} "
            f"| {int(args.ci_level * 100)}% CI=[{format_metric(auc_ci['ci_low'])}, {format_metric(auc_ci['ci_high'])}]"
        )

    summary_df = summarize_classifier_performance(
        df_results=df_results,
        aggregated_predictions_df=aggregated_predictions_df,
        bootstrap_results=bootstrap_results,
    )
    aggregated_output_dir = experiment_dir / "aggregated_performance"
    save_aggregated_performance_outputs(
        aggregated_predictions_df=aggregated_predictions_df,
        summary_df=summary_df,
        bootstrap_results=bootstrap_results,
        output_dir=aggregated_output_dir,
        ci_level=args.ci_level,
    )
    log_progress(f"Saved aggregated OOF summaries and confidence intervals to: {aggregated_output_dir}")

    if args.calculate_differences:
        log_progress("Running statistical comparison across classifiers...")
        model_diff_dir = experiment_dir / "model_differences"
        model_diff_dir.mkdir(parents=True, exist_ok=True)
        comparison_script = Path(__file__).resolve().parent / "2_model_differences.py"
        postprocess_cmd = [
            sys.executable,
            str(comparison_script),
            "--csv_preds",
            str(predictions_path),
            "--csv_results",
            str(results_path),
            "--metric",
            "val_auc",
            "--alpha",
            "0.05",
            "--outdir",
            str(model_diff_dir),
        ]
        subprocess.call(postprocess_cmd)
    else:
        log_progress("Skipping statistical comparison across classifiers.")

    if args.fine_tune_best_model:
        if len(df_results) == 0:
            log_progress("Skipping fine-tuning because no evaluation results were produced.")
            return

        best_model_row = choose_best_classifier(summary_df)
        best_model_name = best_model_row["Classifier"]
        best_model_cli_name = MODEL_NAME_TO_CLI_NAME[best_model_name]
        fine_tune_script = Path(__file__).resolve().parent / "3_retrain_best_model_and_evaluate.py"
        fine_tune_cmd = [
            sys.executable,
            str(fine_tune_script),
            "--csv",
            str(data_path),
            "--data_pre",
            str(data_root),
            "--results_base",
            str(args.results_base),
            "--feature_strategy",
            str(args.feature_strategy),
            "--model",
            best_model_cli_name,
            "--bootstrap_iterations",
            str(args.bootstrap_iterations),
            "--ci_level",
            str(args.ci_level),
            "--min_features",
            str(args.min_features),
            "--max_features_cap",
            str(args.max_features_cap),
            "--samples_per_feature",
            str(args.samples_per_feature),
            "--minority_samples_per_feature",
            str(args.minority_samples_per_feature),
            "--fdr_alpha",
            str(args.fdr_alpha),
            "--correlation_threshold",
            str(args.correlation_threshold),
            "--selection_n_jobs",
            str(args.selection_n_jobs),
            "--search_iterations",
            str(args.search_iterations),
            "--search_n_jobs",
            str(args.search_n_jobs),
        ]
        if args.experiment_name:
            fine_tune_cmd.extend(["--experiment_name", sanitize_experiment_name(args.experiment_name)])
        selection_summary_path = experiment_dir / "best_model_selection_summary.txt"
        with selection_summary_path.open("w", encoding="utf-8") as file_handle:
            file_handle.write("Best-model selection based on aggregated patient-level OOF AUC\n\n")
            for _, row in summary_df.sort_values(
                by=["oof_auc", "oof_auc_ci_low", "val_auc_median"],
                ascending=[False, False, False],
            ).iterrows():
                file_handle.write(
                    f"{row['Classifier']}: OOF AUC={row['oof_auc']:.4f} "
                    f"[{row['oof_auc_ci_low']:.4f}, {row['oof_auc_ci_high']:.4f}] | "
                    f"median fold AUC={row['val_auc_median']:.4f}\n"
                )
        log_progress(
            f"Running fine-tuning for the best classifier selected by aggregated patient-level OOF AUC: "
            f"{best_model_name} | OOF AUC={best_model_row['oof_auc']:.4f} "
            f"[{best_model_row['oof_auc_ci_low']:.4f}, {best_model_row['oof_auc_ci_high']:.4f}]"
        )
        subprocess.call(fine_tune_cmd)


if __name__ == "__main__":
    main()

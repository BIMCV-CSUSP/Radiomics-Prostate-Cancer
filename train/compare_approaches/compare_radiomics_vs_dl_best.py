#!/usr/bin/env python3
"""Compare the best radiomics model against the best deep learning model across paired folds."""

from __future__ import annotations

import argparse
import ast
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn import metrics

matplotlib.use("Agg")
try:
    import scienceplots  # noqa: F401

    plt.style.use(["science", "grid"])
except ModuleNotFoundError:
    plt.style.use("default")
DPI = 300
BOX_STYLE = dict(color="black")


def parse_list_column(series: pd.Series):
    """Convert list-like strings stored in CSV files into Python objects."""

    return series.apply(lambda value: ast.literal_eval(value) if isinstance(value, str) else value)


def load_radiomics_predictions(csv_path: Path, classifier: str) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Load fold-wise radiomics predictions for Repeat 1 and folds 1-5."""

    df = pd.read_csv(csv_path)
    df = df[(df["Classifier"] == classifier) & (df["Repeat"] == 1) & (df["Fold"] <= 5)]
    df["y_val"] = parse_list_column(df["y_val"])
    df["y_prob"] = parse_list_column(df["y_prob"])
    return {
        int(row["Fold"]): (np.array(row["y_val"]), np.array(row["y_prob"]))
        for _, row in df.iterrows()
    }


def load_deep_learning_predictions(csv_path: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Load fold-wise deep learning predictions, normalising split numbering to folds 1-5."""

    df = pd.read_csv(csv_path)
    min_split = df["split"].min()
    fold_dict = {}
    for split_id, group_df in df.groupby("split"):
        split_id = int(split_id)
        if min_split == 0:
            if split_id >= 5:
                continue
            fold_id = split_id + 1
        else:
            if split_id > 5:
                continue
            fold_id = split_id
        fold_dict[fold_id] = (group_df["true_label"].values, group_df["prob_class_1"].values)
    return fold_dict


def build_auc_vector(fold_dict: dict[int, tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """Compute the AUC of folds 1-5 in order."""

    auc_values = []
    for fold_id in range(1, 6):
        if fold_id not in fold_dict:
            raise ValueError(f"Missing fold {fold_id} in the prediction file.")
        y_true, y_prob = fold_dict[fold_id]
        auc_values.append(metrics.roc_auc_score(y_true, y_prob))
    return np.asarray(auc_values)


def paired_wilcoxon_and_effect_size(first: np.ndarray, second: np.ndarray):
    """Compute paired Wilcoxon statistics and paired Cohen's d."""

    statistic, p_value = wilcoxon(first, second, alternative="two-sided")
    paired_difference = first - second
    effect_size = (
        paired_difference.mean() / paired_difference.std(ddof=1)
        if paired_difference.std(ddof=1)
        else np.nan
    )
    return statistic, p_value, effect_size


def main():
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(
        description="Compare paired fold-wise AUC between radiomics and deep learning."
    )
    parser.add_argument(
        "--dl_preds_csv",
        type=Path,
        default=Path(
            "results/deep_learning/model_comparison/predict_and_analyse_probs/gland_analysis/predictions/config1_predictions.csv"
        ),
    )
    parser.add_argument(
        "--radiomics_preds_csv",
        type=Path,
        default=Path(
            "results/radiomics/most_discriminant/gland/predictions_features_all_gland_most_discriminant.csv"
        ),
    )
    parser.add_argument(
        "--radiomics_model",
        default="Logistic Regression",
        help="Classifier name exactly as stored in the radiomics prediction CSV.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("results/compare_best_radiomics_dl"),
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)

    deep_learning_auc = build_auc_vector(load_deep_learning_predictions(args.dl_preds_csv))
    radiomics_auc = build_auc_vector(
        load_radiomics_predictions(args.radiomics_preds_csv, args.radiomics_model)
    )

    statistic, p_value, effect_size = paired_wilcoxon_and_effect_size(
        deep_learning_auc, radiomics_auc
    )

    summary_lines = [
        f"Deep learning prediction CSV: {args.dl_preds_csv.name}",
        f"Radiomics classifier: {args.radiomics_model}",
        "",
        "Fold-wise AUC computed directly from the predicted probabilities",
    ]
    for fold_index in range(5):
        summary_lines.append(
            f"  Fold {fold_index + 1} - Deep learning: {deep_learning_auc[fold_index]:.3f} | "
            f"Radiomics: {radiomics_auc[fold_index]:.3f}"
        )
    summary_lines.extend(
        [
            "",
            f"Wilcoxon signed-rank test: statistic={statistic:.4f}, p-value={p_value:.4e}",
            f"Paired Cohen's d: {effect_size:.3f}",
            "",
            "Conclusion:",
        ]
    )

    if p_value < args.alpha:
        winner = (
            "Deep learning"
            if np.median(deep_learning_auc) > np.median(radiomics_auc)
            else "Radiomics"
        )
        summary_lines.extend(
            [
                f"  Statistically significant difference detected at alpha={args.alpha}.",
                f"  The higher median AUC was obtained by: {winner}.",
            ]
        )
    else:
        summary_lines.append("  No statistically significant difference was detected across these five folds.")

    (args.outdir / "summary.txt").write_text("\n".join(summary_lines), encoding="utf-8")

    dl_config_name = args.dl_preds_csv.stem.replace("_predictions", "")
    methods = [
        (deep_learning_auc, f"Deep learning\n({dl_config_name})", np.median(deep_learning_auc)),
        (radiomics_auc, f"Radiomics\n({args.radiomics_model})", np.median(radiomics_auc)),
    ]
    methods.sort(key=lambda item: item[2], reverse=True)

    plt.figure(figsize=(6, 4))
    plt.boxplot(
        [item[0] for item in methods],
        labels=[item[1] for item in methods],
        boxprops=BOX_STYLE,
        medianprops=BOX_STYLE,
        whiskerprops=BOX_STYLE,
        capprops=BOX_STYLE,
        flierprops=dict(
            marker="o",
            markersize=4,
            markerfacecolor="gray",
            markeredgecolor="black",
            linestyle="none",
        ),
    )
    plt.ylabel("AUC")
    plt.tight_layout()
    plt.savefig(args.outdir / "boxplot_auc.png", dpi=DPI, bbox_inches="tight")
    plt.close()


if __name__ == "__main__":
    main()

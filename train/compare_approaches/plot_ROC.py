from __future__ import annotations

import argparse
import ast
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics

mpl.use("Agg")
try:
    import scienceplots  # noqa: F401

    plt.style.use(["science", "grid"])
except ModuleNotFoundError:
    plt.style.use("default")
DPI = 300
COLOR_DL, COLOR_RAD = "#0072B2", "#D55E00"


def parse_list_column(series: pd.Series):
    """Convert list-like strings stored in CSV files into Python objects."""

    return series.apply(lambda value: ast.literal_eval(value) if isinstance(value, str) else value)


def load_radiomics(csv_path: Path, classifier: str):
    """Load fold-wise radiomics predictions for one classifier."""

    df = pd.read_csv(csv_path)
    df = df[(df["Classifier"] == classifier) & (df["Repeat"] == 1)]
    df["y_val"], df["y_prob"] = parse_list_column(df["y_val"]), parse_list_column(df["y_prob"])
    return {int(row["Fold"]): (row["y_val"], row["y_prob"]) for _, row in df.iterrows()}


def load_deep_learning(csv_path: Path):
    """Load fold-wise deep learning predictions."""

    df = pd.read_csv(csv_path)
    return {
        int(fold_id): (group_df["true_label"].values, group_df["prob_class_1"].values)
        for fold_id, group_df in df.groupby("split")
    }


def compute_mean_roc(
    fold_dict: dict[int, tuple[np.ndarray, np.ndarray]],
    n_points: int = 100,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Interpolate fold ROC curves on a common grid and return mean ± standard deviation."""

    tprs = []
    aucs = []
    mean_fpr = np.linspace(0, 1, n_points)

    for y_true, y_prob in fold_dict.values():
        fpr, tpr, _ = metrics.roc_curve(y_true, y_prob)
        aucs.append(metrics.auc(fpr, tpr))
        tprs.append(np.interp(mean_fpr, fpr, tpr))

    tprs = np.vstack(tprs)
    mean_tpr = tprs.mean(axis=0)
    std_tpr = tprs.std(axis=0)
    mean_auc = float(np.mean(aucs))
    return mean_fpr, mean_tpr, std_tpr, mean_auc


def plot_mean_roc(dl_dict, rad_dict, out_path: Path, lw_curve: float = 1.5, alpha_shading: float = 0.2):
    """Plot mean ROC curves with standard deviation shading for both approaches."""

    fpr_dl, tpr_dl_mean, tpr_dl_std, auc_dl = compute_mean_roc(dl_dict)
    fpr_rad, tpr_rad_mean, tpr_rad_std, auc_rad = compute_mean_roc(rad_dict)

    figure, axis = plt.subplots(figsize=(8, 6))
    handle_dl, = axis.plot(
        fpr_dl,
        tpr_dl_mean,
        label=f"Deep learning (AUC={auc_dl:.3f})",
        color=COLOR_DL,
        lw=lw_curve,
    )
    axis.fill_between(
        fpr_dl,
        np.maximum(0, tpr_dl_mean - tpr_dl_std),
        np.minimum(1, tpr_dl_mean + tpr_dl_std),
        color=COLOR_DL,
        alpha=alpha_shading,
    )

    handle_rad, = axis.plot(
        fpr_rad,
        tpr_rad_mean,
        label=f"Radiomics (AUC={auc_rad:.3f})",
        color=COLOR_RAD,
        lw=lw_curve,
    )
    axis.fill_between(
        fpr_rad,
        np.maximum(0, tpr_rad_mean - tpr_rad_std),
        np.minimum(1, tpr_rad_mean + tpr_rad_std),
        color=COLOR_RAD,
        alpha=alpha_shading,
    )

    axis.plot([0, 1], [0, 1], "--", color="gray", lw=1)
    axis.set_xlabel("False Positive Rate", fontsize=12, labelpad=10)
    axis.set_ylabel("True Positive Rate", fontsize=12, labelpad=10)
    axis.tick_params(axis="both", which="major", labelsize=10)

    handles_labels_aucs = [
        (handle_dl, f"Deep learning (AUC={auc_dl:.3f})", auc_dl),
        (handle_rad, f"Radiomics (AUC={auc_rad:.3f})", auc_rad),
    ]
    handles_labels_aucs.sort(key=lambda item: item[2], reverse=True)
    legend = axis.legend(
        [item[0] for item in handles_labels_aucs],
        [item[1] for item in handles_labels_aucs],
        fontsize=10,
    )
    for legend_line in legend.get_lines():
        legend_line.set_linewidth(2.5)

    figure.tight_layout()
    figure.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(figure)
    print(f"Saved ROC comparison plot to: {out_path}")


def main():
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(description="Plot mean ROC curves for deep learning and radiomics.")
    parser.add_argument(
        "--dl_preds_csv",
        type=Path,
        default=Path(
            "results/deep_learning/model_comparison/predict_and_analyse_probs/gland_analysis/predictions/config1_predictions.csv"
        ),
        help="CSV file with deep learning predictions.",
    )
    parser.add_argument(
        "--radiomics_preds",
        type=Path,
        default=Path(
            "results/radiomics/most_discriminant/gland/predictions_features_all_gland_most_discriminant.csv"
        ),
        help="CSV file with radiomics fold predictions.",
    )
    parser.add_argument(
        "--radiomics_model",
        type=str,
        default="Logistic Regression",
        help="Radiomics classifier name stored in the CSV file.",
    )
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("results/compare_best_radiomics_dl"),
        help="Directory where the plot will be saved.",
    )
    args = parser.parse_args()

    args.outdir.mkdir(parents=True, exist_ok=True)
    radiomics_predictions = load_radiomics(args.radiomics_preds, args.radiomics_model)
    deep_learning_predictions = load_deep_learning(args.dl_preds_csv)

    out_path = args.outdir / "roc_mean_comparison.png"
    plot_mean_roc(deep_learning_predictions, radiomics_predictions, out_path)


if __name__ == "__main__":
    main()

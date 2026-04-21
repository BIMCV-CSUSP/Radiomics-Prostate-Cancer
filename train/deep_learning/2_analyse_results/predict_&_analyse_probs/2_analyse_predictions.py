#!/usr/bin/env python
"""Run patient-level statistical comparisons across deep learning prediction files."""

from __future__ import annotations

import argparse
import glob
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests

matplotlib.use("Agg")
try:
    import scienceplots  # noqa: F401

    plt.style.use(["science", "grid"])
except ModuleNotFoundError:
    plt.style.use("default")
DPI = 300


def perform_p_value_analysis(df: pd.DataFrame, metric_col: str, alpha: float, output_dir: str):
    """Compare models at patient level for the requested prediction metric."""

    os.makedirs(output_dir, exist_ok=True)
    pivot = (
        df.pivot_table(index="patient_id", columns="model", values=metric_col, aggfunc="mean")
        .dropna(axis=0)
    )
    order = (
        df.groupby("model")[metric_col].median().sort_values(ascending=False).index.tolist()
    )
    pivot = pivot[order]

    friedman_inputs = [pivot[column].values for column in pivot.columns]
    friedman_statistic, p_global = friedmanchisquare(*friedman_inputs)

    summary_lines = [
        "=================================",
        f"Patient-level FRIEDMAN TEST | metric: {metric_col}",
        f"Statistic: {friedman_statistic:.4f}, p-value: {p_global:.4e}",
        f"alpha = {alpha}",
    ]
    if p_global < alpha:
        summary_lines.append("=> Statistically significant differences were detected across the models.")
    else:
        summary_lines.append("=> No statistically significant differences were detected across the models.")
    summary_lines.append("=================================\n")

    pairwise_matrix = None
    if p_global < alpha:
        model_names = pivot.columns.tolist()
        raw_p_values = []
        model_pairs = []
        for i in range(len(model_names)):
            for j in range(i + 1, len(model_names)):
                try:
                    _, p_value = wilcoxon(
                        pivot.iloc[:, i].values,
                        pivot.iloc[:, j].values,
                        alternative="two-sided",
                    )
                except ValueError:
                    p_value = np.nan
                raw_p_values.append(p_value)
                model_pairs.append((i, j))

        _, corrected_p_values, _, _ = multipletests(raw_p_values, alpha=alpha, method="holm")
        pairwise_matrix = np.ones((len(model_names), len(model_names)))
        summary_lines.append("Pairwise comparisons (Wilcoxon + Holm correction):")

        significant_lines = []
        for index, (i, j) in enumerate(model_pairs):
            corrected_p = corrected_p_values[index]
            pairwise_matrix[i, j] = corrected_p
            pairwise_matrix[j, i] = corrected_p
            line = f"    {model_names[i]} vs {model_names[j]}: corrected p-value = {corrected_p:.4e}"
            if corrected_p < alpha:
                line += " => SIGNIFICANT DIFFERENCE"
                significant_lines.append(line)
            summary_lines.append(line)

        summary_lines.append("\nSignificant pairwise comparisons:")
        summary_lines.extend(significant_lines or ["    None."])
    else:
        summary_lines.append("Pairwise comparisons were skipped because the global test was not significant.")

    report_path = os.path.join(output_dir, f"p_value_analysis_{metric_col}.txt")
    with open(report_path, "w", encoding="utf-8") as file_handle:
        file_handle.write("\n".join(summary_lines))
    print(f"Saved statistical report to: {report_path}")

    plt.figure(figsize=(10, 6))
    pivot.boxplot(
        color="black",
        boxprops=dict(color="black"),
        medianprops=dict(color="black"),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
        flierprops=dict(color="black"),
    )
    ylabel_map = {
        "prob_class_1": "Probability of the positive class",
        "prob_class_0": "Probability of the negative class",
    }
    plt.ylabel(ylabel_map.get(metric_col, metric_col))
    plt.xticks(rotation=45, ha="right")
    boxplot_path = os.path.join(output_dir, f"boxplot_{metric_col}.png")
    plt.savefig(boxplot_path, dpi=DPI, bbox_inches="tight")
    plt.close()
    print(f"Saved boxplot to: {boxplot_path}")

    if pairwise_matrix is not None:
        figure, axis = plt.subplots(figsize=(8, 6))
        axis.grid(False)
        heatmap = axis.imshow(pairwise_matrix, interpolation="nearest", aspect="auto", cmap="cividis")
        axis.set_xticks(np.arange(len(model_names)))
        axis.set_yticks(np.arange(len(model_names)))
        axis.set_xticklabels(model_names, rotation=45, ha="right")
        axis.set_yticklabels(model_names)
        axis.set_xticks(np.arange(-0.5, len(model_names), 1), minor=True)
        axis.set_yticks(np.arange(-0.5, len(model_names), 1), minor=True)
        axis.grid(which="minor", color="black", linestyle="--", linewidth=1)
        axis.tick_params(which="minor", bottom=False, left=False)

        for i in range(len(model_names)):
            for j in range(len(model_names)):
                color = "white" if pairwise_matrix[i, j] < alpha else "black"
                axis.text(j, i, f"{pairwise_matrix[i, j]:.3f}", ha="center", va="center", color=color, fontsize=8)

        figure.colorbar(heatmap, ax=axis, fraction=0.046, pad=0.04)
        heatmap_path = os.path.join(output_dir, f"heatmap_pvalues_{metric_col}.png")
        plt.tight_layout()
        plt.savefig(heatmap_path, dpi=DPI)
        plt.close()
        print(f"Saved p-value heatmap to: {heatmap_path}")


def main():
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(
        description="Analyse statistical differences across deep learning prediction files."
    )
    parser.add_argument(
        "-i",
        "--predictions_dir",
        type=str,
        default="results/deep_learning/model_comparison/predict_and_analyse_probs/gland_analysis/predictions",
        help="Directory containing prediction CSV files.",
    )
    parser.add_argument(
        "-m",
        "--metric_col",
        nargs="+",
        default=["prob_class_1", "prob_class_0"],
        help="Prediction columns to compare.",
    )
    parser.add_argument("-a", "--alpha", type=float, default=0.05, help="Significance level.")
    parser.add_argument(
        "-o",
        "--output_dir",
        type=str,
        default="results/deep_learning/model_comparison/predict_and_analyse_probs/gland_analysis/statistical_analysis",
        help="Directory where the statistical outputs will be saved.",
    )
    args = parser.parse_args()

    csv_files = sorted(glob.glob(os.path.join(args.predictions_dir, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No prediction CSV files were found in {args.predictions_dir}")

    df_all = pd.concat((pd.read_csv(csv_path) for csv_path in csv_files), ignore_index=True)
    print(f"Loaded {len(csv_files)} prediction files with {len(df_all)} total rows.")

    for metric_name in args.metric_col:
        print(f"\n=== Analysing metric: {metric_name} ===")
        perform_p_value_analysis(
            df=df_all,
            metric_col=metric_name,
            alpha=args.alpha,
            output_dir=args.output_dir,
        )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Run non-parametric statistical comparisons across radiomics classifiers."""

from __future__ import annotations

import argparse
from pathlib import Path

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


def main():
    """Load fold metrics, run Friedman/Wilcoxon tests, and export English reports and plots."""

    parser = argparse.ArgumentParser(
        description="Compare classifier performance with Friedman and Wilcoxon tests."
    )
    parser.add_argument("--csv_preds", type=str, required=True, help="Path to the fold prediction CSV.")
    parser.add_argument("--csv_results", type=str, required=True, help="Path to the fold metric CSV.")
    parser.add_argument("--metric", type=str, default="val_auc", help="Metric column to compare.")
    parser.add_argument("--alpha", type=float, default=0.05, help="Significance level.")
    parser.add_argument("--outdir", type=str, default=".", help="Output directory.")
    args = parser.parse_args()

    output_dir = Path(args.outdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df_results = pd.read_csv(args.csv_results)
    metric_column = args.metric
    pivot_df = (
        df_results.sort_values(by=["Classifier", "Fold"])
        .pivot_table(index="Fold", columns="Classifier", values=metric_column)
        .dropna(axis=0)
    )

    ordered_classifiers = (
        df_results.groupby("Classifier")[metric_column].median().sort_values(ascending=False).index.tolist()
    )
    pivot_df = pivot_df[ordered_classifiers]

    friedman_inputs = [pivot_df[classifier].values for classifier in pivot_df.columns]
    friedman_statistic, friedman_p = friedmanchisquare(*friedman_inputs)

    summary_lines = [
        "=================================",
        f"FRIEDMAN TEST for metric: {metric_column}",
        f"Statistic: {friedman_statistic:.4f}, p-value: {friedman_p:.4e}",
        f"alpha = {args.alpha}",
    ]

    if friedman_p < args.alpha:
        summary_lines.append(
            "=> There are statistically significant differences across classifiers (reject H0)."
        )
    else:
        summary_lines.append(
            "=> No statistically significant differences were detected across classifiers (fail to reject H0)."
        )
    summary_lines.append("=================================\n")

    pairwise_matrix = None
    if friedman_p < args.alpha:
        classifiers = pivot_df.columns.tolist()
        pairwise_matrix = np.ones((len(classifiers), len(classifiers)))

        raw_p_values = []
        pairs = []
        for i in range(len(classifiers)):
            for j in range(i + 1, len(classifiers)):
                statistic, pair_p = wilcoxon(
                    pivot_df.iloc[:, i].values,
                    pivot_df.iloc[:, j].values,
                    alternative="two-sided",
                )
                raw_p_values.append(pair_p)
                pairs.append((i, j, statistic))

        reject_flags, corrected_p_values, _, _ = multipletests(
            raw_p_values,
            alpha=args.alpha,
            method="holm",
        )

        summary_lines.append("Pairwise comparisons (Wilcoxon + Holm correction):")
        significant_lines = []
        for (i, j, statistic), corrected_p_value, reject_flag in zip(
            pairs, corrected_p_values, reject_flags
        ):
            pairwise_matrix[i, j] = corrected_p_value
            pairwise_matrix[j, i] = corrected_p_value
            result_line = (
                f"    {classifiers[i]} vs {classifiers[j]} | "
                f"statistic={statistic:.4f}, corrected p-value={corrected_p_value:.4e}"
            )
            if reject_flag:
                result_line += " => SIGNIFICANT DIFFERENCE"
                significant_lines.append(result_line)
            else:
                result_line += " => not significant"
            summary_lines.append(result_line)

        summary_lines.append("\nSignificant pairwise comparisons:")
        summary_lines.extend(significant_lines or ["    None."])
    else:
        summary_lines.append("Pairwise tests were skipped because the global Friedman test was not significant.")

    summary_path = output_dir / "model_differences_summary.txt"
    summary_path.write_text("\n".join(summary_lines), encoding="utf-8")
    print(f"Saved statistical summary to: {summary_path}")

    plt.figure(figsize=(8, 5))
    pivot_df.boxplot(
        boxprops=dict(color="black"),
        medianprops=dict(color="black"),
        whiskerprops=dict(color="black"),
        capprops=dict(color="black"),
        flierprops=dict(color="black"),
    )
    plt.ylabel("Validation AUC")
    plt.xticks(rotation=45, ha="right")
    boxplot_path = output_dir / "boxplot_metric.png"
    plt.savefig(boxplot_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved boxplot to: {boxplot_path}")

    if pairwise_matrix is not None:
        fig, axis = plt.subplots(figsize=(6, 5))
        axis.grid(False)
        heatmap = axis.imshow(pairwise_matrix, interpolation="nearest", cmap="cividis", aspect="auto")
        axis.set_xticks(np.arange(len(classifiers)))
        axis.set_yticks(np.arange(len(classifiers)))
        axis.set_xticklabels(classifiers, rotation=45, ha="right")
        axis.set_yticklabels(classifiers)
        axis.set_xticks(np.arange(-0.5, len(classifiers), 1), minor=True)
        axis.set_yticks(np.arange(-0.5, len(classifiers), 1), minor=True)
        axis.grid(which="minor", color="black", linestyle="--", linewidth=1)
        axis.tick_params(which="minor", bottom=False, left=False)

        for i in range(len(classifiers)):
            for j in range(len(classifiers)):
                value = pairwise_matrix[i, j]
                text_color = "white" if value < args.alpha else "black"
                axis.text(j, i, f"{value:.3f}", ha="center", va="center", color=text_color, fontsize=8)

        fig.colorbar(heatmap, ax=axis, fraction=0.046, pad=0.04)
        heatmap_path = output_dir / "heatmap_pvalues.png"
        plt.tight_layout()
        plt.savefig(heatmap_path, dpi=300)
        plt.close()
        print(f"Saved p-value heatmap to: {heatmap_path}")


if __name__ == "__main__":
    main()

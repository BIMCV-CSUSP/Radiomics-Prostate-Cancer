#!/usr/bin/env python3
"""Compare gland-only versus full-volume radiomics results classifier by classifier."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

matplotlib.use("Agg")
try:
    import scienceplots  # noqa: F401

    plt.style.use(["science", "grid"])
except ModuleNotFoundError:
    plt.style.use("default")


def one_sided_from_two_sided(statistic, p_two_sided, direction=+1):
    """Convert a two-sided Wilcoxon p-value into a one-sided p-value."""

    return p_two_sided / 2 if statistic * direction > 0 else 1 - p_two_sided / 2


def interquartile_range(values):
    """Compute the interquartile range of a NumPy array."""

    q75, q25 = np.percentile(values, [75, 25])
    return q75 - q25


def compare_models(gland_csv: str, full_csv: str, outdir: str, alpha: float = 0.05):
    """Compare gland and full-volume validation AUC distributions for each classifier."""

    df_gland = pd.read_csv(gland_csv)
    df_full = pd.read_csv(full_csv)
    shared_models = sorted(set(df_gland["Classifier"]).intersection(df_full["Classifier"]))
    if not shared_models:
        raise ValueError("No shared classifiers were found between the gland and full-volume result files.")

    output_dir = Path(outdir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for model_name in shared_models:
        gland_auc = df_gland.loc[df_gland["Classifier"] == model_name, "val_auc"].values
        full_auc = df_full.loc[df_full["Classifier"] == model_name, "val_auc"].values

        wilcoxon_statistic, p_two_sided = wilcoxon(gland_auc, full_auc)
        p_one_sided = one_sided_from_two_sided(wilcoxon_statistic, p_two_sided, direction=+1)

        gland_median = np.median(gland_auc)
        full_median = np.median(full_auc)
        gland_iqr = interquartile_range(gland_auc)
        full_iqr = interquartile_range(full_auc)
        better_view = "gland-only" if gland_median > full_median else "full-volume"

        model_dir = output_dir / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        summary_lines = [
            f"=== {model_name}: gland-only vs full-volume ===",
            "",
            "Validation AUC (median [IQR])",
            f"  - Gland-only:  {gland_median:.4f} [{gland_iqr:.4f}]",
            f"  - Full-volume: {full_median:.4f} [{full_iqr:.4f}]",
            "",
            "Paired Wilcoxon test (two-sided)",
            f"  W = {wilcoxon_statistic:.4f}, p = {p_two_sided:.4e}",
            f"  Conclusion: {'SIGNIFICANT DIFFERENCE' if p_two_sided < alpha else 'not significant'} (alpha={alpha})",
            "",
            "One-sided Wilcoxon test (H1: gland-only > full-volume)",
            f"  p = {p_one_sided:.4e}",
            "",
            "Summary",
            (
                f"  The higher median validation AUC was obtained by the {better_view} approach."
                if p_two_sided < alpha
                else "  No statistically significant difference was detected between both views."
            ),
        ]
        (model_dir / "results.txt").write_text("\n".join(summary_lines), encoding="utf-8")

        plt.figure(figsize=(6, 4))
        plt.boxplot(
            [gland_auc, full_auc],
            labels=["Gland-only", "Full\nvolume"],
            boxprops=dict(color="black"),
            medianprops=dict(color="black"),
            whiskerprops=dict(color="black"),
            capprops=dict(color="black"),
            flierprops=dict(color="black"),
        )
        plt.ylabel("Validation AUC")
        plt.title(f"{model_name}: paired Wilcoxon p={p_two_sided:.3f}")
        plt.tight_layout()
        plt.savefig(model_dir / "boxplot.png", dpi=300)
        plt.close()


def main():
    """CLI entrypoint."""

    parser = argparse.ArgumentParser(
        description="Compare validation AUC between gland-only and full-volume radiomics models."
    )
    parser.add_argument(
        "--gland_csv",
        default="results/radiomics/most_discriminant/gland/results_features_all_gland_most_discriminant.csv",
        help="CSV file with gland-only radiomics results.",
    )
    parser.add_argument(
        "--full_csv",
        default="results/radiomics/most_discriminant/full/results_features_all_full_most_discriminant.csv",
        help="CSV file with full-volume radiomics results.",
    )
    parser.add_argument(
        "--output_dir",
        default="results/radiomics/most_discriminant/gland_vs_full",
        help="Directory where the comparison outputs will be saved.",
    )
    args = parser.parse_args()

    compare_models(args.gland_csv, args.full_csv, args.output_dir)


if __name__ == "__main__":
    main()

from __future__ import annotations
import argparse, ast
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
mpl.use("Agg")
import matplotlib.pyplot as plt
import scienceplots                         # noqa: F401
from sklearn import metrics

plt.style.use(["science", "grid"])
DPI = 300
COLOR_DL, COLOR_RAD = "#0072B2", "#D55E00"


# --------------------------- utilidades --------------------------- #
def _parse(series: pd.Series):
    return series.apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

def load_radiomics(csv_path: Path, classifier: str):
    df = pd.read_csv(csv_path)
    df = df[(df["Classifier"] == classifier) & (df["Repeat"] == 1)]
    df["y_val"], df["y_prob"] = _parse(df["y_val"]), _parse(df["y_prob"])
    return {int(r["Fold"]): (r["y_val"], r["y_prob"]) for _, r in df.iterrows()}

def load_dl(csv_path: Path):
    df = pd.read_csv(csv_path)
    return {int(fid): (g["true_label"].values, g["prob_class_1"].values)
            for fid, g in df.groupby("split")}

def select_folds(fold_dict):
    aucs = {f: metrics.roc_auc_score(*fold_dict[f]) for f in fold_dict}
    best = max(aucs, key=aucs.get)
    median_val = np.median(list(aucs.values()))
    median = min(aucs, key=lambda f: abs(aucs[f] - median_val))
    return best, median, aucs

def plot_pair(dl, rad, f_dl, f_rad, title, out,
              lw_curve: float = 1.5,      # grosor en la gráfica
              lw_legend: float = 2.5):    # grosor en la leyenda
    """
    Dibuja dos curvas ROC (DL y Radiomics) dejando personalizar:
        • lw_curve  → grosor de las curvas en el plot
        • lw_legend → grosor de las líneas mostradas en la leyenda
    """
    fig, ax = plt.subplots(figsize=(8, 6))

    # --- DL ---------------------------------------------------------------
    fpr_dl, tpr_dl, _ = metrics.roc_curve(*dl)
    auc_dl = metrics.auc(fpr_dl, tpr_dl)
    ax.plot(fpr_dl, tpr_dl,
            label=f"Deep Learning (fold={f_dl}, AUC={auc_dl:.3f})",
            color=COLOR_DL, lw=lw_curve)

    # --- Radiomics --------------------------------------------------------
    fpr_r, tpr_r, _ = metrics.roc_curve(*rad)
    auc_r = metrics.auc(fpr_r, tpr_r)
    ax.plot(fpr_r, tpr_r,
            label=f"Radiomics (fold={f_rad}, AUC={auc_r:.3f})",
            color=COLOR_RAD, lw=lw_curve)

    # Línea azar
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1, label="_nolegend_")

    # Estética
    ax.set_xlabel("False Positive Rate", fontsize=12, labelpad=10)
    ax.set_ylabel("True Positive Rate", fontsize=12, labelpad=10)
    ax.set_title(title, fontsize=14)
    ax.tick_params(axis="both", which="major", labelsize=10)

    # Leyenda y ajuste de grosor SOLO allí
    leg = ax.legend(fontsize=10)
    for leg_line in leg.get_lines():
        leg_line.set_linewidth(lw_legend)

    fig.tight_layout()
    fig.savefig(out, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"   ➜ gráfico guardado en: {out}")

# -------------------------------- main ----------------------------------- #
def main():
    parser = argparse.ArgumentParser(
        description="Comparar ROC entre DL y Radiomics usando predicciones en CSV")
    parser.add_argument("--dl_preds_csv", type=Path, default="../../results/deep_learning/model_comparison/predict_&_analyse_probs/gland_analysis/predictions/config1_predictions.csv",
                        help="CSV con predicciones del modelo DL")
    parser.add_argument("--radiomics_preds", type=Path, default="../../results/radiomics/most_discriminant/gland/preds_features_all_gland_most_discriminant.csv",
                        help="CSV 'preds_…csv' con predicciones Radiomics")
    parser.add_argument("--radiomics_model", type=str, default="Random Forest",
                        help="Nombre del clasificador dentro del CSV Radiomics")
    parser.add_argument("--outdir", type=Path, default=Path("../../results/compare_best_radiomics_dl/roc_comparison_plots"),
                        help="Directorio donde se guardarán las figuras")
    a = parser.parse_args()

    print(">> Radiomics"); rad = load_radiomics(a.radiomics_preds, a.radiomics_model)
    r_best, r_med, r_auc = select_folds(rad); print("   AUC:", r_auc)
    print("\n>> Deep Learning"); dl = load_dl(a.dl_preds_csv)
    d_best, d_med, d_auc = select_folds(dl); print("   AUC:", d_auc)

    a.outdir.mkdir(parents=True, exist_ok=True)

    # 1) Máx general
    plot_pair(dl[d_best], rad[r_best], d_best, r_best,
              "Curvas ROC – Fold con AUC máximo"
              + (" (mismo fold)" if d_best==r_best else " (folds distintos)"),
              a.outdir / "roc_max_fold.png")
    # 2) Mediano general
    plot_pair(dl[d_med], rad[r_med], d_med, r_med,
              "Curvas ROC – Fold mediano"
              + (" (mismo fold)" if d_med==r_med else " (folds distintos)"),
              a.outdir / "roc_median_fold.png")

    # ----- extras si difieren ------------------------------------------------
    if d_best != r_best:
        plot_pair(dl[r_best], rad[r_best], r_best, r_best,
                  "Curvas ROC – Ambos métodos en el mejor fold de Radiomics",
                  a.outdir / "roc_best_on_rad_fold.png")
        plot_pair(dl[d_best], rad[d_best], d_best, d_best,
                  "Curvas ROC – Ambos métodos en el mejor fold de DL",
                  a.outdir / "roc_best_on_dl_fold.png")

    if d_med != r_med:
        plot_pair(dl[r_med], rad[r_med], r_med, r_med,
                  "Curvas ROC – Ambos métodos en el fold mediano de Radiomics",
                  a.outdir / "roc_median_on_rad_fold.png")
        plot_pair(dl[d_med], rad[d_med], d_med, d_med,
                  "Curvas ROC – Ambos métodos en el fold mediano de DL",
                  a.outdir / "roc_median_on_dl_fold.png")

    print("\n✓ Figuras generadas en", a.outdir.resolve())


if __name__ == "__main__":
    main()
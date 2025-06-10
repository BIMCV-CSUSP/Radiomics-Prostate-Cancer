#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Comparación emparejada Radiomics vs Deep-Learning (5 fold = Repeat 1, Fold 1-5)

• Lee dos CSV de predicciones:
      ––dl_preds_csv           (columnas: split, true_label, prob_class_1 …)
      ––radiomics_preds_csv    (columnas: Classifier, Fold, Repeat, y_val, y_prob …)
• Calcula el AUC de cada fold a partir de las probabilidades.
• Wilcoxon pareado  +  Cohen’s d.
• Guarda summary.txt y boxplot_auc.png.
"""

from __future__ import annotations
import argparse, ast
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import scienceplots                     # noqa: F401
from sklearn import metrics
from scipy.stats import wilcoxon

plt.style.use(["science", "grid"])
DPI = 300
BOX_KW = dict(color="black")


# ──────────────────── utilidades ──────────────────── #
def _parse(series: pd.Series):
    """Convierte strings tipo '[0,1,0]' a lista."""
    return series.apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)

def load_radiomics_preds(csv_path: Path, classifier: str) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """
    Devuelve {fold → (y_true, y_prob)}  para Repeat==1 y Fold 1-5.
    """
    df = pd.read_csv(csv_path)
    df = df[(df["Classifier"] == classifier) & (df["Repeat"] == 1) & (df["Fold"] <= 5)]
    df["y_val"]  = _parse(df["y_val"])
    df["y_prob"] = _parse(df["y_prob"])
    return {int(r["Fold"]): (np.array(r["y_val"]), np.array(r["y_prob"]))
            for _, r in df.iterrows()}

def load_dl_preds(csv_path: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """
    Convierte la columna `split` (0..4 **o** 1..5) en índices 1-5.
    """
    df = pd.read_csv(csv_path)
    min_split = df["split"].min()          # 0  o  1
    out = {}
    for sp, g in df.groupby("split"):
        sp = int(sp)
        if min_split == 0:                 # 0-based → fold = sp+1
            if sp >= 5:
                continue                   # sólo 5 folds
            fold = sp + 1
        else:                              # 1-based → fold = sp
            if sp > 5:
                continue
            fold = sp
        out[fold] = (g["true_label"].values, g["prob_class_1"].values)
    return out

def auc_vector(fold_dict: dict[int, tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
    """
    Devuelve np.array con los AUC de Fold 1-5 (en ese orden).
    Lanza error si falta algún fold.
    """
    aucs = []
    for f in range(1, 6):
        if f not in fold_dict:
            raise ValueError(f"Falta el fold {f} en las predicciones")
        y, p = fold_dict[f]
        aucs.append(metrics.roc_auc_score(y, p))
    return np.asarray(aucs)

def wilcoxon_cohen(x: np.ndarray, y: np.ndarray):
    stat, p = wilcoxon(x, y, alternative="two-sided")
    diff = x - y
    d = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) else np.nan
    return stat, p, d


# ──────────────────── main ────────────────────────── #
def main():
    parser = argparse.ArgumentParser(
        description="Compara AUC de DL vs Radiomics en 5 folds emparejados"
    )
    parser.add_argument("--dl_preds_csv", type=Path,
        default="../../results/deep_learning/model_comparison/predict_&_analyse_probs/gland_analysis/predictions/config1_predictions.csv")
    parser.add_argument("--radiomics_preds_csv", type=Path,
        default="../../results/radiomics/most_discriminant/gland/preds_features_all_gland_most_discriminant.csv")
    parser.add_argument("--radiomics_model", default="Logistic Regression",
        help="Nombre exacto del clasificador dentro del CSV Radiomics")
    parser.add_argument("--outdir", type=Path,
        default="../../results/compare_best_radiomics_dl")
    parser.add_argument("--alpha", type=float, default=0.05)
    a = parser.parse_args()

    a.outdir.mkdir(parents=True, exist_ok=True)

    # 1.  Leer predicciones  → AUC fold 1-5
    dl_auc  = auc_vector(load_dl_preds(a.dl_preds_csv))
    rad_auc = auc_vector(load_radiomics_preds(a.radiomics_preds_csv,
                                              a.radiomics_model))

    # 2.  Wilcoxon  +  Cohen’s d
    stat, p, d = wilcoxon_cohen(dl_auc, rad_auc)

    # 3.  Summary.txt
    lines = [
        f"DL predictions CSV : {a.dl_preds_csv.name}",
        f"Radiomics model    : {a.radiomics_model}",
        "",
        "Fold-wise AUC (calculado desde las probabilidades)",
    ]
    for i in range(5):
        lines.append(f"  Fold {i+1} – DL: {dl_auc[i]:.3f} | Rad: {rad_auc[i]:.3f}")
    lines += [
        "",
        f"Wilcoxon signed-rank : statistic={stat:.4f}, p={p:.4e}",
        f"Cohen’s d (pareado)  : {d:.3f}  "
        f"→ {'pequeño' if abs(d)<0.2 else 'medio' if abs(d)<0.5 else 'grande'}",
        "",
        "Conclusión:",
    ]
    if p < a.alpha:
        winner = "DL" if np.median(dl_auc) > np.median(rad_auc) else "Radiomics"
        lines += [f"  Diferencia estadísticamente significativa (α={a.alpha}).",
                  f"  → {winner} obtiene mayor AUC mediana."]
    else:
        lines.append("  No se observa diferencia significativa con estos 5 folds.")

    (a.outdir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("✓ summary.txt guardado en", a.outdir.resolve())

    # 4.  Box-plot
    dl_config_name = a.dl_preds_csv.stem.replace("_predictions", "")

    auc_medians = [np.median(dl_auc), np.median(rad_auc)]
    methods = [
        (dl_auc, f"Deep Learning\n({dl_config_name})", auc_medians[0]),
        (rad_auc, f"Radiómica\n({a.radiomics_model})", auc_medians[1]),
    ]
    # Orden descendente
    methods.sort(key=lambda x: x[2], reverse=True)

    plt.figure(figsize=(6, 4))
    plt.boxplot([x[0] for x in methods],
                labels=[x[1] for x in methods],
                boxprops=BOX_KW, medianprops=BOX_KW,
                whiskerprops=BOX_KW, capprops=BOX_KW,
                flierprops=dict(marker='o', markersize=4,
                                markerfacecolor='gray',
                                markeredgecolor='black', linestyle='none'))
    plt.ylabel("AUC")
    plt.tight_layout()
    plt.savefig(a.outdir / "boxplot_auc.png", dpi=DPI, bbox_inches="tight")
    plt.close()
    print("✓ boxplot_auc.png guardado en", a.outdir.resolve())


if __name__ == "__main__":
    main()

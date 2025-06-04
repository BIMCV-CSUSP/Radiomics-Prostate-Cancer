#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Comparación emparejada Radiomics vs Deep-Learning (mejores modelos)

1) Busca dentro de --dl_dir subcarpetas con CSV de resultados (uno por fold).
   · Selecciona el modelo con mayor mediana de val_auc.
2) Lee --radio_csv, filtra Repeat==1 y Fold 1-5 para el modelo indicado
   (o el primero si no se pasa --radio_model).
3) Empareja los 5 AUC (ordenados por Fold) y aplica:
     · Wilcoxon signed-rank   (dos colas)
     · Tamaño del efecto → Cohen’s d pareado
4) Guarda:
   · summary.txt
   · boxplot.png
"""

import argparse, os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(['science', 'grid'])
dpi = 300

from pathlib import Path
from scipy.stats import wilcoxon, norm

##############################################################################
# utilidades
##############################################################################

def read_auc_from_folder(folder, metric='val_auc'):
    """Devuelve un np.array con el máximo del metric en cada CSV de la carpeta."""
    vals = []
    for f in sorted(os.listdir(folder)):
        if f.endswith('.csv'):
            df = pd.read_csv(os.path.join(folder, f))
            if metric in df.columns:
                vals.append(df[metric].max())
    return np.asarray(vals)


def best_dl_model(dl_dir, metric='val_auc'):
    """
    Devuelve (nombre_modelo, auc_vector_ordenado)
    · nombre_modelo = subcarpeta con mayor mediana de metric
    · auc_vector = np.array de los 5 primeros folds (Fold 1-5)
    """
    best_name, best_auc = None, None
    for sub in sorted(Path(dl_dir).iterdir()):
        if not sub.is_dir(): continue
        auc = read_auc_from_folder(sub, metric)
        if len(auc) < 5:                         # esperamos 5 CSV → 5 folds
            continue
        auc5 = auc[:5]                           # Fold 1-5 (orden alfabético)
        med  = np.median(auc5)
        if best_auc is None or med > np.median(best_auc):
            best_name, best_auc = sub.name, auc5
    if best_auc is None:
        raise RuntimeError("No se encontraron CSV válidos en {}".format(dl_dir))
    return best_name, best_auc


def get_radiomics_auc(csv_path, model_name=None, metric='val_auc'):
    """
    Extrae los 5 primeros folds (Repeat==1, Fold 1-5) del CSV de radiomics.
    Si model_name es None toma el primer modelo que aparezca.
    """
    df = pd.read_csv(csv_path)
    if model_name is None:
        model_name = df['Classifier'].unique()[0]
    df_filt = df[(df['Classifier'] == model_name) &
                 (df['Repeat'] == 1) &
                 (df['Fold'] <= 5)].sort_values('Fold')
    if df_filt.shape[0] != 5:
        raise ValueError(f"No se encontraron 5 folds para {model_name}")
    return df_filt[metric].values, model_name

def paired_effect_size(x, y):
    """Wilcoxon + Cohen’s d pareado."""
    stat, p = wilcoxon(x, y, alternative='two-sided')
    diff = x - y
    d = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) != 0 else np.nan
    return stat, p, d

##############################################################################
# script principal
##############################################################################

def main():
    parser = argparse.ArgumentParser(
        description="Compara el mejor modelo DL vs Radiomics (5 folds pareados)"
    )
    parser.add_argument("--dl_dir",   default="../../artifacts/deep_learning/gland/results/",
                        help="Carpeta con subcarpetas de resultados DL (CSV por fold)")
    parser.add_argument("--radio_csv", default="../../results/radiomics/most_discriminant/gland/resultados_features_all_gland_most_discriminant.csv",
                        help="CSV de resultados radiomics")
    parser.add_argument("--radio_model", default="Random Forest",
                        help="Nombre exacto del modelo radiomics (opcional)")
    parser.add_argument("--outdir",   default="../../results/compare_best_radiomics_dl",
                        help="Carpeta de salida")
    parser.add_argument("--alpha", type=float, default=0.05)
    args = parser.parse_args()

    out = Path(args.outdir); out.mkdir(parents=True, exist_ok=True)

    # 1. Deep Learning
    dl_name, dl_auc = best_dl_model(args.dl_dir)
    # 2. Radiomics
    rad_auc, rad_name = get_radiomics_auc(args.radio_csv, args.radio_model)

    # 3. Wilcoxon pareado
    stat, p, d = paired_effect_size(dl_auc, rad_auc)
    alpha = args.alpha

    # 4) Resumen
    lines = [
        f"DL best model  : {dl_name}",
        f"Radiomics model: {rad_name}",
        "",
        "Fold-wise AUC"
    ]
    for i in range(5):
        lines.append(f"  Fold {i+1:<2d} – DL: {dl_auc[i]:.3f} | Rad: {rad_auc[i]:.3f}")
    lines += [
        "",
        f"Wilcoxon signed-rank : statistic={stat:.4f}, p={p:.4e}",
        f"Cohen's d (pareado)  : {d:.3f}  "
        f"→ {'pequeño' if abs(d)<0.2 else 'medio' if abs(d)<0.5 else 'grande'}",
        "",
        "Conclusión:"
    ]
    if p < alpha:
        winner = "DL" if np.median(dl_auc) > np.median(rad_auc) else "Radiomics"
        lines += [f"  Diferencia estadísticamente significativa (α={alpha}).",
                  f"  → {winner} obtiene mayor AUC mediana."]
    else:
        lines.append("  No se observa diferencia significativa con estos 5 folds.")

    (out / "summary.txt").write_text("\n".join(lines), encoding="utf-8")
    print("Resumen escrito en", out / "summary.txt")

    # 5) Box-plot
    boxprops   = dict(color='black')
    medianprops = dict(color='black')
    whiskerprops = dict(color='black')
    capprops     = dict(color='black')
    flierprops   = dict(markerfacecolor='gray', marker='o', markersize=4,
                        linestyle='none', markeredgecolor='black')

    plt.figure(figsize=(6, 4))
    plt.boxplot([dl_auc, rad_auc],
                labels=[f"DL\n({dl_name})", f"Radiomics\n({rad_name})"],
                boxprops=boxprops, medianprops=medianprops,
                whiskerprops=whiskerprops, capprops=capprops,
                flierprops=flierprops)
    plt.ylabel("AUC")
    plt.title("AUC en los 5 folds (emparejados)")
    plt.tight_layout()
    boxplot_path = out / "boxplot_auc.png"
    plt.savefig(boxplot_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print("Box-plot guardado en", boxplot_path)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse, os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
from sklearn import metrics
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science', 'grid'])

def one_sided_from_two_sided(stat, p_two, direction=+1):
    """Convierte p-valor bicaudal en unilateral (direction = +1 → glándula > full)."""
    return p_two/2 if stat*direction > 0 else 1 - p_two/2

def iqr(arr):
    """Rango intercuartílico de un array NumPy."""
    q75, q25 = np.percentile(arr, [75, 25])
    return q75 - q25

def aucs_from_predictions(pred_csv):
    """Devuelve vector de AUC (uno por split) a partir de un CSV de predicciones."""
    aucs = []
    df = pd.read_csv(pred_csv)
    if {'split', 'true_label', 'prob_class_1'}.issubset(df.columns):
        for split, grupo in df.groupby('split'):
            y_true = grupo['true_label'].values
            y_prob = grupo['prob_class_1'].values
            if len(np.unique(y_true)) < 2:
                auc = np.nan
            else:
                auc = metrics.roc_auc_score(y_true, y_prob)
            aucs.append(auc)
    return np.array(aucs)

def aucs_from_folder(folder, metric='val_auc'):
    """Devuelve un vector con el máximo AUC de cada CSV de la carpeta."""
    vals = []
    for f in sorted(os.listdir(folder)):
        if f.endswith('.csv'):
            df = pd.read_csv(os.path.join(folder, f))
            if metric in df.columns:
                vals.append(df[metric].max())
    return np.asarray(vals)

def compare_configs(gland_pred_dir, full_dir, out_dir, metric='val_auc', alpha=0.05):
    os.makedirs(out_dir, exist_ok=True)

    # Configuraciones de glándula (de los CSVs)
    gland_cfgs = {fname.replace('_predictions.csv', '') for fname in os.listdir(gland_pred_dir) if fname.endswith('_predictions.csv')}
    # Configuraciones de full (subdirectorios)
    full_cfgs  = {c for c in os.listdir(full_dir) if os.path.isdir(os.path.join(full_dir, c))}
    comunes = sorted(gland_cfgs & full_cfgs)
    if not comunes:
        raise ValueError("No hay configuraciones comunes entre carpetas.")

    for cfg in comunes:
        # Glándula: lee el CSV de predicciones correspondiente a esa configuración
        pred_csv = os.path.join(gland_pred_dir, f"{cfg}_predictions.csv")
        auc_g = aucs_from_predictions(pred_csv)
        auc_f = aucs_from_folder(os.path.join(full_dir, cfg), metric)

        med_g, med_f = np.nanmedian(auc_g), np.nanmedian(auc_f)
        iqr_g, iqr_f = iqr(auc_g), iqr(auc_f)
        diff_vec = auc_g - auc_f
        w_stat, p_two = wilcoxon(auc_g, auc_f)
        p_one = one_sided_from_two_sided(w_stat, p_two, +1)

        # --- Cohen's d para muestras pareadas ---
        std_diff = np.nanstd(diff_vec, ddof=1)
        cohen_d = np.nanmean(diff_vec) / std_diff if std_diff else np.nan
        efecto  = ("GRANDE" if abs(cohen_d) >= 0.8 else
                   "MEDIO"  if abs(cohen_d) >= 0.5 else
                   "PEQUEÑO" if abs(cohen_d) >= 0.2 else "DESPRECIABLE")

        cfg_out = os.path.join(out_dir, cfg)
        os.makedirs(cfg_out, exist_ok=True)
        with open(os.path.join(cfg_out, 'results.txt'), 'w', encoding='utf-8') as f:
            f.write(f"=== Configuración: {cfg} ===\n\n")
            f.write("AUC (mediana [IQR])\n")
            f.write(f"  • Glándula............. {med_g:.4f} [{iqr_g:.4f}]\n")
            f.write(f"  • Imagen completa...... {med_f:.4f} [{iqr_f:.4f}]\n\n")
            f.write("Test de Wilcoxon pareado (dos colas)\n")
            f.write(f"  W = {w_stat:.4f},  p = {p_two:.4e}\n")
            f.write("Conclusión: " +
                    ("DIFERENCIA SIGNIFICATIVA" if p_two < alpha else "no significativa") +
                    f" (α = {alpha})\n\n")
            f.write("Wilcoxon unilateral (H₁: glándula > full)\n")
            f.write(f"  p = {p_one:.4e}\n\n")
            f.write(f"Cohen's d = {cohen_d:.3f}  →  {efecto}\n\n")
            if p_two < alpha:
                mejor = "glándula" if med_g > med_f else "imagen completa"
                f.write(f"Resumen: el enfoque **{mejor}** obtiene mayor AUC mediano.\n")
            else:
                f.write("Resumen: no se detectan diferencias significativas.\n")
                
        # --- Boxplot ---
        plt.figure(figsize=(7,4))
        plt.boxplot([auc_g, auc_f],
                    labels=['Glándula', 'Imagen\ncompleta'],
                    boxprops=dict(color='black'),
                    medianprops=dict(color='black'),
                    whiskerprops=dict(color='black'),
                    capprops=dict(color='black'),
                    flierprops=dict(color='black'))
        plt.title(f"{cfg}: AUC  (p={p_two:.3f})")
        plt.ylabel("AUC en validación")
        plt.tight_layout()
        plt.savefig(os.path.join(cfg_out, 'boxplot.png'), dpi=300)
        plt.close()

def main():
    parser = argparse.ArgumentParser(
        description="Comparación estadística de configuraciones de Deep Learning entre dos enfoques"
    )
    parser.add_argument(
        '--gland_pred_dir',
        default="../../../../results/deep_learning/model_comparison/predict_&_analyse_probs/gland_analysis/predictions",
        help="Directorio con predicciones CSV para glándula"
    )
    parser.add_argument(
        '--full_dir',
        default="../../../../artifacts/deep_learning/full/results/",
        help="Directorio raíz con configuraciones para imagen completa"
    )
    parser.add_argument(
        '--output_dir',
        default="../../../../results/deep_learning/model_comparison/simple_statistical_analysis/gland_vs_full/",
        help="Directorio donde guardar los resultados"
    )
    parser.add_argument(
        '--metric_col',
        default='val_auc',
        help="Columna/métrica de validación a comparar (solo para full)"
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.05,
        help="Nivel de significación estadística"
    )
    args = parser.parse_args()
    
    compare_configs(args.gland_pred_dir, args.full_dir, args.output_dir, args.metric_col, args.alpha)

if __name__ == '__main__':
    main()
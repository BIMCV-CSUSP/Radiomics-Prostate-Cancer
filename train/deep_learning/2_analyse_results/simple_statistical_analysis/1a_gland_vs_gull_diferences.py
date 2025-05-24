#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compara, para cada configuración de Deep Learning, la estrategia
glándula vs. imagen completa mediante:

  • Test de Wilcoxon pareado (dos colas) + su versión unilateral
    (H₁: glándula > full).
  • Tamaño del efecto (Cohen's d) sobre las diferencias de AUC.
  • Informe de texto y boxplot con los resultados.

Pensado para validación cruzada con sólo 5 folds por configuración.
"""

import argparse, os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science', 'grid'])

# ---------- utilidades -------------------------------------------------
def one_sided_from_two_sided(stat, p_two, direction=+1):
    """Convierte p-valor bicaudal en unilateral (direction = +1 → glándula > full)."""
    return p_two/2 if stat*direction > 0 else 1 - p_two/2

def read_auc_from_folder(folder, metric='val_auc'):
    """Devuelve un vector con el máximo AUC de cada CSV de la carpeta."""
    vals = []
    for f in sorted(os.listdir(folder)):
        if f.endswith('.csv'):
            df = pd.read_csv(os.path.join(folder, f))
            if metric in df.columns:
                vals.append(df[metric].max())
    return np.asarray(vals)
# -----------------------------------------------------------------------

def compare_configs(gland_dir, full_dir, out_dir,
                    metric='val_auc', alpha=0.05):
    os.makedirs(out_dir, exist_ok=True)

    gland_cfgs = {c for c in os.listdir(gland_dir)
                  if os.path.isdir(os.path.join(gland_dir, c))}
    full_cfgs  = {c for c in os.listdir(full_dir)
                  if os.path.isdir(os.path.join(full_dir,  c))}
    comunes = sorted(gland_cfgs & full_cfgs)
    if not comunes:
        raise ValueError("No hay configuraciones comunes entre carpetas.")

    for cfg in comunes:
        auc_g = read_auc_from_folder(os.path.join(gland_dir, cfg), metric)
        auc_f = read_auc_from_folder(os.path.join(full_dir,  cfg), metric)

        mean_g, mean_f = auc_g.mean(), auc_f.mean()
        diff_vec       = auc_g - auc_f
        w_stat, p_two  = wilcoxon(auc_g, auc_f)              # Wilcoxon dos colas
        p_one          = one_sided_from_two_sided(w_stat, p_two, +1)

        # --- Cohen's d para muestras pareadas ---
        cohen_d = diff_vec.mean() / diff_vec.std(ddof=1) if diff_vec.std(ddof=1) else np.nan
        efecto  = ("GRANDE" if abs(cohen_d) >= 0.8 else
                   "MEDIO"  if abs(cohen_d) >= 0.5 else
                   "PEQUEÑO" if abs(cohen_d) >= 0.2 else "DESPRECIABLE")

        # --- Informe de resultados ---
        cfg_out = os.path.join(out_dir, cfg)
        os.makedirs(cfg_out, exist_ok=True)
        with open(os.path.join(cfg_out, 'results.txt'), 'w', encoding='utf-8') as f:
            f.write(f"=== Configuración: {cfg} ===\n\n")
            f.write(f"{metric} (media ± sd)\n")
            f.write(f"  • Glándula............. {mean_g:.4f} ± {auc_g.std(ddof=1):.4f}\n")
            f.write(f"  • Imagen completa...... {mean_f:.4f} ± {auc_f.std(ddof=1):.4f}\n\n")
            f.write("Test de Wilcoxon pareado (dos colas)\n")
            f.write(f"  W = {w_stat:.4f},  p = {p_two:.4e}\n")
            f.write("Conclusión: " +
                    ("DIFERENCIA SIGNIFICATIVA" if p_two < alpha else "no significativa") +
                    f"  (α = {alpha})\n\n")
            f.write("Wilcoxon unilateral (H₁: glándula > full)\n")
            f.write(f"  p = {p_one:.4e}\n\n")
            f.write(f"Cohen's d = {cohen_d:.3f}  →  {efecto}\n\n")
            if p_two < alpha:
                mejor = "glándula" if mean_g > mean_f else "imagen completa"
                f.write(f"Resumen: el enfoque **{mejor}** obtiene mayor {metric} medio.\n")
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
        plt.title(f"{cfg}: {metric}  (p={p_two:.3f})")
        plt.ylabel(metric)
        plt.tight_layout()
        plt.savefig(os.path.join(cfg_out, 'boxplot.png'), dpi=300)
        plt.close()

def main():
    parser = argparse.ArgumentParser(
        description="Comparación estadística de configuraciones de Deep Learning entre dos enfoques"
    )
    parser.add_argument(
        '--gland_dir',
        default="../../../../artifacts/deep_learning/gland/results/",
        help="Directorio raíz con configuraciones para glándula"
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
        help="Columna/métrica de validación a comparar"
    )
    parser.add_argument(
        '--alpha',
        type=float,
        default=0.05,
        help="Nivel de significación estadística"
    )
    args = parser.parse_args()
    
    compare_configs(args.gland_dir, args.full_dir, args.output_dir, args.metric_col, args.alpha)

if __name__ == '__main__':
    main()

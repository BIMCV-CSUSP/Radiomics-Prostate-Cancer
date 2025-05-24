#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scienceplots
plt.style.use(['science', 'grid'])

def one_sided_from_two_sided(stat, p_two, direction=+1):
    """Convierte un p-valor bicaudal en unilateral."""
    return p_two/2 if stat*direction > 0 else 1 - p_two/2

def compare_models(gland_csv, full_csv, outdir, alpha=0.05):
    """
    Compara, para cada clasificador, la estrategia glándula vs. imagen completa
    usando SIEMPRE el test de Wilcoxon para muestras pareadas.
    """
    df_gland = pd.read_csv(gland_csv)
    df_full  = pd.read_csv(full_csv)

    modelos = sorted(set(df_gland['Classifier']).intersection(df_full['Classifier']))
    if not modelos:
        raise ValueError("No hay modelos en común entre ambos ficheros.")

    os.makedirs(outdir, exist_ok=True)

    for model in modelos:
        auc_g = df_gland.loc[df_gland['Classifier'] == model, 'val_auc'].values
        auc_f = df_full .loc[df_full ['Classifier'] == model, 'val_auc'].values

        # --- Wilcoxon (dos colas) ---
        w_stat, p_two = wilcoxon(auc_g, auc_f)
        # --- Wilcoxon (una cola, H1: glándula > full) ---
        p_one = one_sided_from_two_sided(w_stat, p_two, direction=+1)

        # Estadística descriptiva
        mean_g, mean_f = np.mean(auc_g), np.mean(auc_f)
        diff_dir       = 'glándula' if mean_g > mean_f else 'imagen completa'

        # --- Informe ---
        mdl_dir = os.path.join(outdir, model)
        os.makedirs(mdl_dir, exist_ok=True)
        with open(os.path.join(mdl_dir, 'results.txt'), 'w', encoding='utf-8') as f:
            f.write(f"=== {model}: glándula vs. imagen completa ===\n\n")
            f.write(f"AUC (media ± sd)\n")
            f.write(f"  • Glándula............. {mean_g:.4f} ± {np.std(auc_g, ddof=1):.4f}\n")
            f.write(f"  • Imagen completa...... {mean_f:.4f} ± {np.std(auc_f, ddof=1):.4f}\n\n")
            f.write("Test de Wilcoxon pareado (dos colas)\n")
            f.write(f"  W = {w_stat:.4f},  p = {p_two:.4e}\n")
            f.write("Conclusión: " +
                    ("DIFERENCIA SIGNIFICATIVA" if p_two < alpha else "no significativa") +
                    f" (α = {alpha})\n\n")
            f.write("Wilcoxon unilateral (H₁: glándula > full)\n")
            f.write(f"  p = {p_one:.4e}\n\n")
            f.write("Resumen:\n")
            if p_two < alpha:
                f.write(f"  El enfoque con mayor AUC medio es **{diff_dir}**.\n")
            else:
                f.write("  No se detectan diferencias significativas entre enfoques.\n")

        # --- Boxplot ---
        plt.figure(figsize=(6,4))
        plt.boxplot([auc_g, auc_f],
                    labels=['Glándula','Imagen\ncompleta'],
                    boxprops=dict(color='black'),
                    medianprops=dict(color='black'),
                    whiskerprops=dict(color='black'),
                    capprops=dict(color='black'),
                    flierprops=dict(color='black'))
        plt.ylabel("AUC en validación")
        plt.title(f"{model}: AUC (Wilcoxon p={p_two:.3f})")
        plt.tight_layout()
        plt.savefig(os.path.join(mdl_dir, 'boxplot.png'), dpi=300)
        plt.close()

def main():
    parser = argparse.ArgumentParser(
        description="Compara AUC de modelos entrenados con dos enfoques distintos"
    )
    parser.add_argument(
        '--gland_csv',
        default='../../../results/radiomics/most_discriminant/gland/resultados_features_all_gland_most_discriminant.csv',
        help="CSV con resultados del modelo (sólo glándula)"
    )
    parser.add_argument(
        '--full_csv',
        default='../../../results/radiomics/most_discriminant/full/resultados_features_all_full_most_discriminant.csv',
        help="CSV con resultados del modelo (imagen completa)"
    )
    parser.add_argument(
        '--output_dir',
        default='../../../results/radiomics/most_discriminant/gland_vs_full',
        help="Directorio donde guardar los resultados"
    )
    args = parser.parse_args()
    
    compare_models(args.gland_csv, args.full_csv, args.output_dir)


if __name__ == '__main__':
    main()
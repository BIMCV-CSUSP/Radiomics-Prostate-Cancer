#!/usr/bin/env python3

import argparse
import os
import pandas as pd
import numpy as np
from scipy.stats import ttest_rel, wilcoxon, shapiro
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(['science', 'grid'])

def one_sided_p_from_two_sided(stat, p_two, direction):
    """
    Convierte un p-valor bicaudal en un p-valor unilateral.
    `direction` debe ser +1 si H1 es mean(diff) > 0, o -1 si H1 es mean(diff) < 0.
    """
    if stat * direction > 0:
        return p_two / 2
    else:
        return 1 - (p_two / 2)


def compare_models(gland_csv, full_csv, output_dir, alpha=0.05):
    """
    Compara los resultados de validación de distintos modelos entre dos enfoques:
    solo glándula vs imagen completa. Para cada clasificador:
      - Calcula la media y desviación estándar del AUC de validación en cada enfoque.
      - Realiza un t-test pareado (paramétrico) y un test de Wilcoxon (no paramétrico).
      - Realiza la versión unilateral del t-test y de Wilcoxon para H1: glándula > full.
      - Genera un boxplot comparativo.
      - Escribe un informe interpretando los resultados, explicando cada test y la conclusión final.
    """
    df_gland = pd.read_csv(gland_csv)
    df_full  = pd.read_csv(full_csv)

    os.makedirs(output_dir, exist_ok=True)
    modelos = sorted(set(df_gland['Classifier']).intersection(df_full['Classifier']))
    if not modelos:
        raise ValueError("No hay modelos en común entre ambos ficheros.")

    for model in modelos:
        model_dir = os.path.join(output_dir, model)
        os.makedirs(model_dir, exist_ok=True)
        
        auc_g = df_gland[df_gland['Classifier'] == model]['val_auc'].values
        auc_f = df_full[df_full['Classifier'] == model]['val_auc'].values

        # Estadísticos descriptivos
        mean_g, std_g = np.mean(auc_g), np.std(auc_g, ddof=1)
        mean_f, std_f = np.mean(auc_f), np.std(auc_f, ddof=1)
        diff = mean_g - mean_f

        # Test de normalidad Shapiro-Wilk sobre la diferencia de AUCs
        diff_vec = auc_g - auc_f
        w_shap, p_shap = shapiro(diff_vec)
        normalidad = p_shap > alpha

        # Según normalidad, test adecuado
        if normalidad:
            test_name = "t-test pareado"
            t_stat, p_two_t = ttest_rel(auc_g, auc_f)
            # Test unilateral
            p_one_t = one_sided_p_from_two_sided(t_stat, p_two_t, direction=+1)
            # Para uniformidad, igualamos nombre de variables
            stat_report = t_stat
            p_report = p_two_t
            p_one_report = p_one_t
        else:
            test_name = "Wilcoxon signed-rank"
            w_stat, p_two_w = wilcoxon(auc_g, auc_f)
            p_one_w = one_sided_p_from_two_sided(w_stat, p_two_w, direction=+1)
            stat_report = w_stat
            p_report = p_two_w
            p_one_report = p_one_w

        # Interpretaciones
        diff_dir = 'glándula' if mean_g > mean_f else 'imagen completa'
        interpret_res = (
            f"Significativo (p={p_report:.4f} < {alpha}) -> Se rechaza H₀: hay diferencia entre enfoques."
            if p_report < alpha else
            f"No significativo (p={p_report:.4f} ≥ {alpha}) -> No se rechaza H₀: no hay diferencia significativa."
        )
        interpret_one = (
            f"Prueba unilateral: p={p_one_report:.4f} < {alpha}. Hay evidencia de que glándula > imagen completa."
            if p_one_report < alpha else
            f"Prueba unilateral: p={p_one_report:.4f} ≥ {alpha}. No hay evidencia suficiente de que glándula > imagen completa."
        )

        resumen = ""
        if p_report < alpha:
            resumen += (
                f"Resumen:\n"
                f"Existe evidencia estadística de diferencia entre enfoques según el {test_name}.\n"
                f"El enfoque con mayor AUC medio es: **{diff_dir}**\n"
            )
            if mean_g > mean_f and p_one_report < alpha:
                resumen += "Además, la prueba unilateral respalda que 'glándula' es superior.\n"
            elif mean_g < mean_f:
                resumen += "Sin embargo, el enfoque de imagen completa obtiene mayor AUC medio.\n"
        else:
            resumen += (
                f"Resumen:\n"
                f"No se ha encontrado diferencia significativa entre ambos enfoques según el {test_name}.\n"
                f"AUC medios: glándula = {mean_g:.3f}, imagen completa = {mean_f:.3f}\n"
            )

        # Guardar el informe
        txt_path = os.path.join(model_dir, 'results.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"=== Comparación de modelos: {model} ===\n\n")
            f.write(f"Estadística descriptiva:\n")
            f.write(f"  - AUC validación glándula: media = {mean_g:.4f}, std = {std_g:.4f}\n")
            f.write(f"  - AUC validación imagen completa: media = {mean_f:.4f}, std = {std_f:.4f}\n")
            f.write(f"  - Diferencia de medias: {diff:.4f} (glándula - full)\n\n")
            f.write(f"Test de normalidad Shapiro-Wilk sobre las diferencias de AUC:\n")
            f.write(f"  - Estadístico W = {w_shap:.4f}, p-valor = {p_shap:.4f} -> ")
            f.write("Se asume normalidad\n" if normalidad else "No se asume normalidad\n")
            f.write(f"\nTest aplicado: {test_name}\n")
            f.write(f"  - Estadístico = {stat_report:.4f}\n")
            f.write(f"  - p-valor (2 colas) = {p_report:.4f}\n")
            f.write(f"  - {interpret_res}\n")
            f.write(f"\n{interpret_one}\n\n")
            f.write(resumen)

        # Boxplot
        props = dict(
            boxprops=dict(color='black'),
            medianprops=dict(color='black'),
            whiskerprops=dict(color='black'),
            capprops=dict(color='black'),
            flierprops=dict(color='black')
        )
        plt.figure(figsize=(8,5))
        plt.boxplot([auc_g, auc_f], labels=['Glándula','Imagen completa'], **props)
        plt.title(f"Comparación AUC validación: {model}")
        plt.ylabel("AUC en validación")
        plt.xticks(rotation=45, ha='right')
        boxplot_path = os.path.join(model_dir, 'boxplot.png')
        plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
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
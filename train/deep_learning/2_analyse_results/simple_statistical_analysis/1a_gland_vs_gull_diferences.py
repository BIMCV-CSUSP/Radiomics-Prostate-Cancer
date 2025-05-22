#!/usr/bin/env python3
# -*- coding: utf-8 -*-

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
    if stat * direction > 0:
        return p_two / 2
    else:
        return 1 - (p_two / 2)

def read_auc_from_folder(config_folder, metric_col='val_auc'):
    """
    Lee los resultados de los 5 CSVs en la carpeta y devuelve un vector de AUCs.
    """
    aucs = []
    for file in sorted(os.listdir(config_folder)):
        if file.endswith('.csv'):
            df = pd.read_csv(os.path.join(config_folder, file))
            # Coge el máximo valor de la métrica en el CSV (por si es por época)
            if metric_col in df.columns:
                auc = df[metric_col].max()
                aucs.append(auc)
    return np.array(aucs)

def compare_configs(gland_dir, full_dir, output_dir, metric_col='val_auc', alpha=0.05):
    """
    Compara cada configuración (subcarpeta) entre gland y full.
    Hace test de normalidad sobre la diferencia, escoge test adecuado (t o Wilcoxon),
    calcula Cohen's d y genera informe y boxplot.
    """
    os.makedirs(output_dir, exist_ok=True)
    # Listar las configuraciones presentes en ambos enfoques
    gland_configs = sorted([c for c in os.listdir(gland_dir) if os.path.isdir(os.path.join(gland_dir, c))])
    full_configs = sorted([c for c in os.listdir(full_dir) if os.path.isdir(os.path.join(full_dir, c))])
    common_configs = sorted(list(set(gland_configs).intersection(full_configs)))
    if not common_configs:
        raise ValueError("No hay configuraciones en común entre ambos directorios.")

    for config in common_configs:
        config_dir_g = os.path.join(gland_dir, config)
        config_dir_f = os.path.join(full_dir, config)
        auc_g = read_auc_from_folder(config_dir_g, metric_col)
        auc_f = read_auc_from_folder(config_dir_f, metric_col)

        mean_g, std_g = np.mean(auc_g), np.std(auc_g, ddof=1)
        mean_f, std_f = np.mean(auc_f), np.std(auc_f, ddof=1)
        diff = mean_g - mean_f

        # Cohen's d para muestras pareadas
        diff_vec = auc_g - auc_f
        mean_diff = np.mean(diff_vec)
        std_diff = np.std(diff_vec, ddof=1)
        cohen_d = mean_diff / std_diff if std_diff != 0 else np.nan

        # Test de normalidad Shapiro-Wilk sobre las diferencias
        w_shap, p_shap = shapiro(diff_vec)
        normalidad = p_shap > alpha

        if normalidad:
            test_name = "t-test pareado"
            t_stat, p_two_t = ttest_rel(auc_g, auc_f)
            p_one_t = one_sided_p_from_two_sided(t_stat, p_two_t, direction=+1)
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

        # Interpretación
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

        interpret_cohen = (
            f"Tamaño del efecto (Cohen's d): {cohen_d:.3f} -> "
            f"{'GRANDE' if abs(cohen_d) >= 0.8 else 'MEDIO' if abs(cohen_d) >= 0.5 else 'PEQUEÑO' if abs(cohen_d) >= 0.2 else 'DESPRECIABLE'}"
        )

        resumen = ""
        if p_report < alpha:
            resumen += (
                f"Resumen:\n"
                f"Existe evidencia estadística de diferencia entre enfoques según el {test_name}.\n"
                f"El enfoque con mayor {metric_col} medio es: **{diff_dir}**\n"
            )
            if mean_g > mean_f and p_one_report < alpha:
                resumen += "Además, la prueba unilateral respalda que 'glándula' es superior.\n"
            elif mean_g < mean_f:
                resumen += "Sin embargo, el enfoque de imagen completa obtiene mayor valor medio.\n"
            resumen += interpret_cohen + "\n"
        else:
            resumen += (
                f"Resumen:\n"
                f"No se ha encontrado diferencia significativa entre ambos enfoques según el {test_name}.\n"
                f"Medias: glándula = {mean_g:.3f}, imagen completa = {mean_f:.3f}\n"
            )
            resumen += interpret_cohen + "\n"

        # Crear salida por configuración
        model_dir = os.path.join(output_dir, config)
        os.makedirs(model_dir, exist_ok=True)
        txt_path = os.path.join(model_dir, 'results.txt')
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(f"=== Comparación de configuración: {config} ===\n\n")
            f.write(f"Estadística descriptiva:\n")
            f.write(f"  - {metric_col} glándula: media = {mean_g:.4f}, std = {std_g:.4f}\n")
            f.write(f"  - {metric_col} imagen completa: media = {mean_f:.4f}, std = {std_f:.4f}\n")
            f.write(f"  - Diferencia de medias: {diff:.4f} (glándula - full)\n\n")
            f.write(f"Test de normalidad Shapiro-Wilk sobre las diferencias de {metric_col}:\n")
            f.write(f"  - Estadístico W = {w_shap:.4f}, p-valor = {p_shap:.4f} -> ")
            f.write("Se asume normalidad\n" if normalidad else "No se asume normalidad\n")
            f.write(f"\nTest aplicado: {test_name}\n")
            f.write(f"  - Estadístico = {stat_report:.4f}\n")
            f.write(f"  - p-valor (2 colas) = {p_report:.4f}\n")
            f.write(f"  - {interpret_res}\n")
            f.write(f"\n{interpret_one}\n\n")
            f.write(f"{interpret_cohen}\n")
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
        plt.title(f"Comparación {metric_col}: {config}")
        plt.ylabel(metric_col)
        plt.xticks(rotation=45, ha='right')
        boxplot_path = os.path.join(model_dir, 'boxplot.png')
        plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
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

#!/usr/bin/env python
"""
Script para análisis estadístico de predicciones de múltiples modelos.

Este script implementa un análisis estadístico para comparar diferentes 
modelos de deep learning a nivel de paciente, siguiendo un enfoque 
de tests no paramétricos adecuado para comparaciones múltiples:

Proceso estadístico:
1. Test de Friedman para detectar diferencias globales entre modelos
2. Si hay diferencias globales, comparaciones post-hoc con Wilcoxon + corrección Holm
3. Generación de visualizaciones y reportes detallados
"""

import os
import glob
import argparse
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import scienceplots

plt.style.use(['science', 'grid'])
dpi = 300

def perform_p_value_analysis(
    df: pd.DataFrame,
    metric_col: str,
    alpha: float,
    output_dir: str
):
    """
    Realiza análisis estadístico completo para comparar modelos usando métrica especificada.
    
    Proceso:
    1) Test de Friedman para diferencias globales en `metric_col`
    2) Comparaciones pareadas con Wilcoxon (dos colas) + corrección Holm
    3) Boxplot de `metric_col` por modelo
    4) Heatmap de p-values corregidos (si el test global es significativo)
    5) Informe de texto con resultados detallados
    
    Args:
        df (pd.DataFrame): DataFrame con predicciones
        metric_col (str): Nombre de la columna con la métrica a analizar
        alpha (float): Nivel de significancia para los tests
        output_dir (str): Directorio para guardar resultados
        
    Returns:
        list: Líneas de texto con el resumen del análisis
    """

    # Crear directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)

    # ===== 1. Preparación de matriz paciente × modelo =====
    # Pivotamos para tener una matriz donde:
    # - Cada fila es un paciente
    # - Cada columna es un modelo
    # - Los valores son la media de la métrica por paciente y modelo
    pivot = (df.pivot_table(index='patient_id',      
                            columns='model',
                            values=metric_col, 
                            aggfunc='mean')          
              .dropna(axis=0)) 
                         
    # Ordenamos los modelos por mediana de la métrica (descendente)
    orden = (df.groupby('model')[metric_col]
               .median()
               .sort_values(ascending=False)
               .index.tolist())
    pivot = pivot[orden] # Reorganizamos columnas según este orden

    # ===== 2. Test de Friedman =====
    datos = [pivot[col].values for col in pivot.columns]
    stat, p_global = friedmanchisquare(*datos)

    # ===== 3. Preparamos informe de texto =====
    lines = []
    lines.append("=================================")
    lines.append(f"TEST DE FRIEDMAN por paciente | métrica: {metric_col}")
    lines.append(f"Estadístico: {stat:.4f}, p-value: {p_global:.4e}")
    lines.append(f"alpha = {alpha}")

    # Interpretación del resultado global
    if p_global < alpha:
        lines.append("=> HAY diferencias estadísticamente significativas entre los modelos (rechazamos H0).")
    else:
        lines.append("=> NO se evidencian diferencias estadísticamente significativas entre los modelos (no se rechaza H0).")
    lines.append("=================================\n")

    # ===== 4. Comparaciones post-hoc si el test global es significativo =====
    if p_global < alpha:
        modelos = pivot.columns.tolist()
        n = len(modelos)
        pvals, pares = [], []

        # Realizamos todas las comparaciones pareadas posibles
        for i in range(n):
            for j in range(i+1, n):
                xi, xj = pivot.iloc[:, i].values, pivot.iloc[:, j].values
                try:
                    # Test de Wilcoxon para muestras pareadas
                    _, p = wilcoxon(xi, xj, alternative='two-sided')
                except:
                    p = np.nan
                pvals.append(p)
                pares.append((i, j))

        # Corrección de Holm-Bonferroni para comparaciones múltiples
        _, p_corr, _, _ = multipletests(pvals, alpha=alpha, method='holm')

        # Creamos matriz simétrica de p-values corregidos
        matriz_p = np.ones((n, n))
        for k, (i, j) in enumerate(pares):
            matriz_p[i, j] = matriz_p[j, i] = p_corr[k] # Asignar p-value corregido

        # Añadimos resultados al informe
        lines.append("Resultados comparaciones 2 a 2 (Wilcoxon + Holm):")
        for k, (i, j) in enumerate(pares):
            lines.append(f"    {modelos[i]} vs {modelos[j]}: p‑valor corregido = {p_corr[k]:.4e}")

        # Destacamos solo los pares con diferencias significativas
        sig = [f"    {modelos[i]} vs {modelos[j]}: p‑valor corregido = {p_corr[k]:.4e}"
               for k, (i, j) in enumerate(pares) if p_corr[k] < alpha]
        lines.append("\nComparaciones con diferencia significativa:")
        lines.extend(sig or ["    Ninguna encontrada."])
    else:
        # Si no hay diferencias globales, no hacemos comparaciones pareadas
        matriz_p = None
        lines.append("No se realizan comparaciones 2 a 2 porque el test global no es significativo.")

    # ===== 5. Guardamos informe de texto =====
    ruta_txt = os.path.join(output_dir, f"p_value_analysis_{metric_col}.txt")
    with open(ruta_txt, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    print(f"  → Informe guardado en: {ruta_txt}")

    # ===== 6. Boxplot =====
    plt.figure(figsize=(10, 6))
    pivot.boxplot(color='black',
                  boxprops=dict(color='black'),
                  medianprops=dict(color='black'),
                  whiskerprops=dict(color='black'),
                  capprops=dict(color='black'),
                  flierprops=dict(color='black'))
    
    # plt.title(f"Distribución de {metric_col} por modelo")
    ylabel_dict = {
        'prob_class_1': 'Probabilidad de la clase positiva',
        'prob_class_0': 'Probabilidad de la clase negativa'
    }
    plt.ylabel(ylabel_dict.get(metric_col, metric_col))
    plt.xticks(rotation=45, ha='right')

    # Guardar boxplot 
    boxplot_path = os.path.join(output_dir, f"boxplot_{metric_col}.png")
    plt.savefig(boxplot_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  → Boxplot guardado en: {boxplot_path}")

    # ===== 7. Heatmap de p-values (solo si hay diferencias globales) =====
    if matriz_p is not None:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.grid(False)

        cax = ax.imshow(matriz_p, interpolation='nearest', aspect='auto', cmap='cividis')

        ax.set_xticks(np.arange(len(modelos)))
        ax.set_yticks(np.arange(len(modelos)))
        ax.set_xticklabels(modelos, rotation=45, ha='right')
        ax.set_yticklabels(modelos)

        ax.set_xticks(np.arange(-0.5, len(modelos), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(modelos), 1), minor=True)
        ax.grid(which='minor', color='black', linestyle='--', linewidth=1)
        ax.tick_params(which='minor', bottom=False, left=False)
        
        for i in range(len(modelos)):
            for j in range(len(modelos)):
                color = 'white' if matriz_p[i, j] < alpha else 'black'
                ax.text(j, i, f"{matriz_p[i, j]:.3f}", ha='center', va='center', color=color, fontsize=8)
        
        fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
        plt.tight_layout()

        # Guardar heatmap
        heatmap_path = os.path.join(output_dir, f"heatmap_pvalues_{metric_col}.png")
        plt.savefig(heatmap_path, dpi=dpi)
        plt.close()
        print(f"  → Heatmap p‑values guardado en: {heatmap_path}")

    return lines

def main():
    """
    Función principal que coordina el análisis estadístico.
    
    1. Procesa argumentos de línea de comandos
    2. Lee y combina archivos CSV de predicciones
    3. Ejecuta análisis estadístico para cada métrica especificada
    """
    
    # ===== Procesamiento de argumentos =====
    parser = argparse.ArgumentParser(
        description="Analiza diferencias estadísticas entre modelos usando patient_id como unidad."
    )
    parser.add_argument(
        "-i", "--predictions_dir", type=str, default="../../../../results/deep_learning/model_comparison/predict_&_analyse_probs/gland_analysis/predictions",
        help="Carpeta con los CSV de predicciones"
    )
    parser.add_argument(
        "-m", "--metric_col", nargs='+', default=["prob_class_1", "prob_class_0"],
        help="Una o más columnas de la métrica a analizar (separadas por espacio)"
    )
    parser.add_argument(
        "-a", "--alpha", type=float, default=0.05,
        help="Nivel de significación"
    )
    parser.add_argument(
        "-o", "--output_dir", type=str, default="../../../../results/deep_learning/model_comparison/predict_&_analyse_probs/gland_analysis/statistical_analysis",
        help="Directorio de salida"
    )
    args = parser.parse_args()

    # ===== Carga de datos =====
    # Buscar todos los archivos CSV en el directorio de predicciones
    csv_files = sorted(glob.glob(os.path.join(args.predictions_dir, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No se encontraron CSV en {args.predictions_dir}")

    # Combinar todos los CSVs en un único DataFrame
    df_all = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)
    print(f"Leídos {len(csv_files)} archivos, total de filas: {len(df_all)}")

    # ===== Análisis por métrica =====
    # Analizar cada métrica especificada
    for metric in args.metric_col:
        print(f"\n=== Analizando métrica: {metric} ===")
        perform_p_value_analysis(
            df=df_all,
            metric_col=metric,
            alpha=args.alpha,
            output_dir=args.output_dir
        )

if __name__ == "__main__":
    main()
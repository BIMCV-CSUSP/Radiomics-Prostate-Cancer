#!/usr/bin/env python
"""
Script para análisis estadístico de modelos de clasificación.

Este script analiza resultados de múltiples modelos de aprendizaje automático,
realizando comparaciones estadísticas (test de Friedman y Wilcoxon con corrección),
calculando tamaños de efecto (Cohen's d) y generando diversas visualizaciones
para facilitar la interpretación de resultados.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import re
import ast
from scipy import stats
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests
import argparse

# Configuración para visualizaciones
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import scienceplots

plt.style.use(['science', 'grid'])
dpi = 300

def extract_model_name(path):
    """
    Extrae el nombre del modelo de la ruta de un directorio.
    
    Args:
        path (str): Ruta al directorio del modelo
        
    Returns:
        str: Nombre del modelo extraído
    """
    return os.path.basename(path)

def read_results_folder(folder_path):
    """
    Lee todos los CSVs de resultados en una carpeta y los combina en un DataFrame.
    
    Busca archivos CSV con "split" en el nombre, extrae el número de split
    del nombre del archivo, y añade el nombre del modelo como columna.
    
    Args:
        folder_path (str): Ruta a la carpeta de resultados
        
    Returns:
        pd.DataFrame: DataFrame con todos los resultados combinados o DataFrame vacío si no hay resultados
    """
    all_data = []
    
    # Verifica si la carpeta existe
    if not os.path.exists(folder_path):
        print(f"La carpeta {folder_path} no existe")
        return pd.DataFrame()
    
    # Busca todos los archivos CSV en la carpeta que contengan "split" en el nombre
    csv_files = [f for f in os.listdir(folder_path) if f.endswith('.csv') and 'split' in f]
    
    if not csv_files:
        print(f"No se encontraron archivos CSV en {folder_path}")
        return pd.DataFrame()
    
    # Lee cada archivo CSV
    for file in csv_files:
        file_path = os.path.join(folder_path, file)
        try:
            df = pd.read_csv(file_path)
            
            # Extrae el número de split del nombre del archivo
            split_match = re.search(r'split_(\d+)_results', file)
            if split_match:
                split_num = int(split_match.group(1))
            else:
                split_num = 0
            
            # Si el split no está en el dataframe, añádelo
            if 'split' not in df.columns or pd.isna(df['split']).all():
                df['split'] = split_num
                
            all_data.append(df)
        except Exception as e:
            print(f"Error al leer {file_path}: {e}")
    
    if not all_data:
        return pd.DataFrame()
    
    # Concatena todos los DataFrames
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Añade el nombre del modelo como columna
    model_name = extract_model_name(folder_path)
    combined_df['model'] = model_name
    
    return combined_df

def find_results_folders(root_path):
    """
    Busca todas las carpetas de resultados que contengan archivos CSV.
    
    Recorre recursivamente el directorio raíz buscando carpetas que
    contengan archivos CSV de resultados (con "split" en el nombre).
    
    Args:
        root_path (str): Directorio raíz para la búsqueda
        
    Returns:
        list: Lista de rutas a las carpetas de resultados encontradas
    """
    results_folders = []
    
    for root, dirs, files in os.walk(root_path):
        # Busca carpetas con archivos CSV de resultados
        if any(f.endswith('.csv') and 'split' in f for f in files):
            results_folders.append(root)
    
    return results_folders

def process_string_lists(df):
    """
    Procesa columnas que contienen listas almacenadas como strings.
    
    Convierte las listas almacenadas como strings de Python en listas reales,
    y crea columnas separadas para cada elemento de la lista (para cada clase).
    
    Args:
        df (pd.DataFrame): DataFrame con columnas que contienen listas como strings
        
    Returns:
        pd.DataFrame: DataFrame procesado con nuevas columnas para cada clase
    """
    
    # Columnas que contienen listas como strings
    list_columns = ['per_class_precision', 'per_class_recall', 'per_class_f1', 'per_class_accuracy']
    
    for col in list_columns:
        if col in df.columns:
            # Convierte la cadena de texto a lista Python
            df[col] = df[col].apply(lambda x: ast.literal_eval(x) if isinstance(x, str) else x)
            
            # Crea columnas separadas para cada clase
            for i in range(2):
                df[f'{col}_class{i}'] = df[col].apply(lambda x: x[i] if isinstance(x, list) and len(x) > i else np.nan)
    
    return df

def plot_radar_chart(df, metrics, figsize=(10, 8), output_dir='plots'):
    """
    Genera un gráfico de radar para comparar modelos según múltiples métricas.
    
    Permite una visualización compacta de múltiples métricas simultáneamente,
    facilitando la comparación global entre modelos.
    
    Args:
        df (pd.DataFrame): DataFrame con estadísticas agregadas por modelo
        metrics (list): Lista de métricas a incluir en el gráfico
        figsize (tuple, optional): Tamaño de la figura
        output_dir (str, optional): Directorio para guardar el gráfico
        
    Returns:
        None: Guarda el gráfico como archivo PNG
    """
    # Preparación de datos: filtra para obtener solo la media de cada métrica
    radar_data = pd.DataFrame()
    
    for metric in metrics:
        if (metric, 'mean') in df.columns:
            radar_data[metric] = df[(metric, 'mean')]
    
    # Configuración del gráfico
    models = radar_data.index
    num_metrics = len(metrics)
    angles = np.linspace(0, 2*np.pi, num_metrics, endpoint=False).tolist()
    angles += angles[:1]  
    
    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    
    # Añade cada modelo al gráfico de radar
    for i, model in enumerate(models):
        values = radar_data.loc[model].values.flatten().tolist()
        values += values[:1]  
        ax.plot(angles, values, linewidth=2, label=model)
        ax.fill(angles, values, alpha=0.1)
    
    # Etiquetas y leyenda
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(metrics)
    ax.set_title('Comparación de modelos: métricas principales')
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    
    # Asegura que el directorio de salida existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Guarda la figura
    plt.savefig(os.path.join(output_dir, 'model_comparison_radar.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    return

def perform_statistical_analysis(best_results_df, metric_col, alpha=0.05, output_dir='stats'):
    """
    Realiza análisis estadístico completo para comparar modelos.
    
    Ejecuta test de Friedman para detectar diferencias globales entre modelos,
    seguido de comparaciones post-hoc con Wilcoxon y corrección múltiple.
    Calcula tamaños de efecto (Cohen's d) y genera visualizaciones.
    
    Args:
        best_results_df (pd.DataFrame): DataFrame con los mejores resultados por modelo y split
        metric_col (str): Nombre de la métrica a analizar
        alpha (float, optional): Nivel de significancia para los tests
        output_dir (str, optional): Directorio para guardar resultados
        
    Returns:
        list: Líneas de texto con el resumen del análisis estadístico
    """
    
    # Prepara el DataFrame en formato adecuado para el test de Friedman
    pivot_df = best_results_df.pivot_table(
        index='split', 
        columns='model', 
        values=metric_col
    )

    # Manejo de valores NaN
    if pivot_df.isnull().any().any():
        print("Advertencia: existen valores NaN en la tabla. Se eliminan filas con NaN.")
        pivot_df.dropna(axis=0, inplace=True)
    
    # Ordena los modelos por valor mediano descendente de la métrica
    median_metric_per_model = best_results_df.groupby("model")[metric_col].median().sort_values(ascending=False)
    ordered_models = median_metric_per_model.index.tolist()
    pivot_df = pivot_df[ordered_models]
    
    # Prepara datos para el test de Friedman
    data_for_friedman = [pivot_df[model].values for model in pivot_df.columns]
    
    # Test de Friedman
    stat, p_value = friedmanchisquare(*data_for_friedman)
    
    # Prepara el resumen del análisis
    summary_text = []
    summary_text.append("=================================")
    summary_text.append(f"TEST DE FRIEDMAN para métrica: {metric_col}")
    summary_text.append(f"Estadístico: {stat:.4f}, p-value: {p_value:.4e}")
    summary_text.append(f"alpha = {alpha}")
    
    # Interpretación del resultado del test de Friedman
    if p_value < alpha:
        summary_text.append("=> HAY diferencias estadísticamente significativas entre los modelos (rechazamos H0).")
    else:
        summary_text.append("=> NO se evidencian diferencias estadísticamente significativas entre los modelos (no se rechaza H0).")
    summary_text.append("=================================\n")
    
    # Variables para almacenar resultados
    pairwise_matrix = None
    effect_size_matrix = None
    pvalue_significant_pairs = []
    cohen_significant_pairs = []
    
    # Comparaciones post-hoc si el test global es significativo
    if p_value < alpha:
        models = pivot_df.columns.tolist()
        n_models = len(models)
        pvals = []
        pairs = []
        cohen_values = []
        all_pairs_summary = []
        
        # Calcula las comparaciones pareadas usando Wilcoxon
        for i in range(n_models):
            for j in range(i+1, n_models):
                scores_i = pivot_df.iloc[:, i].values
                scores_j = pivot_df.iloc[:, j].values
                
                # Test Wilcoxon (dos colas)
                try:
                    stat_w, p_val = wilcoxon(scores_i, scores_j, alternative='two-sided')
                except Exception as e:
                    p_val = np.nan  
                pvals.append(p_val)
                pairs.append((i, j))
                
                # Cálculo de Cohen's d para muestras pareadas:
                diff = scores_i - scores_j
                mean_diff = np.mean(diff)
                std_diff = np.std(diff, ddof=1)
                cohen_d = mean_diff / std_diff if std_diff != 0 else np.nan
                cohen_values.append(cohen_d)
        
        # Corrección múltiple de los p-values (método Holm)
        reject_array, pvals_corrected, _, _ = multipletests(pvals, alpha=alpha, method='holm')
        
        # Inicializa las matrices para p-values corregidos y tamaños del efecto
        pairwise_matrix = np.ones((n_models, n_models))
        effect_size_matrix = np.zeros((n_models, n_models))
        
        # Define un umbral para considerar un efecto como relevante (Cohen's d >= 0.5 es efecto medio)
        cohen_threshold = 0.5
        
        # Procesa resultados de las comparaciones pareadas
        idx = 0
        for (i, j) in pairs:
            p_corr = pvals_corrected[idx]
            cohen_d = cohen_values[idx]

            # Almacena valores en las matrices
            pairwise_matrix[i, j] = p_corr
            pairwise_matrix[j, i] = p_corr
            effect_size_matrix[i, j] = cohen_d
            effect_size_matrix[j, i] = cohen_d

            # Crea cadena de resultados para esta comparación
            result_str = f"    {models[i]} vs {models[j]}: p-value (Wilcoxon, corregido)={p_corr:.4e}, Cohen's d={cohen_d:.4f}"
            all_pairs_summary.append(result_str)
            
            # Identifica comparaciones significativas por p-value o tamaño del efecto
            if p_corr < alpha:
                pvalue_significant_pairs.append(result_str + " => DIFERENCIA SIGNIFICATIVA POR p-value")
            if abs(cohen_d) >= cohen_threshold:
                cohen_significant_pairs.append(result_str + " => DIFERENCIA SIGNIFICATIVA POR COHEN'S d")
            idx += 1
        
        # Añade resultados al resumen
        summary_text.append("Resultados comparaciones 2 a 2 (Wilcoxon + corrección múltiple y cálculo de Cohen's d):")
        for line in all_pairs_summary:
            summary_text.append(line)
            
        summary_text.append("\nComparaciones con diferencia significativa por p-value:")
        if pvalue_significant_pairs:
            for line in pvalue_significant_pairs:
                summary_text.append(line)
        else:
            summary_text.append("    No se encontraron diferencias significativas por p-value en comparaciones 2 a 2.")
        
        summary_text.append("\nComparaciones con diferencia significativa por Cohen's d:")
        if cohen_significant_pairs:
            for line in cohen_significant_pairs:
                summary_text.append(line)
        else:
            summary_text.append("    No se encontraron diferencias significativas por Cohen's d en comparaciones 2 a 2.")
    else:
        summary_text.append("No se realizan comparaciones 2 a 2 porque el test global no es significativo.")
    
    # Asegura que el directorio de salida existe
    os.makedirs(output_dir, exist_ok=True)
    
    # Guarda el resumen en un archivo de texto
    txt_path = os.path.join(output_dir, f"statistical_analysis_{metric_col}.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for line in summary_text:
            f.write(line + "\n")
    
    print(f"  --> Resumen estadístico guardado en: {txt_path}")
    
    # ======= Generación de visualizaciones =======

    # Genera boxplot de la métrica por modelo (ordenado por mediana descendente)
    plt.figure(figsize=(10, 6))
    boxprops = dict(color='black')
    medianprops = dict(color='black')
    whiskerprops = dict(color='black')
    capprops = dict(color='black')
    flierprops = dict(color='black')
    
    # Crea boxplot con propiedades personalizadas
    pivot_df.boxplot(
        boxprops=boxprops,
        medianprops=medianprops,
        whiskerprops=whiskerprops,
        capprops=capprops,
        flierprops=flierprops
    )
    plt.title(f"Distribución de {metric_col} por modelo")
    plt.ylabel(metric_col)
    plt.xticks(rotation=45, ha='right')
    
    boxplot_path = os.path.join(output_dir, f"boxplot_{metric_col}.png")
    plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  --> Boxplot guardado en: {boxplot_path}")
    

    # Genera heatmap de p-values post-hoc (Wilcoxon con corrección múltiple)
    if pairwise_matrix is not None:
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.grid(False)
        cax = ax.imshow(pairwise_matrix, interpolation='nearest', cmap='cividis', aspect='auto')
    
        ax.set_title("Matriz de p-values (Wilcoxon con corrección múltiple)")
        ax.set_xticks(np.arange(len(models)))
        ax.set_yticks(np.arange(len(models)))
        ax.set_xticklabels(models, rotation=45, ha="right")
        ax.set_yticklabels(models)
    
        ax.set_xticks(np.arange(-0.5, len(models), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(models), 1), minor=True)
        ax.grid(which='minor', color='black', linestyle='--', linewidth=1)
        ax.tick_params(which='minor', bottom=False, left=False)
    
        for i in range(len(models)):
            for j in range(len(models)):
                pval_ij = pairwise_matrix[i, j]
                text_color = "white" if pval_ij < alpha else "black"
                ax.text(j, i, f"{pval_ij:.3f}", ha="center", va="center", color=text_color, fontsize=8)
    
        fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    
        heatmap_path = os.path.join(output_dir, f"heatmap_pvalues_{metric_col}.png")
        plt.tight_layout()
        plt.savefig(heatmap_path, dpi=300)
        plt.close()
        print(f"  --> Heatmap de p-values guardado en: {heatmap_path}")
    

    # Genera heatmap del tamaño del efecto (Cohen's d)
    if effect_size_matrix is not None:
        fig, ax = plt.subplots(figsize=(8, 7))
        ax.grid(False)
        cax = ax.imshow(effect_size_matrix, interpolation='nearest', cmap='cividis', aspect='auto')
    
        ax.set_title("Matriz de tamaño del efecto (Cohen's d)")
        ax.set_xticks(np.arange(len(models)))
        ax.set_yticks(np.arange(len(models)))
        ax.set_xticklabels(models, rotation=45, ha="right")
        ax.set_yticklabels(models)
    
        ax.set_xticks(np.arange(-0.5, len(models), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(models), 1), minor=True)
        ax.grid(which='minor', color='black', linestyle='--', linewidth=1)
        ax.tick_params(which='minor', bottom=False, left=False)
    
        for i in range(len(models)):
            for j in range(len(models)):
                es_val = effect_size_matrix[i, j]
                ax.text(j, i, f"{es_val:.3f}", ha="center", va="center", color="black", fontsize=8)
    
        fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    
        effect_heatmap_path = os.path.join(output_dir, f"heatmap_effectsize_{metric_col}.png")
        plt.tight_layout()
        plt.savefig(effect_heatmap_path, dpi=300)
        plt.close()
        print(f"  --> Heatmap de tamaño del efecto guardado en: {effect_heatmap_path}")
    
    return summary_text

def analyze_results(root_path, output_base='results_analysis'):
    """
    Función principal que coordina todo el proceso de análisis de resultados.
    
    Esta función:
    1. Encuentra carpetas con resultados
    2. Combina todos los resultados en un DataFrame
    3. Identifica el epoch con AUC óptimo para cada modelo/split
    4. Realiza análisis estadístico comparativo (Friedman + Wilcoxon)
    5. Genera visualizaciones (boxplots, heatmaps, gráficos de radar, etc.)
    
    Args:
        root_path (str): Ruta raíz donde buscar carpetas de resultados
        output_base (str, optional): Directorio base para guardar el análisis
        
    Returns:
        None
    """

    # Configura directorios de salida
    csv_dir = os.path.join(output_base, 'csv')
    stats_dir = os.path.join(output_base, 'statistical_analysis')
    plots_dir = os.path.join(output_base, 'general_plots')
    
    # Crea directorios si no existen
    for dir_path in [csv_dir, stats_dir, plots_dir]:
        os.makedirs(dir_path, exist_ok=True)
    
    # Busca carpetas con resultados
    results_folders = find_results_folders(root_path)
    
    if not results_folders:
        print(f"No se encontraron carpetas de resultados en {root_path}")
        return
    
    print(f"Se encontraron {len(results_folders)} carpetas de resultados")
    
    ######################################
    #      Concatenando resultados       #
    ######################################
    
    all_results = []
    for folder in results_folders:
        model_results = read_results_folder(folder)
        if not model_results.empty:
            if 'model' not in model_results.columns:
                model_name = extract_model_name(folder)
                print(f"Añadiendo columna 'model' a resultados de {model_name}")
                model_results['model'] = model_name
            
            all_results.append(model_results)
    
    if not all_results:
        print("No se encontraron resultados en ninguna carpeta")
        return
    
    # Combina todos los resultados en un único DataFrame
    combined_results = pd.concat(all_results, ignore_index=True)
    combined_results = process_string_lists(combined_results)
    combined_results = pd.concat(all_results, ignore_index=True)

    # Reorganiza columnas para poner 'model' primero (si existe)
    if 'model' in combined_results.columns:
        cols = combined_results.columns.tolist()
        cols.remove('model')
        combined_results = combined_results[['model'] + cols]
    
    # Guarda resultados combinados
    combined_results.to_csv(os.path.join(csv_dir, 'all_results_combined.csv'), index=False)
    print(f"Resultados combinados guardados en: {os.path.join(csv_dir, 'all_results_combined.csv')}")
    
    ######################################
    #      Epoch con AUC óptimo          #
    ######################################
    
    # Para cada modelo y split, encuentra la época con mejor AUC
    optimal_results = []
    for model in combined_results['model'].unique():
        for split in combined_results[combined_results['model'] == model]['split'].unique():
            model_split_data = combined_results[(combined_results['model'] == model) & 
                                              (combined_results['split'] == split)]
            
            if 'val_auc' in model_split_data.columns:
                # Encontrar el índice con el AUC máximo (óptimo)
                optimal_idx = model_split_data['val_auc'].idxmax()
                optimal_row = model_split_data.loc[optimal_idx].to_dict()
                
                optimal_row['model'] = model
                optimal_row['split'] = split
                
                optimal_results.append(optimal_row)
    
    # Crea DataFrame con los resultados óptimos
    optimal_results_df = pd.DataFrame(optimal_results)

    # Guarda resultados óptimos
    optimal_results_df.to_csv(os.path.join(csv_dir, 'optimal_auc_results.csv'), index=False)
    print(f"Resultados con AUC óptimo guardados en: {os.path.join(csv_dir, 'optimal_auc_results.csv')}")
    
    ########################################################
    #      Análisis estadístico (Friedman + Wilcoxon)      #
    ########################################################

    # Realiza análisis estadístico para el AUC (si está disponible)
    if 'val_auc' in optimal_results_df.columns:
        perform_statistical_analysis(
            optimal_results_df,
            'val_auc', 
            alpha=0.05, 
            output_dir=stats_dir
        )
    else:
        print("No se encontró la métrica val_auc para realizar análisis estadístico")
    
    ##################################
    #      Gráficos adicionales      #
    ##################################
    
    # Define métricas a visualizar
    plot_metrics = [
        'val_auc', 'val_f1_binary', 'val_accuracy', 
        'val_balanced_accuracy', 'val_specificity', 'val_sensitivity',
    ]

    # Filtra métricas disponibles
    existing_metrics = [m for m in plot_metrics if m in combined_results.columns]
    if not existing_metrics:
        print("No se encontraron métricas de validación en los datos")
        return
    
    # Calcula estadísticas resumen por modelo
    summary_stats = optimal_results_df.groupby('model')[existing_metrics].agg(['mean', 'std', 'min', 'max', 'median'])
    summary_stats.to_csv(os.path.join(csv_dir, 'model_summary_statistics.csv'))
    print(f"Estadísticas resumen guardadas en: {os.path.join(csv_dir, 'model_summary_statistics.csv')}")
    
    # Gráfico de radar
    plot_radar_chart(
        summary_stats, 
        [m for m in existing_metrics], 
        output_dir=plots_dir
    )
    print(f"Gráfico de radar guardado en: {os.path.join(plots_dir, 'model_comparison_radar.png')}")
    
    # Gráficos de barras para cada métrica
    plt.figure(figsize=(14, 10))
    for i, metric in enumerate(existing_metrics):
        plt.subplot(2, 3, i+1)
        
        # Calculamos la mediana y ordenamos los modelos en base a ella
        median_values = optimal_results_df.groupby('model')[metric].median().sort_values(ascending=False)
        ordered_models = median_values.index
        
        # Usamos barplot con la mediana como estimador y sin barras de error
        sns.barplot(
            x='model',
            y=metric,
            data=optimal_results_df,
            order=ordered_models,
            estimator=np.median,
            errorbar=None
        )
        
        plt.title(f'Comparación de {metric}')
        plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, 'metrics_comparison_barplot.png'), dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Gráfico de barras comparativo guardado en: {os.path.join(plots_dir, 'metrics_comparison_barplot.png')}")
    
    print("\nAnálisis completado exitosamente.")

def main():
    """
    Función de punto de entrada para el script cuando se ejecuta directamente.
    
    Procesa argumentos de línea de comandos y ejecuta el análisis de resultados.
    
    Argumentos de línea de comandos:
        --mode: Tipo de resultados a analizar ('gland' o 'full')
        --data_root: Directorio raíz donde se ubican las carpetas de resultados
        --output_base: Prefijo para el directorio de salida del análisis
    """
    parser = argparse.ArgumentParser(
        description="Análisis completo de resultados de modelos de machine learning."
    )
    parser.add_argument(
        "--mode", type=str, choices=["gland", "full"], default="gland",
        help="Tipo de resultados a analizar: 'gland' o 'full'."
    )
    parser.add_argument(
        "--data_root", type=str, default="../../../../artifacts/deep_learning",
        help="Directorio raíz donde se ubican las carpetas de resultados."
    )
    parser.add_argument(
        "--output_base", type=str, default="../../../../results/deep_learning/model_comparison/simple_statistical_analysis/",
        help="Prefijo para el directorio de salida del análisis."
    )

    args = parser.parse_args()

    # Construye las rutas específicas basadas en los argumentos
    root_path = os.path.join(args.data_root, args.mode, "results")
    output_dir = os.path.join(args.output_base, f"{args.mode}_analysis")

    # Ejecuta el análisis con las rutas especificadas
    analyze_results(root_path, output_dir)

if __name__ == "__main__":
    main()
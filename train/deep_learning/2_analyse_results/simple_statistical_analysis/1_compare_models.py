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
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests
from sklearn import metrics
import argparse
import matplotlib
matplotlib.use('Agg')
import scienceplots

plt.style.use(['science', 'grid'])
dpi = 300

def find_csv_files(folder_path):
    """
    Devuelve una lista con la ruta completa de todos los archivos CSV en la carpeta indicada.
    """
    return [os.path.join(folder_path, f) for f in os.listdir(folder_path)
            if f.endswith('.csv')]

def get_model_metrics_from_preds(csv_file):
    """
    Calcula métricas por split a partir de CSV de predicciones con columnas:
    split, true_label, prob_class_1
    Devuelve un DataFrame con columnas: model, split, test_auc, test_f1, ...
    """
    df = pd.read_csv(csv_file)
    model_name = os.path.splitext(os.path.basename(csv_file))[0].replace('_predictions', '')
    metrics_rows = []
    if 'split' not in df.columns or 'true_label' not in df.columns or 'prob_class_1' not in df.columns:
        print(f"Archivo {csv_file} no tiene columnas necesarias, ignorado.")
        return pd.DataFrame()
    for split, part in df.groupby('split'):
        y_true = part['true_label'].values
        y_prob = part['prob_class_1'].values
        y_pred = (y_prob >= 0.5).astype(int)
        if len(np.unique(y_true)) < 2:
            auc = np.nan  # No se puede calcular si sólo hay una clase
        else:
            auc = metrics.roc_auc_score(y_true, y_prob)
        metrics_rows.append({
            'model': model_name,
            'split': int(split),
            'test_auc': auc,
            'test_f1': metrics.f1_score(y_true, y_pred),
            'test_accuracy': metrics.accuracy_score(y_true, y_pred),
            'test_balanced_accuracy': metrics.balanced_accuracy_score(y_true, y_pred),
            'test_sensitivity': metrics.recall_score(y_true, y_pred, pos_label=1),
            'test_specificity': metrics.recall_score(y_true, y_pred, pos_label=0)
        })
    return pd.DataFrame(metrics_rows)

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
    # ax.set_title('Comparación de modelos: métricas principales')
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
    # plt.title(f"Distribución de {metric_col} por modelo")
    plt.ylabel("AUC en validación")
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
    
        # ax.set_title("Matriz de p-values (Wilcoxon con corrección múltiple)")
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
    
        # ax.set_title("Matriz de tamaño del efecto (Cohen's d)")
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
    # Configura directorios de salida
    csv_dir = os.path.join(output_base, 'csv')
    stats_dir = os.path.join(output_base, 'statistical_analysis')
    plots_dir = os.path.join(output_base, 'general_plots')
    for dir_path in [csv_dir, stats_dir, plots_dir]:
        os.makedirs(dir_path, exist_ok=True)

    # Lee los archivos CSV de predicciones
    csv_files = find_csv_files(root_path)
    if not csv_files:
        print(f"No se encontraron archivos CSV en {root_path}")
        return

    print(f"Se encontraron {len(csv_files)} archivos CSV en {root_path}")
    all_results = []
    for file in csv_files:
        model_metrics = get_model_metrics_from_preds(file)
        if not model_metrics.empty:
            all_results.append(model_metrics)
    if not all_results:
        print("No se encontraron resultados válidos en los CSVs")
        return

    combined_results = pd.concat(all_results, ignore_index=True)
    combined_results.to_csv(os.path.join(csv_dir, 'all_results_combined.csv'), index=False)
    print(f"Resultados combinados guardados en: {os.path.join(csv_dir, 'all_results_combined.csv')}")

    ########################################################
    #      Análisis estadístico (Friedman + Wilcoxon)      #
    ########################################################

    if 'test_auc' in combined_results.columns:
        perform_statistical_analysis(
            combined_results,
            'test_auc',
            alpha=0.05,
            output_dir=stats_dir
        )
    else:
        print("No se encontró la métrica test_auc para realizar análisis estadístico")

    ##################################
    #      Gráficos adicionales      #
    ##################################
    plot_metrics = [
        'test_auc', 'test_f1', 'test_accuracy',
        'test_balanced_accuracy', 'test_specificity', 'test_sensitivity',
    ]
    existing_metrics = [m for m in plot_metrics if m in combined_results.columns]
    if not existing_metrics:
        print("No se encontraron métricas de validación en los datos")
        return

    # Calcula estadísticas resumen por modelo
    summary_stats = combined_results.groupby('model')[existing_metrics].agg(['mean', 'std', 'min', 'max', 'median'])
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
        median_values = combined_results.groupby('model')[metric].median().sort_values(ascending=False)
        ordered_models = median_values.index
        sns.barplot(
            x='model',
            y=metric,
            data=combined_results,
            order=ordered_models,
            estimator=np.median,
            errorbar=None
        )
        # plt.title(f'Comparación de {metric}')
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
        "--data_root", type=str, default="../../../../results/deep_learning/model_comparison/predict_&_analyse_probs/",
        help="Directorio raíz donde se ubican las carpetas de resultados."
    )
    parser.add_argument(
        "--output_base", type=str, default="../../../../results/deep_learning/model_comparison/simple_statistical_analysis/",
        help="Prefijo para el directorio de salida del análisis."
    )

    args = parser.parse_args()

    # Construye las rutas específicas basadas en los argumentos
    root_path = os.path.join(args.data_root, f"{args.mode}_analysis", "predictions")
    output_dir = os.path.join(args.output_base, f"{args.mode}_analysis")

    # Ejecuta el análisis con las rutas especificadas
    analyze_results(root_path, output_dir)

if __name__ == "__main__":
    main()
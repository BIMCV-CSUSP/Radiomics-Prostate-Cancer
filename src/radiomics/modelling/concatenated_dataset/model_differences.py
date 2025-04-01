#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import scienceplots

plt.style.use(['science', 'grid'])
dpi = 300

from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests

def main():
    parser = argparse.ArgumentParser(
        description="Comparación estadística de clasificadores (test global y post-hoc)."
    )
    parser.add_argument("--csv_preds", type=str, help="Ruta al CSV con predicciones", required=True)
    parser.add_argument("--csv_results", type=str, help="Ruta al CSV con las métricas por fold", required=True)
    parser.add_argument("--metric", type=str, default="val_auc", help="Métrica a comparar entre clasificadores")
    parser.add_argument("--alpha", type=float, default=0.05, help="Nivel de significancia")
    parser.add_argument("--outdir", type=str, default=".", help="Directorio de salida")

    args = parser.parse_args()
    
    df_results = pd.read_csv(args.csv_results)
   
    metric_col = args.metric
    alpha = args.alpha

    classifiers = df_results["Classifier"].unique()
    df_results = df_results.sort_values(by=["Classifier", "Fold"])

    pivot_df = df_results.pivot_table(
        index="Fold", 
        columns="Classifier", 
        values=metric_col
    )

    if pivot_df.isnull().any().any():
        print("Advertencia: existen valores NaN en la tabla. Se pueden remover o imputar.")
        pivot_df.dropna(axis=0, inplace=True)
    
    median_auc_per_clf = df_results.groupby("Classifier")[metric_col].median().sort_values(ascending=False)
    ordered_classifiers = median_auc_per_clf.index.tolist()
    pivot_df = pivot_df[ordered_classifiers]

    data_for_friedman = []
    for clf in pivot_df.columns:
        data_for_friedman.append(pivot_df[clf].values)

    # Test de Friedman
    stat, p_value = friedmanchisquare(*data_for_friedman)

    summary_text = []
    summary_text.append("=================================")
    summary_text.append(f"TEST DE FRIEDMAN para métrica: {metric_col}")
    summary_text.append(f"Estadístico: {stat:.4f}, p-value: {p_value:.4e}")
    summary_text.append(f"alpha = {alpha}")
    
    if p_value < alpha:
        summary_text.append("=> HAY diferencias estadísticamente significativas entre los clasificadores (rechazamos H0).")
    else:
        summary_text.append("=> NO se evidencian diferencias estadísticamente significativas entre los clasificadores (no se rechaza H0).")
    summary_text.append("=================================\n")
    
    # Comparaciones post-hoc 2 a 2 si el test global es significativo
    pairwise_matrix = None
    if p_value < alpha:
        clfs = pivot_df.columns.tolist()
        n_clfs = len(clfs)
        
        pairwise_matrix = np.ones((n_clfs, n_clfs))
        
        pvals = []
        pairs = []
        
        for i in range(n_clfs):
            for j in range(i+1, n_clfs):
                scores_i = pivot_df.iloc[:, i].values
                scores_j = pivot_df.iloc[:, j].values
                
                # Test Wilcoxon (dos colas)
                w_stat, p_val_pair = wilcoxon(scores_i, scores_j, alternative='two-sided')
                pvals.append(p_val_pair)
                pairs.append((i, j))

        # Corrección de comparaciones múltiples
        reject_array, pvals_corrected, _, _ = multipletests(pvals, alpha=alpha, method='holm')
        
        for (i, j), p_corr, reject_bool in zip(pairs, pvals_corrected, reject_array):
            pairwise_matrix[i, j] = p_corr
            pairwise_matrix[j, i] = p_corr
        
        summary_text.append("Resultados comparaciones 2 a 2 (Wilcoxon + corrección múltiple):")
        all_pairs_summary = []
        significant_pairs_summary = []
        for (i, j), p_corr, reject_bool in zip(pairs, pvals_corrected, reject_array):
            c1, c2 = clfs[i], clfs[j]
            result_str = f"    {c1} vs {c2}: p-value corregido={p_corr:.4e}"
            if reject_bool:
                result_str += " => DIFERENCIA SIGNIFICATIVA"
                significant_pairs_summary.append(result_str)
            else:
                result_str += " => sin diferencia significativa"
            all_pairs_summary.append(result_str)
        
        for line in all_pairs_summary:
            summary_text.append(line)
        summary_text.append("\nComparaciones con diferencia significativa:")
        if significant_pairs_summary:
            for line in significant_pairs_summary:
                summary_text.append(line)
        else:
            summary_text.append("    No se encontraron diferencias significativas en comparaciones 2 a 2.")
    else:
        summary_text.append("No se realizan comparaciones 2 a 2 porque el test global no es significativo.")
    
    # Guardar el resumen en un archivo de texto
    os.makedirs(args.outdir, exist_ok=True)
    txt_path = os.path.join(args.outdir, "model_differences_summary.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        for line in summary_text:
            f.write(line + "\n")
    
    print(f"  --> Resumen estadístico guardado en: {txt_path}")

    # 10) Gráfico 1: Boxplot de la métrica por clasificador (ordenado por AUC mediano descendente)
    boxprops = dict(color='black')                        
    medianprops = dict(color='black')                     
    whiskerprops = dict(color='black')                    
    capprops = dict(color='black')                        
    flierprops = dict(color='black')                      
    
    plt.figure(figsize=(8, 5))
    pivot_df.boxplot(
        boxprops=boxprops,
        medianprops=medianprops,
        whiskerprops=whiskerprops,
        capprops=capprops,
        flierprops=flierprops
    )
    plt.title("Distribución del AUC en validación por clasificador")
    plt.ylabel("AUC en validación")
    plt.xticks(rotation=45, ha='right')
    
    boxplot_path = os.path.join(args.outdir, "boxplot_metric.png")
    plt.savefig(boxplot_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  --> Boxplot guardado en: {boxplot_path}")

    # 11) Gráfico 2: Heatmap de p-values post-hoc (si se hicieron comparaciones)
    if pairwise_matrix is not None:
        fig, ax = plt.subplots(figsize=(6, 5))
        ax.grid(False)
        cax = ax.imshow(pairwise_matrix, interpolation='nearest', cmap='cividis', aspect='auto')
    
        ax.set_title("Matriz de p-values corregidos (Wilcoxon)")
    
        ax.set_xticks(np.arange(len(clfs)))
        ax.set_yticks(np.arange(len(clfs)))
        ax.set_xticklabels(clfs, rotation=45, ha="right")
        ax.set_yticklabels(clfs)
    
        ax.set_xticks(np.arange(-0.5, len(clfs), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(clfs), 1), minor=True)
    
        ax.grid(which='minor', color='black', linestyle='--', linewidth=1)
        ax.tick_params(which='minor', bottom=False, left=False)
    
        for i in range(len(clfs)):
            for j in range(len(clfs)):
                pval_ij = pairwise_matrix[i, j]
                text_color = "white" if pval_ij < 0.05 else "black"
                ax.text(j, i, f"{pval_ij:.3f}", ha="center", va="center", color=text_color, fontsize=8)
    
        fig.colorbar(cax, ax=ax, fraction=0.046, pad=0.04)
    
        heatmap_path = os.path.join(args.outdir, "heatmap_pvalues.png")
        plt.tight_layout()
        plt.savefig(heatmap_path, dpi=300)
        plt.close()
        print(f"  --> Heatmap de p-values guardado en: {heatmap_path}")

if __name__ == "__main__":
    main()
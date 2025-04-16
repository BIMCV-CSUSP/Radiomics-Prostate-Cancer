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
    1) Test de Friedman para diferencias globales en `metric_col`.
    2) Comparaciones pareadas Wilcoxon (dos colas) con corrección Holm.
    3) Boxplot de `metric_col` por modelo (todo en negro).
    4) Heatmap de p‑values corregidos (si el test global es significativo).
    5) Informe de texto en formato detallado.

    ***Esta versión usa patient_id como unidad experimental.***
    """
    os.makedirs(output_dir, exist_ok=True)

    # --- 1. Matriz paciente × modelo ---
    pivot = (df.pivot_table(index='patient_id',      
                            columns='model',
                            values=metric_col, 
                            aggfunc='mean')          
              .dropna(axis=0))                      

    orden = (df.groupby('model')[metric_col]
               .median()
               .sort_values(ascending=False)
               .index.tolist())
    pivot = pivot[orden]

    # --- 2. Test de Friedman ---
    datos = [pivot[col].values for col in pivot.columns]
    stat, p_global = friedmanchisquare(*datos)

    # --- 3. Preparamos informe ---
    lines = []
    lines.append("=================================")
    lines.append(f"TEST DE FRIEDMAN por paciente | métrica: {metric_col}")
    lines.append(f"Estadístico: {stat:.4f}, p-value: {p_global:.4e}")
    lines.append(f"alpha = {alpha}")
    if p_global < alpha:
        lines.append("=> HAY diferencias estadísticamente significativas entre los modelos (rechazamos H0).")
    else:
        lines.append("=> NO se evidencian diferencias estadísticamente significativas entre los modelos (no se rechaza H0).")
    lines.append("=================================\n")

    # --- 4. Comparaciones 2 a 2 si procede ---
    if p_global < alpha:
        modelos = pivot.columns.tolist()
        n = len(modelos)
        pvals, pares = [], []

        for i in range(n):
            for j in range(i+1, n):
                xi, xj = pivot.iloc[:, i].values, pivot.iloc[:, j].values
                try:
                    _, p = wilcoxon(xi, xj, alternative='two-sided')
                except:
                    p = np.nan
                pvals.append(p)
                pares.append((i, j))

        # Corrección Holm
        _, p_corr, _, _ = multipletests(pvals, alpha=alpha, method='holm')

        # Matriz simétrica de p‑values corregidos
        matriz_p = np.ones((n, n))
        for k, (i, j) in enumerate(pares):
            matriz_p[i, j] = matriz_p[j, i] = p_corr[k]

        # Informe comparaciones
        lines.append("Resultados comparaciones 2 a 2 (Wilcoxon + Holm):")
        for k, (i, j) in enumerate(pares):
            lines.append(f"    {modelos[i]} vs {modelos[j]}: p‑valor corregido = {p_corr[k]:.4e}")

        # Pares significativos
        sig = [f"    {modelos[i]} vs {modelos[j]}: p‑valor corregido = {p_corr[k]:.4e}"
               for k, (i, j) in enumerate(pares) if p_corr[k] < alpha]
        lines.append("\nComparaciones con diferencia significativa:")
        lines.extend(sig or ["    Ninguna encontrada."])
    else:
        matriz_p = None
        lines.append("No se realizan comparaciones 2 a 2 porque el test global no es significativo.")

    # --- 5. Guardamos informe ---
    ruta_txt = os.path.join(output_dir, f"p_value_analysis_{metric_col}.txt")
    with open(ruta_txt, 'w', encoding='utf-8') as f:
        f.write("\n".join(lines))
    print(f"  → Informe guardado en: {ruta_txt}")

    # --- 6. Boxplot en negro ---
    plt.figure(figsize=(10, 6))
    pivot.boxplot(color='black',
                  boxprops=dict(color='black'),
                  medianprops=dict(color='black'),
                  whiskerprops=dict(color='black'),
                  capprops=dict(color='black'),
                  flierprops=dict(color='black'))
    plt.title(f"Distribución de {metric_col} por modelo")
    plt.ylabel(metric_col)
    plt.xticks(rotation=45, ha='right')
    boxplot_path = os.path.join(output_dir, f"boxplot_{metric_col}.png")
    plt.savefig(boxplot_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  → Boxplot guardado en: {boxplot_path}")

    # --- 7. Heatmap de p‑values ---
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
        heatmap_path = os.path.join(output_dir, f"heatmap_pvalues_{metric_col}.png")
        plt.savefig(heatmap_path, dpi=dpi)
        plt.close()
        print(f"  → Heatmap p‑values guardado en: {heatmap_path}")

    return lines

def main():
    parser = argparse.ArgumentParser(
        description="Analiza diferencias estadísticas entre modelos usando patient_id como unidad."
    )
    parser.add_argument(
        "-i", "--predictions_dir", type=str, default="predictions",
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
        "-o", "--output_dir", type=str, default="statistical_analysis",
        help="Directorio de salida"
    )
    args = parser.parse_args()

    csv_files = sorted(glob.glob(os.path.join(args.predictions_dir, "*.csv")))
    if not csv_files:
        raise FileNotFoundError(f"No se encontraron CSV en {args.predictions_dir}")

    df_all = pd.concat((pd.read_csv(f) for f in csv_files), ignore_index=True)
    print(f"Leídos {len(csv_files)} archivos, total de filas: {len(df_all)}")

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
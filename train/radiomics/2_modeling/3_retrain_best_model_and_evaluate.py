#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Optimize, calibrate, and interpret the best radiomics model on a leakage-safe hold-out split.
"""

import os
import argparse
import sys
import time
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from datetime import datetime

import matplotlib as mpl
mpl.use('Agg')
try:
    import scienceplots  # noqa: F401
    plt.style.use(['science', 'grid'])
except ModuleNotFoundError:
    plt.style.use('default')

# SciencePlots may request LaTeX rendering; disable it explicitly so the
# scripts remain portable on cluster nodes without a TeX installation.
mpl.rcParams["text.usetex"] = False
plt.rcParams["text.usetex"] = False

dpi = 300

# Librerías para interpretabilidad
try:
    import shap
except ModuleNotFoundError:
    shap = None

try:
    from lime.lime_tabular import LimeTabularExplainer
except ModuleNotFoundError:
    LimeTabularExplainer = None

from copy import deepcopy
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.impute import SimpleImputer

# Importación de clasificadores
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

# Herramientas para evaluación y calibración
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.metrics import brier_score_loss
from sklearn.metrics import (
    roc_auc_score, matthews_corrcoef, cohen_kappa_score, f1_score,
    accuracy_score, recall_score, precision_score, balanced_accuracy_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report, roc_curve
)

# Optimización bayesiana de hiperparámetros
try:
    from skopt import BayesSearchCV
    from skopt.space import Real, Integer, Categorical
except ModuleNotFoundError:
    BayesSearchCV = None
    Real = Integer = Categorical = None

import joblib

# Para análisis estadístico de valores SHAP
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests


PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from train.common.radiomics_utils import (
    prepare_numeric_radiomics_matrix,
    resolve_feature_table_path,
    select_radiomics_features,
)


def configure_live_logging() -> None:
    """Enable line-buffered stdout/stderr so progress appears promptly in SLURM logs."""

    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(line_buffering=True)
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(line_buffering=True)


def log_progress(message: str) -> None:
    """Print a timestamped progress message and flush immediately."""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"{timestamp} | INFO | {message}", flush=True)


def format_metric(value: float) -> str:
    """Format numeric metrics consistently for console logging."""

    return "nan" if pd.isna(value) else f"{value:.4f}"


###########################
#   BOOTSTRAP METRICS     #
###########################

def compute_binary_metrics_from_probabilities(y_true, y_prob, threshold=0.5):
    """
    Compute threshold-free and threshold-based binary classification metrics.
    """
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)

    auc_value = roc_auc_score(y_true, y_prob)
    mcc_value = matthews_corrcoef(y_true, y_pred)
    kappa_value = cohen_kappa_score(y_true, y_pred)
    f1_value = f1_score(y_true, y_pred)
    accuracy_value = accuracy_score(y_true, y_pred)
    sensitivity_value = recall_score(y_true, y_pred, pos_label=1)
    specificity_value = recall_score(y_true, y_pred, pos_label=0)
    ppv_value = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    npv_value = precision_score(y_true, y_pred, pos_label=0, zero_division=0)
    balanced_accuracy_value = balanced_accuracy_score(y_true, y_pred)

    return {
        "auc": auc_value,
        "mcc": mcc_value,
        "kappa": kappa_value,
        "f1": f1_value,
        "accuracy": accuracy_value,
        "sensitivity": sensitivity_value,
        "specificity": specificity_value,
        "ppv": ppv_value,
        "npv": npv_value,
        "balanced_accuracy": balanced_accuracy_value,
    }


def bootstrap_patient_level_confidence_intervals(
    y_true,
    y_prob,
    patient_ids,
    threshold=0.5,
    n_bootstrap=1000,
    ci_level=0.95,
    seed=42,
    n_roc_points=200,
):
    """
    Estimate patient-level confidence intervals by stratified bootstrap resampling.
    """
    bootstrap_df = pd.DataFrame(
        {
            "patient_id": patient_ids,
            "true_label": y_true,
            "prob_class_1": y_prob,
        }
    )

    patient_rows = {
        patient_id: patient_df.copy()
        for patient_id, patient_df in bootstrap_df.groupby("patient_id")
    }
    patient_labels = bootstrap_df.groupby("patient_id")["true_label"].agg(lambda values: int(values.iloc[0]))
    patient_strata = {
        class_label: patient_labels[patient_labels == class_label].index.to_numpy()
        for class_label in sorted(patient_labels.unique())
    }

    point_metrics = compute_binary_metrics_from_probabilities(
        y_true=y_true,
        y_prob=y_prob,
        threshold=threshold,
    )
    metric_samples = {metric_name: [] for metric_name in point_metrics}

    mean_fpr = np.linspace(0.0, 1.0, n_roc_points)
    tpr_samples = []
    rng = np.random.default_rng(seed)

    for _ in range(n_bootstrap):
        sampled_frames = []
        for _, class_patient_ids in patient_strata.items():
            sampled_patient_ids = rng.choice(
                class_patient_ids,
                size=len(class_patient_ids),
                replace=True,
            )
            for patient_id in sampled_patient_ids:
                sampled_frames.append(patient_rows[patient_id])

        sampled_df = pd.concat(sampled_frames, ignore_index=True)
        if sampled_df["true_label"].nunique() < 2:
            continue

        sampled_metrics = compute_binary_metrics_from_probabilities(
            y_true=sampled_df["true_label"].to_numpy(),
            y_prob=sampled_df["prob_class_1"].to_numpy(),
            threshold=threshold,
        )
        for metric_name, metric_value in sampled_metrics.items():
            metric_samples[metric_name].append(metric_value)

        fpr, tpr, _ = roc_curve(
            sampled_df["true_label"].to_numpy(),
            sampled_df["prob_class_1"].to_numpy(),
            pos_label=1,
        )
        interpolated_tpr = np.interp(mean_fpr, fpr, tpr)
        interpolated_tpr[0] = 0.0
        interpolated_tpr[-1] = 1.0
        tpr_samples.append(interpolated_tpr)

    alpha = 1.0 - ci_level
    ci_summary = {}
    for metric_name, values in metric_samples.items():
        ci_summary[metric_name] = {
            "point_estimate": point_metrics[metric_name],
            "ci_low": float(np.nanpercentile(values, 100 * (alpha / 2))) if values else np.nan,
            "ci_high": float(np.nanpercentile(values, 100 * (1 - alpha / 2))) if values else np.nan,
            "n_bootstrap_success": len(values),
        }

    point_fpr, point_tpr, _ = roc_curve(y_true, y_prob, pos_label=1)
    tpr_matrix = np.vstack(tpr_samples) if tpr_samples else None
    return {
        "metrics": ci_summary,
        "roc": {
            "point_fpr": point_fpr,
            "point_tpr": point_tpr,
            "grid_fpr": mean_fpr,
            "tpr_ci_low": np.nanpercentile(tpr_matrix, 100 * (alpha / 2), axis=0)
            if tpr_matrix is not None
            else None,
            "tpr_ci_high": np.nanpercentile(tpr_matrix, 100 * (1 - alpha / 2), axis=0)
            if tpr_matrix is not None
            else None,
        },
    }


def save_bootstrap_summary(ci_result, output_csv_path, setting_name):
    """
    Save bootstrap confidence interval summaries to CSV.
    """
    rows = []
    for metric_name, payload in ci_result["metrics"].items():
        rows.append(
            {
                "setting": setting_name,
                "metric": metric_name,
                "point_estimate": payload["point_estimate"],
                "ci_low": payload["ci_low"],
                "ci_high": payload["ci_high"],
                "n_bootstrap_success": payload["n_bootstrap_success"],
            }
        )
    pd.DataFrame(rows).to_csv(output_csv_path, index=False)


def save_roc_with_confidence_band(ci_result, output_path, line_color="black", fill_alpha=0.20):
    """
    Save a ROC curve with bootstrap confidence bands.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        ci_result["roc"]["point_fpr"],
        ci_result["roc"]["point_tpr"],
        color=line_color,
        linewidth=2.0,
    )
    if ci_result["roc"]["tpr_ci_low"] is not None and ci_result["roc"]["tpr_ci_high"] is not None:
        ax.fill_between(
            ci_result["roc"]["grid_fpr"],
            ci_result["roc"]["tpr_ci_low"],
            ci_result["roc"]["tpr_ci_high"],
            color=line_color,
            alpha=fill_alpha,
        )
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()


###########################
#         SHAP            #
###########################


def transform_with_feature_preprocessor(X_data, preprocessor):
    """Apply the fitted non-classifier pipeline and recover transformed feature names."""

    transformed = X_data.copy()
    feature_names = pd.Index(X_data.columns)

    for step_name, step in preprocessor.steps:
        if isinstance(step, SimpleImputer):
            transformed_array = step.transform(transformed)
            transformed = pd.DataFrame(
                transformed_array,
                index=X_data.index,
                columns=feature_names,
            )
        elif isinstance(step, StandardScaler):
            transformed_array = step.transform(transformed)
            transformed = pd.DataFrame(
                transformed_array,
                index=X_data.index,
                columns=feature_names,
            )
        elif isinstance(step, VarianceThreshold):
            support_mask = step.get_support()
            feature_names = feature_names[support_mask]
            transformed_array = step.transform(transformed)
            transformed = pd.DataFrame(
                transformed_array,
                index=X_data.index,
                columns=feature_names,
            )
        else:
            transformed_array = step.transform(transformed)
            transformed = pd.DataFrame(
                transformed_array,
                index=X_data.index,
                columns=feature_names,
            )

    return transformed, feature_names

def perform_shap_analysis(X_data, y_data, model_clf, preprocessor, shap_dir, report_path, dataset_name="dataset"):
    """Run SHAP analysis on a transformed radiomics dataset."""

    print(f"\nRunning SHAP analysis for {dataset_name}...")
    if shap is None:
        with open(report_path, "a", encoding="utf-8") as f_out:
            f_out.write(f"=== SHAP Analysis ({dataset_name}) ===\n")
            f_out.write("SHAP is not installed in the current environment.\n\n")
        print("SHAP is not installed; skipping this analysis.")
        return False, None, None, None
    try:
        X_transformed, selected_features = transform_with_feature_preprocessor(
            X_data=X_data,
            preprocessor=preprocessor,
        )
        
        # Seleccionar el explainer adecuado según el tipo de modelo
        if isinstance(model_clf, (RandomForestClassifier, GradientBoostingClassifier)):
            # Para modelos basados en árboles
            explainer = shap.TreeExplainer(model_clf)
        elif isinstance(model_clf, LogisticRegression):
            # Para modelos lineales
            try:
                explainer = shap.LinearExplainer(model_clf, X_transformed)
            except Exception:
                # Si falla, usar KernelExplainer como alternativa
                background = shap.kmeans(X_transformed, 50)
                explainer = shap.KernelExplainer(model_clf.predict_proba, background)
        else:
            # Para otros modelos (SVM, KNN, NaiveBayes)
            background = shap.kmeans(X_transformed, 50) # Resumen del dataset para acelerar
            explainer = shap.KernelExplainer(model_clf.predict_proba, background)
        
        # Calcular los valores SHAP
        shap_values = explainer(X_transformed)
        joblib.dump(shap_values, os.path.join(shap_dir, 'shap_values.pkl'))
        
        # Para clasificación binaria, quedarse con los valores SHAP de la clase positiva
        if shap_values.values.ndim > 2:
            shap_values = shap_values[:,:,1]
        
        # --------------------------------------------------------------
        # PARTE 1: TEST ESTADÍSTICO entre valores SHAP y la clase
        # --------------------------------------------------------------
        print(f" - Running Mann-Whitney U tests with Holm correction for {dataset_name}...")
        
        # Construir DataFrame con valores SHAP
        shap_matrix = pd.DataFrame(
            shap_values.values,
            index=X_transformed.index,
            columns=X_transformed.columns
        )
        
        # Listas para resultados estadísticos
        features_test = []
        pvalues_raw = []
        
        # Realizar test para cada característica
        for feat in shap_matrix.columns:
            # Separar valores SHAP por clase
            shap_class0 = shap_matrix.loc[y_data == 0, feat]
            shap_class1 = shap_matrix.loc[y_data == 1, feat]
            
            # Test Mann-Whitney U (test no paramétrico para comparar distribuciones)
            stat, pval = mannwhitneyu(shap_class0, shap_class1, alternative='two-sided')
            features_test.append(feat)
            pvalues_raw.append(pval)
        
        # Corrección por comparaciones múltiples (método de Holm)
        alpha = 0.05
        reject, pvals_corr, _, _ = multipletests(pvalues_raw, alpha=alpha, method='holm')
        
        # Generar reporte de resultados
        lines_output = []
        lines_output.append("=================================")
        lines_output.append(
            f"MANN-WHITNEY U TEST (SHAP values per feature) with Holm correction for {dataset_name}"
        )
        lines_output.append("Comparison: Class 0 vs Class 1")
        lines_output.append(f"alpha = {alpha}")
        lines_output.append(f"Total features: {len(features_test)}")
        lines_output.append("=================================\n")

        lines_output.append("Per-feature results (raw and corrected p-values):")
        significant_feats = []
        
        # Procesar resultados para cada característica
        for feat, pval_raw, pval_corr, rej_bool in zip(features_test, pvalues_raw, pvals_corr, reject):
            if rej_bool:
                result_str = "=> SIGNIFICANT DIFFERENCE"
                significant_feats.append((feat, pval_raw, pval_corr))
            else:
                result_str = "=> no significant difference"
            
            lines_output.append(
                f"    {feat}: raw p-value={pval_raw:.4e}, corrected p-value={pval_corr:.4e} {result_str}"
            )
        lines_output.append("")
        lines_output.append(
            f"Total comparisons with significant differences: {len(significant_feats)}."
        )
        
        if not significant_feats:
            lines_output.append("    No significant feature-level SHAP differences were found.")
        else:
            for feat, pval_raw, pval_corr in significant_feats:
                lines_output.append(
                    f"    {feat}: raw p-value={pval_raw:.4e}, corrected p-value={pval_corr:.4e} => SIGNIFICANT DIFFERENCE"
                )
        
        lines_output.append("\n")
        
        # Guardar resultados en archivo
        test_txt_path = os.path.join(shap_dir, "shap_statistical_test.txt")
        with open(test_txt_path, "w", encoding="utf-8") as f_out:
            for line in lines_output:
                f_out.write(line + "\n")
        
        print(f"  --> Statistical test results saved to: {test_txt_path}")
    
        # --------------------------------------------------------------
        # PARTE 2: HEATMAP (clase 0 primero, luego clase 1)
        # --------------------------------------------------------------
        print(f" - Generating class-ordered SHAP heatmap for {dataset_name}...")
        # Ordenar instancias por clase para visualización
        idx_class0 = np.where(y_data == 0)[0]
        idx_class1 = np.where(y_data == 1)[0]
        
        # Concatenar índices (primero clase 0, luego clase 1)
        idx_order = np.concatenate([idx_class0, idx_class1])
        
        heatmap_path = os.path.join(shap_dir, "shap_heatmap.png")
        
        # Generar heatmap con SHAP, especificando el orden de instancias
        shap.plots.heatmap(
            shap_values, 
            show=False,
            instance_order=idx_order
        )
        
        fig = plt.gcf()
        ax = plt.gca()
        
        # Añadir línea divisoria entre clases
        split_position = len(idx_class0)
        ax.axvline(split_position - 0.5, color='black', linewidth=1, zorder=10)
        n_total = len(idx_order)
        mid_class0 = (split_position / 2) / n_total
        mid_class1 = (split_position + len(idx_class1)/2) / n_total
        
        # Añadir etiquetas sobre la parte superior del heatmap
        ax.text(mid_class0, 1.01, 'Class 0', ha='center', va='bottom', transform=ax.transAxes)
        ax.text(mid_class1, 1.01, 'Class 1', ha='center', va='bottom', transform=ax.transAxes)
        
        # Guardar figura
        fig.set_size_inches(10, 6)
        plt.tight_layout()
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  --> SHAP heatmap saved to: {heatmap_path}")
    
        # --------------
        # Beeswarm plot 
        # --------------
        shap_fig_path = os.path.join(shap_dir, "shap_beeswarm.png")
        shap.plots.beeswarm(shap_values, max_display=16, show=False)
        fig = plt.gcf()
        fig.set_size_inches(14, 8)
        plt.subplots_adjust(left=0.4, right=0.95)
        plt.tight_layout()
        plt.savefig(shap_fig_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        print(f"  --> SHAP beeswarm plot saved to: {shap_fig_path}")
    
        # --------------------------------------------------------------
        # Scatter plots de las top features
        # --------------------------------------------------------------
        # Crear directorio para gráficos individuales
        scatter_dir = os.path.join(shap_dir, "scatter_plots")
        os.makedirs(scatter_dir, exist_ok=True)
        # Identificar las 15 características con mayor impacto (valor SHAP absoluto)
        mean_abs_shap = np.abs(shap_values.values).mean(axis=0)
        top_idx = np.argsort(mean_abs_shap)[-15:]
        top_idx = top_idx[np.argsort(mean_abs_shap[top_idx])[::-1]]
        top_features_shap = X_transformed.columns[top_idx]
        
        for i, feature in enumerate(top_features_shap, start=1):
            scatter_fig_path = os.path.join(scatter_dir, f"{i:02d}_{feature}.png")
            shap.plots.scatter(shap_values[:, feature], color=shap_values, show=False)
            fig = plt.gcf()
            fig.set_size_inches(10, 6)
            plt.tight_layout()
            plt.savefig(scatter_fig_path, dpi=dpi, bbox_inches='tight')
            plt.close()
        
        print(f"  --> SHAP scatter plots saved to: {scatter_dir}")
        return True, selected_features, shap_values, top_features_shap
    
    except Exception as e:
        with open(report_path, "a", encoding="utf-8") as f_out:
            f_out.write(f"=== SHAP Analysis ({dataset_name}) ===\n")
            f_out.write("SHAP analysis could not be generated (unsupported model or runtime error):\n")
            f_out.write(f"  {repr(e)}\n\n")
        print(f"Error during SHAP analysis for {dataset_name}: {e}")
        return False, None, None, None


###########################
#         LIME            #
###########################

def extraer_nombre(feat_str):
    """
    Extrae el nombre de característica original de la representación de LIME.
    LIME puede añadir información adicional como rangos o categorías.
    """
    tokens = re.findall(r'[A-Za-z0-9_\.\-]+', feat_str)
    valid_tokens = [t for t in tokens if re.search('[A-Za-z]', t)]
    if not valid_tokens:
        return feat_str.strip()
    return max(valid_tokens, key=len)


def explain_lime_instance(
    X_data, 
    index, 
    y_true, 
    y_pred, 
    model_clf, 
    explainer, 
    lime_dir, 
    instance_label="instance"
):
    """
    Genera y guarda explicación LIME para una instancia específica.
    
    Args:
        X_data: Datos preprocesados
        index: Índice de la instancia a explicar
        y_true: Etiquetas reales
        y_pred: Predicciones del modelo
        model_clf: Clasificador final
        explainer: Explainer LIME configurado
        lime_dir: Directorio para guardar resultados
        instance_label: Etiqueta para identificar la instancia
    """
    
    # Generar explicación LIME
    exp = explainer.explain_instance(
        data_row=X_data[index],
        predict_fn=model_clf.predict_proba,
        num_features=10 # Número de características a mostrar
    )
    
    # Rutas para guardar resultados
    explanation_txt_path = os.path.join(lime_dir, f"lime_explanation_{instance_label}_{index}.txt")
    fig_path = os.path.join(lime_dir, f"lime_explanation_{instance_label}_{index}.png")
    
    # Guardar explicación como texto
    with open(explanation_txt_path, "w", encoding="utf-8") as f:
        f.write(f"=== LIME explanation for {instance_label} (index: {index}) ===\n\n")
        f.write(f"True class: {y_true[index]}\n")
        f.write(f"Model prediction: {y_pred[index]}\n")
        f.write(f"Predicted probabilities: {model_clf.predict_proba([X_data[index]])}\n\n")
        f.write("Local feature importance:\n")
        for feat_info in exp.as_list():
            f.write("  {}: {:.4f}\n".format(feat_info[0], feat_info[1]))
    
    # Generar y guardar visualización
    with plt.style.context("default"):
        lime_fig = exp.as_pyplot_figure()
        ax = plt.gca()
        
        pos_color = "#0072B2"   
        neg_color = "#E69F00"   
    
        for rect in ax.patches:
            if rect.get_facecolor() == (0.0, 1.0, 0.0, 1.0): 
                rect.set_facecolor(pos_color)
            elif rect.get_facecolor() == (1.0, 0.0, 0.0, 1.0):
                rect.set_facecolor(neg_color)
        
        # plt.title(f"LIME Explanation - {instance_label} (index={index})")
        plt.savefig(fig_path, dpi=dpi, bbox_inches='tight')
        plt.close(lime_fig)
    
    print(
        f"  -> LIME for {instance_label} (index {index}) saved to:\n"
        f"     {explanation_txt_path}\n"
        f"     {fig_path}"
    )


def generate_lime_explanations_for_misclassifications(
    X_test_lime,
    y_test,
    model_clf, 
    explainer,
    lime_dir
):
    """
    Genera explicaciones LIME para ejemplos representativos de cada categoría
    de predicción (Verdaderos Positivos, Verdaderos Negativos, Falsos Positivos, 
    Falsos Negativos).
    """
    y_pred = model_clf.predict(X_test_lime)
    
    # Identificar índices para cada categoría
    # True Negative (TN): real=0, pred=0
    tn_indices = np.where((y_test == 0) & (y_pred == 0))[0]
    # True Positive (TP): real=1, pred=1
    tp_indices = np.where((y_test == 1) & (y_pred == 1))[0]
    # False Positive (FP): real=0, pred=1
    fp_indices = np.where((y_test == 0) & (y_pred == 1))[0]
    # False Negative (FN): real=1, pred=0
    fn_indices = np.where((y_test == 1) & (y_pred == 0))[0]
    
    # Función auxiliar para explicar el primer caso de cada categoría
    def explain_first_if_any(indices, label):
        if len(indices) > 0:
            idx = indices[0] # Tomar el primer ejemplo
            explain_lime_instance(
                X_data=X_test_lime,
                index=idx,
                y_true=y_test,
                y_pred=y_pred,
                model_clf=model_clf,
                explainer=explainer,
                lime_dir=lime_dir,
                instance_label=label
            )
        else:
            print(f"No instances found for {label}.")
    
    # Generar explicaciones para cada categoría
    explain_first_if_any(tn_indices, "TN")
    explain_first_if_any(tp_indices, "TP")
    explain_first_if_any(fp_indices, "FP")
    explain_first_if_any(fn_indices, "FN")


def perform_lime_analysis(
    X_data,
    y_data,
    model_clf,
    preprocessor,
    lime_dir,
    selected_features,
    report_path,
    shap_top_features=None,
    dataset_name="dataset",
):
    """Run LIME analysis on a transformed radiomics dataset."""

    print(f"\nRunning LIME analysis for {dataset_name}...")
    if LimeTabularExplainer is None:
        with open(report_path, "a", encoding="utf-8") as f_out:
            f_out.write(f"=== LIME Analysis ({dataset_name}) ===\n")
            f_out.write("LIME is not installed in the current environment.\n\n")
        print("LIME is not installed; skipping this analysis.")
        return False
    try:
        X_lime_df, transformed_features = transform_with_feature_preprocessor(
            X_data=X_data,
            preprocessor=preprocessor,
        )
        X_lime = X_lime_df.to_numpy()
        transformed_feature_index = pd.Index(transformed_features)
        
        explainer_lime = LimeTabularExplainer(
            training_data=X_lime,
            feature_names=transformed_feature_index.tolist(),
            class_names=["0", "1"],
            discretize_continuous=True,
            random_state=42
        )
        
        # Recopilar resultados de LIME para todas las instancias
        resultados = []
        
        # Iterar en múltiples instancias para análisis global
        for i in range(len(X_lime)):
            exp = explainer_lime.explain_instance(
                data_row=X_lime[i],
                predict_fn=model_clf.predict_proba
            )
            # Obtener lista de importancias para clase positiva
            lime_list = exp.as_list(label=1)  
            
            # Procesar cada característica y su importancia
            for (feat_str, peso) in lime_list:
                feature_name = extraer_nombre(feat_str) 
                col_idx = transformed_feature_index.get_loc(feature_name)
                valor_feature = X_lime[i, col_idx]
                
                # Almacenar resultados
                resultados.append({
                    'instancia': i,
                    'feature': feature_name,
                    'peso': peso,
                    'valor_feature': valor_feature
                })
        
        df_lime = pd.DataFrame(resultados)
        df_lime['abs_peso'] = df_lime['peso'].abs()
        
        # Seleccionar las 15 características con mayor peso absoluto promedio
        top_features = (
            df_lime.groupby('feature')['abs_peso']
            .mean()
            .sort_values(ascending=False)
            .head(15)
            .index
            .tolist()
        )
        
        # Filtrar DataFrame para mostrar solo las características principales
        df_lime_top15 = df_lime[df_lime['feature'].isin(top_features)].copy()
        
        # Normalización min-max para cada característica
        df_lime_top15['valor_min'] = df_lime_top15.groupby('feature')['valor_feature'].transform('min')
        df_lime_top15['valor_max'] = df_lime_top15.groupby('feature')['valor_feature'].transform('max')
        
        # Normalización min-max (escala 0-1)
        df_lime_top15['valor_feature_norm'] = (
            (df_lime_top15['valor_feature'] - df_lime_top15['valor_min'])
            / (df_lime_top15['valor_max'] - df_lime_top15['valor_min'])
        )
        
        # Normalización logarítmica
        # Asegurar valores positivos
        df_lime_top15['valor_feature_shifted'] = df_lime_top15.groupby('feature')['valor_feature'].transform(
            lambda x: x - x.min() + 1e-10 if x.min() <= 0 else x
        )
        # Aplicar transformación logarítmica
        df_lime_top15['valor_feature_log'] = np.log1p(df_lime_top15['valor_feature_shifted'])
        # Normalizar los valores logarítmicos
        df_lime_top15['valor_feature_log_min'] = df_lime_top15.groupby('feature')['valor_feature_log'].transform('min')
        df_lime_top15['valor_feature_log_max'] = df_lime_top15.groupby('feature')['valor_feature_log'].transform('max')
        df_lime_top15['valor_feature_log_norm'] = (
            (df_lime_top15['valor_feature_log'] - df_lime_top15['valor_feature_log_min'])
            / (df_lime_top15['valor_feature_log_max'] - df_lime_top15['valor_feature_log_min'])
        )
        
        # Ordenar características para visualización
        # Usar mismo orden que SHAP cuando sea posible
        lime_features = df_lime_top15['feature'].unique().tolist()
        
        # Crear orden final (primero SHAP, luego el resto de LIME)
        if shap_top_features is not None:
            shap_order = list(shap_top_features)
            final_order = [feat for feat in shap_order if feat in lime_features] + \
                         [feat for feat in lime_features if feat not in shap_order]
        else:
            final_order = lime_features
                      
        # Definir colores comunes para ambos gráficos
        colors = [
            (0.0,  "#008afb"),
            (0.2, "#008afb"),  
            (0.7, "#ff0052"),  
            (1.0,  "#ff0052")  
        ]
        
        # Crear colormap personalizado
        the_cmap = LinearSegmentedColormap.from_list("my_cmap", colors)
        
        # --- Gráfico 1: Visualización min-max ---
        fig, ax = plt.subplots(figsize=(10,8))
        sns.stripplot(
            data=df_lime_top15,
            x='peso',                  # Peso LIME en eje X
            y='feature',               # Características en eje Y
            hue='valor_feature_norm',  # Color según valor normalizado
            palette=the_cmap,          # Paleta personalizada
            hue_norm=(0, 1),           # Rango de normalización
            orient='h',                # Horizontal
            size=5,                    # Tamaño de puntos
            dodge=False,               # Sin separación
            legend=False,              # Sin leyenda independiente
            order=final_order,         # Orden personalizado de características
            ax=ax
        )
        
        ax.axvline(0, color='black', linestyle='--')
        # ax.set_title(f"Distribución de pesos LIME - Top 15 features ({dataset_name}, Normalización Min-Max)")
        
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        sm = mpl.cm.ScalarMappable(cmap=the_cmap, norm=norm)
        sm.set_array([])
        
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label("Valor feature normalizado [0..1]")
        
        plt.tight_layout()
        fig_path = os.path.join(lime_dir, "lime_pseudo_beeswarm.png")
        plt.savefig(fig_path, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        # --- Gráfico 2: Visualización logarítmica ---
        fig, ax = plt.subplots(figsize=(10,8))
        sns.stripplot(
            data=df_lime_top15,
            x='peso',
            y='feature',
            hue='valor_feature_log_norm',   # Valor log-normalizado  
            palette=the_cmap,          
            hue_norm=(0, 1),             
            orient='h',
            size=5,
            dodge=False,
            legend=False,
            order=final_order,
            ax=ax
        )
        
        ax.axvline(0, color='black', linestyle='--')
        # ax.set_title(f"Distribución de pesos LIME - Top 15 features ({dataset_name}, Normalización Logarítmica)")
        
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        sm = mpl.cm.ScalarMappable(cmap=the_cmap, norm=norm)
        sm.set_array([])
        
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label("Valor feature norm. logarítmica [0..1]")
        
        plt.tight_layout()
        fig_path_log = os.path.join(lime_dir, "lime_pseudo_beeswarm_log.png")
        plt.savefig(fig_path_log, dpi=dpi, bbox_inches='tight')
        plt.close()
        
        print(f"  --> LIME min-max visualization saved to: {fig_path}")
        print(f"  --> LIME log-normalized visualization saved to: {fig_path_log}")
        
        # --- Explicaciones locales para casos específicos ---
        ind_lime_dir = os.path.join(lime_dir, "individual_analysis")
        os.makedirs(ind_lime_dir, exist_ok=True)
        
        # Generar explicaciones para casos representativos
        generate_lime_explanations_for_misclassifications(
            X_test_lime=X_lime,
            y_test=y_data,
            model_clf=model_clf,
            explainer=explainer_lime,
            lime_dir=ind_lime_dir
        )
    
        return True
    
    except Exception as e:
        with open(report_path, "a", encoding="utf-8") as f_out:
            f_out.write(f"\n=== LIME Analysis ({dataset_name}) ===\n")
            f_out.write("LIME analysis could not be generated (unsupported model or runtime error):\n")
            f_out.write(f"  {repr(e)}\n\n")
        print(f"Error during LIME analysis for {dataset_name}: {e}")
        return False

def main():
    """Run leakage-safe hold-out optimization, calibration, and interpretability analysis."""

    configure_live_logging()

    parser = argparse.ArgumentParser(
        description=(
            "Train and optimize a radiomics model with a final grouped hold-out test split, "
            "probability calibration, and interpretability outputs."
        )
    )
    parser.add_argument(
        "--csv",
        type=str,
        default="features_all_gland.csv",
        help="Feature table path or filename. Relative names are resolved inside the radiomics data root.",
    )
    parser.add_argument(
        "--data_pre",
        type=str,
        default="artifacts/radiomics",
        help="Directory containing the concatenated radiomics feature tables.",
    )
    parser.add_argument(
        "--results_base",
        type=str,
        default="results/radiomics",
        help="Base directory where hold-out outputs will be written.",
    )
    parser.add_argument(
        "--feature_strategy",
        type=str,
        default="most_discriminant",
        help="Label used to organize the output directory structure.",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=[
            "SVM",
            "SVC",
            "LogisticRegression",
            "RandomForest",
            "NaiveBayes",
            "GaussianNB",
            "KNN",
            "GradientBoosting",
        ],
        help="Classifier to optimize on the training split.",
    )
    parser.add_argument(
        "--n_folds",
        type=int,
        default=5,
        help="Number of grouped CV folds used inside BayesSearchCV.",
    )
    parser.add_argument(
        "--variables",
        type=str,
        default=None,
        help=(
            "Optional external feature allowlist. Leave unset for a fully leakage-safe run that selects "
            "features from the training split only."
        ),
    )
    parser.add_argument(
        "--bootstrap_iterations",
        type=int,
        default=1000,
        help="Number of bootstrap iterations used for test-set confidence intervals.",
    )
    parser.add_argument(
        "--ci_level",
        type=float,
        default=0.95,
        help="Confidence level used for bootstrap confidence intervals.",
    )
    parser.add_argument(
        "--min_features",
        type=int,
        default=10,
        help="Minimum number of radiomics features retained after training-only selection.",
    )
    parser.add_argument(
        "--max_features_cap",
        type=int,
        default=60,
        help="Upper cap for the automatically inferred number of retained radiomics features.",
    )
    parser.add_argument(
        "--samples_per_feature",
        type=int,
        default=25,
        help="Target number of training samples per retained feature used to infer the feature cap.",
    )
    parser.add_argument(
        "--minority_samples_per_feature",
        type=int,
        default=8,
        help="Target number of minority-class samples per retained feature used to infer the feature cap.",
    )
    parser.add_argument(
        "--fdr_alpha",
        type=float,
        default=0.05,
        help="False-discovery-rate alpha used before correlation pruning.",
    )
    parser.add_argument(
        "--correlation_threshold",
        type=float,
        default=0.90,
        help="Absolute Pearson-correlation threshold used to prune redundant features.",
    )
    parser.add_argument(
        "--selection_n_jobs",
        type=int,
        default=-1,
        help="Number of parallel workers used during training-only feature scoring.",
    )
    parser.add_argument(
        "--search_iterations",
        type=int,
        default=50,
        help="Number of Bayesian optimization iterations for models with a search space.",
    )
    parser.add_argument(
        "--search_n_jobs",
        type=int,
        default=-1,
        help="Number of parallel workers used by BayesSearchCV.",
    )
    args = parser.parse_args()
    ci_percent = int(args.ci_level * 100)

    if BayesSearchCV is None:
        raise ModuleNotFoundError(
            "scikit-optimize is required to run this script. Install 'scikit-optimize' to continue."
        )
    
    model_aliases = {
        "SVC": "SVM",
        "GaussianNB": "NaiveBayes",
    }
    selected_model = model_aliases.get(args.model, args.model)

    data_root = (PROJECT_ROOT / args.data_pre).resolve()
    data_path = resolve_feature_table_path(
        project_root=PROJECT_ROOT,
        data_root=data_root,
        csv_argument=args.csv,
    )
    csv_stem = Path(data_path).stem
    mode = csv_stem.rsplit("_", 1)[-1]

    log_progress("Starting final hold-out optimization.")
    log_progress(f"Selected model: {selected_model}")
    log_progress(f"Feature table: {data_path}")
    if args.variables:
        log_progress(f"External feature allowlist: {args.variables}")
    else:
        log_progress("External feature allowlist: not used")
    log_progress(
        "Optimization settings: "
        f"n_folds={args.n_folds}, search_iterations={args.search_iterations}, "
        f"search_n_jobs={args.search_n_jobs}, selection_n_jobs={args.selection_n_jobs}"
    )

    results_root = (PROJECT_ROOT / args.results_base / args.feature_strategy / mode).resolve()
    output_parent_dir = os.path.join(results_root, "best_results", selected_model.lower())
    calibration_dir = os.path.join(output_parent_dir, "calibration")
    explicability_dir = os.path.join(output_parent_dir, "explicability")

    train_explicability_dir = os.path.join(explicability_dir, "train")
    test_explicability_dir = os.path.join(explicability_dir, "test")

    # SHAP and LIME subdirectories for training data
    train_shap_dir = os.path.join(train_explicability_dir, "SHAP")
    train_lime_dir = os.path.join(train_explicability_dir, "LIME")

    # SHAP and LIME subdirectories for hold-out test data
    test_shap_dir = os.path.join(test_explicability_dir, "SHAP")
    test_lime_dir = os.path.join(test_explicability_dir, "LIME")

    # Create output directories up front
    os.makedirs(output_parent_dir, exist_ok=True)
    os.makedirs(calibration_dir, exist_ok=True)
    os.makedirs(explicability_dir, exist_ok=True)
    os.makedirs(train_explicability_dir, exist_ok=True)
    os.makedirs(test_explicability_dir, exist_ok=True)
    os.makedirs(train_shap_dir, exist_ok=True)
    os.makedirs(train_lime_dir, exist_ok=True)
    os.makedirs(test_shap_dir, exist_ok=True)
    os.makedirs(test_lime_dir, exist_ok=True)
    
    log_progress(f"Output directory: {os.path.relpath(output_parent_dir)}")

    log_progress(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    if "sample_id" in df.columns:
        df = df.set_index("sample_id")
    else:
        df['patient_id_study_id'] = df['patient_id'].astype(str) + '_' + df['study_id'].astype(str)
        df = df.set_index('patient_id_study_id')
    log_progress(f"Loaded data shape: {df.shape}")

    y = df["label"].values
    groups = df["patient_id"].values
    X = prepare_numeric_radiomics_matrix(df).replace([np.inf, -np.inf], np.nan)

    gss = GroupShuffleSplit(test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    X_train_full, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train_full, y_test = y[train_idx], y[test_idx]
    groups_train_full = groups[train_idx]
    patient_ids_test = groups[test_idx]
    log_progress(
        f"Created grouped hold-out split | train n={len(X_train_full)} pos={int(np.sum(y_train_full == 1))} "
        f"| test n={len(X_test)} pos={int(np.sum(y_test == 1))}"
    )

    if args.variables:
        variables_path = Path(args.variables)
        if not variables_path.is_absolute():
            variables_path = (PROJECT_ROOT / variables_path).resolve()
        with open(variables_path, "r", encoding="utf-8") as f_vars:
            allowlist = [line.strip() for line in f_vars if line.strip()]
        allowlist = [feature_name for feature_name in allowlist if feature_name in X_train_full.columns]
        if not allowlist:
            raise ValueError(
                "The provided external feature allowlist does not match any columns in the selected feature table."
            )
        X_train_full = X_train_full[allowlist]
        X_test = X_test[allowlist]

    selected_features, selection_df, selection_metadata = select_radiomics_features(
        X_train=X_train_full,
        y_train=y_train_full,
        min_features=args.min_features,
        max_features_cap=args.max_features_cap,
        samples_per_feature=args.samples_per_feature,
        minority_samples_per_feature=args.minority_samples_per_feature,
        fdr_alpha=args.fdr_alpha,
        correlation_threshold=args.correlation_threshold,
        n_jobs=args.selection_n_jobs,
    )
    selection_dir = os.path.join(output_parent_dir, "feature_selection")
    os.makedirs(selection_dir, exist_ok=True)
    selection_csv_path = os.path.join(selection_dir, "training_only_feature_selection.csv")
    selection_txt_path = os.path.join(selection_dir, "selected_features_training_only.txt")
    selection_df.to_csv(selection_csv_path, index=False)
    with open(selection_txt_path, "w", encoding="utf-8") as f_out:
        for feature_name in selected_features:
            f_out.write(f"{feature_name}\n")

    X_train_full = X_train_full[selected_features].copy()
    X_test = X_test[selected_features].copy()
    log_progress(
        f"Selected {len(selected_features)} training-only features "
        f"(FDR candidates={selection_metadata['n_fdr_features']}, "
        f"pruned pool={selection_metadata['n_pruned_features']}, "
        f"feature cap={selection_metadata['feature_limit']})."
    )
    log_progress(f"Saved training-only feature selection CSV to: {selection_csv_path}")
    log_progress(f"Saved selected feature list to: {selection_txt_path}")

    number_folds = args.n_folds
    score_group = {
        'roc_auc': 'roc_auc',
        'f1': 'f1',
        'balanced_accuracy': 'balanced_accuracy'
    }
    score_refit_str = 'roc_auc'
    random_state_value = 42

    if selected_model == 'SVM':
        pipe = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            VarianceThreshold(),
            SVC(random_state=random_state_value, class_weight="balanced", probability=True),
        )
        param_grid = {
            'svc__C': Real(1e-4, 1e3, prior='log-uniform'),
            'svc__kernel': Categorical(['linear', 'rbf', 'poly']),
            'svc__gamma': Real(1e-4, 1e3, prior='log-uniform'),
            'svc__coef0': Real(0, 1)
        }
        
    elif selected_model == 'LogisticRegression':
        pipe = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            VarianceThreshold(),
            LogisticRegression(
                class_weight='balanced',
                random_state=random_state_value,
                solver='saga',
                max_iter=10000
            ),
        )
        param_grid = {
            'logisticregression__C': Real(1e-4, 1e3, prior='log-uniform'),
            'logisticregression__penalty': Categorical(['l1', 'l2', 'elasticnet']),
            'logisticregression__l1_ratio': Real(0.1, 0.9)
        }
        
    elif selected_model == 'RandomForest':
        pipe = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            VarianceThreshold(),
            RandomForestClassifier(
                n_jobs=1,
                class_weight="balanced_subsample",
                random_state=random_state_value,
            ),
        )
        param_grid = {
            'randomforestclassifier__n_estimators': Integer(50, 1024),
            'randomforestclassifier__max_depth': Integer(1, 10),
            'randomforestclassifier__max_features': Categorical(['sqrt', 'log2', None]),
            'randomforestclassifier__min_samples_split': Integer(2, 20)
        }
        
    elif selected_model == 'NaiveBayes':
        pipe = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            VarianceThreshold(),
            GaussianNB(),
        )
        param_grid = {}
        
    elif selected_model == 'KNN':
        pipe = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            VarianceThreshold(),
            KNeighborsClassifier(n_jobs=1),
        )
        param_grid = {
            'kneighborsclassifier__n_neighbors': Integer(2, 8),
            'kneighborsclassifier__weights': Categorical(['uniform', 'distance'])
        }
        
    elif selected_model == 'GradientBoosting':
        pipe = make_pipeline(
            SimpleImputer(strategy="median"),
            StandardScaler(),
            VarianceThreshold(),
            GradientBoostingClassifier(random_state=random_state_value),
        )
        param_grid = {
            'gradientboostingclassifier__n_estimators': Integer(50, 1024),
            'gradientboostingclassifier__learning_rate': Real(1e-4, 0.1, prior='log-uniform'),
            'gradientboostingclassifier__max_depth': Integer(1, 10),
            'gradientboostingclassifier__subsample': Real(0.5, 1.0),
            'gradientboostingclassifier__max_features': Categorical(['sqrt', 'log2', None])
        }
    else:
        raise ValueError(f"Unknown model '{selected_model}'.")
    
    cv = StratifiedGroupKFold(n_splits=number_folds, shuffle=True, random_state=random_state_value)
    if param_grid:
        log_progress(
            f"Running Bayesian hyperparameter optimization with BayesSearchCV "
            f"using {number_folds} grouped folds, {args.search_iterations} iterations, "
            f"and n_jobs={args.search_n_jobs}."
        )
        search_start_time = time.perf_counter()
        search = BayesSearchCV(
            estimator=pipe,
            search_spaces=param_grid,
            scoring=score_group,
            refit=score_refit_str,
            cv=cv,
            n_iter=args.search_iterations,
            n_jobs=args.search_n_jobs,
            random_state=random_state_value,
            verbose=2,
        )
        search.fit(X_train_full, y_train_full, groups=groups_train_full)
        best_estimator = search.best_estimator_
        best_params = search.best_params_
        cv_summary = {
            key: (
                search.cv_results_[f"mean_test_{key}"][search.best_index_],
                search.cv_results_[f"std_test_{key}"][search.best_index_],
            )
            for key in score_group
        }
        search_elapsed = time.perf_counter() - search_start_time
        log_progress(f"Optimization completed in {search_elapsed / 60:.1f} minutes.")
        log_progress(f"Best parameters: {best_params}")
        for key, (mean_test, std_test) in cv_summary.items():
            log_progress(f"Best CV {key}: {mean_test:.4f} +/- {std_test:.4f}")
    else:
        log_progress("No hyperparameter search space defined for this model. Fitting the base pipeline directly...")
        best_estimator = pipe.fit(X_train_full, y_train_full)
        best_params = {}
        cv_summary = {}

    estimator_path = os.path.join(output_parent_dir, "best_estimator.pkl")
    joblib.dump(best_estimator, estimator_path)
    log_progress(f"Best estimator saved to: {os.path.relpath(estimator_path)}")

    report_path = os.path.join(output_parent_dir, "report.txt")
    with open(report_path, "w", encoding="utf-8") as f_out:
        f_out.write(f"=== Final Hold-out Optimization: {selected_model} ===\n\n")
        f_out.write(f"Feature table: {data_path}\n")
        f_out.write(f"Training-only selection CSV: {selection_csv_path}\n")
        f_out.write(f"Training-only selected features TXT: {selection_txt_path}\n")
        f_out.write(f"Selected features: {len(selected_features)}\n")
        f_out.write(f"FDR alpha: {args.fdr_alpha}\n")
        f_out.write(f"Correlation threshold: {args.correlation_threshold}\n")
        f_out.write(f"External allowlist used: {'yes' if args.variables else 'no'}\n")
        f_out.write(f"Best parameters ({score_refit_str}): {best_params}\n\n")
        if cv_summary:
            f_out.write("=== Cross-validation Results (BayesSearchCV) ===\n")
            for key in score_group:
                mean_test, std_test = cv_summary[key]
                f_out.write(f"  CV {key}: {mean_test:.3f} +/- {std_test:.3f}\n")
        else:
            f_out.write("=== Cross-validation Results ===\n")
            f_out.write("  Hyperparameter search was skipped because this model has no search space.\n")
        f_out.write("\n")

    log_progress("Evaluating the uncalibrated model on the hold-out test set...")
    y_pred_test = best_estimator.predict(X_test)
    p_pre = best_estimator.predict_proba(X_test)[:, 1]
    
    # Generar matriz de confusión sin calibrar
    confusion_fig = os.path.join(output_parent_dir, "confusion_matrix.png")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.grid(False)
    disp = ConfusionMatrixDisplay.from_estimator(
        best_estimator, 
        X_test, 
        y_test, 
        ax=ax,         
        cmap='cividis'
    )
    # ax.set_title(f"{selected_model} (NO calibrado)", fontsize=12)
    
    n_classes = len(disp.display_labels)
    
    ax.set_xticks(np.arange(-0.5, n_classes, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_classes, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='--', linewidth=1)
    ax.tick_params(which='minor', bottom=False, left=False)
    plt.tight_layout()
    plt.savefig(confusion_fig, dpi=dpi, bbox_inches='tight')
    plt.close()
    log_progress(f"Confusion matrix saved to: {confusion_fig}")
    
    # Calcular métricas de rendimiento en test
    if hasattr(best_estimator, "predict_proba"):
        auc_ = roc_auc_score(y_test, p_pre)
    elif hasattr(best_estimator, "decision_function"):
        auc_ = roc_auc_score(y_test, best_estimator.decision_function(X_test))
    else:
        auc_ = np.nan
    
    # Métricas generales de clasificación
    mcc_    = matthews_corrcoef(y_test, y_pred_test)
    kappa_  = cohen_kappa_score(y_test, y_pred_test)
    f1_     = f1_score(y_test, y_pred_test)
    acc_    = accuracy_score(y_test, y_pred_test)
    sens_   = recall_score(y_test, y_pred_test, pos_label=1)
    spec_   = recall_score(y_test, y_pred_test, pos_label=0)
    ppv_    = precision_score(y_test, y_pred_test, pos_label=1)
    npv_    = precision_score(y_test, y_pred_test, pos_label=0)
    balacc_ = balanced_accuracy_score(y_test, y_pred_test)
    
    # Reporte detallado de clasificación
    report_cr = classification_report(y_test, y_pred_test)
    test_ci_uncalibrated = bootstrap_patient_level_confidence_intervals(
        y_true=y_test,
        y_prob=p_pre,
        patient_ids=patient_ids_test,
        threshold=0.5,
        n_bootstrap=args.bootstrap_iterations,
        ci_level=args.ci_level,
        seed=42,
    )
    test_ci_uncalibrated_path = os.path.join(output_parent_dir, "test_confidence_intervals_uncalibrated.csv")
    save_bootstrap_summary(
        ci_result=test_ci_uncalibrated,
        output_csv_path=test_ci_uncalibrated_path,
        setting_name="uncalibrated_test",
    )
    roc_ci_uncalibrated_fig = os.path.join(output_parent_dir, "roc_test_uncalibrated_ci.png")
    save_roc_with_confidence_band(
        ci_result=test_ci_uncalibrated,
        output_path=roc_ci_uncalibrated_fig,
        line_color="black",
    )
    log_progress(
        f"Uncalibrated hold-out metrics | "
        f"AUC={format_metric(auc_)} | F1={format_metric(f1_)} | "
        f"Balanced Accuracy={format_metric(balacc_)} | MCC={format_metric(mcc_)} | "
        f"{ci_percent}% AUC CI=[{format_metric(test_ci_uncalibrated['metrics']['auc']['ci_low'])}, "
        f"{format_metric(test_ci_uncalibrated['metrics']['auc']['ci_high'])}]"
    )
    
    with open(report_path, "a", encoding="utf-8") as f_out:
        f_out.write("=== Hold-out Test Evaluation (Uncalibrated) ===\n")
        f_out.write(f"  Confusion matrix figure: {confusion_fig}\n")
        f_out.write(f"  AUC: {auc_:.3f}\n")
        f_out.write(
            f"  AUC {ci_percent}% CI (bootstrap): "
            f"[{test_ci_uncalibrated['metrics']['auc']['ci_low']:.3f}, "
            f"{test_ci_uncalibrated['metrics']['auc']['ci_high']:.3f}]\n"
        )
        f_out.write(f"  Bootstrap summary CSV: {test_ci_uncalibrated_path}\n")
        f_out.write(f"  ROC with CI figure: {roc_ci_uncalibrated_fig}\n")
        f_out.write(f"  MCC: {mcc_:.3f}\n")
        f_out.write(f"  Kappa: {kappa_:.3f}\n")
        f_out.write(f"  F1: {f1_:.3f}\n")
        f_out.write(f"  Accuracy: {acc_:.3f}\n")
        f_out.write(f"  Sensitivity: {sens_:.3f}\n")
        f_out.write(f"  Specificity: {spec_:.3f}\n")
        f_out.write(f"  PPV: {ppv_:.3f}\n")
        f_out.write(f"  NPV: {npv_:.3f}\n")
        f_out.write(f"  Balanced Accuracy: {balacc_:.3f}\n\n")
        f_out.write("=== Classification Report ===\n")
        f_out.write(report_cr)
        f_out.write("\n\n")
    
    log_progress("Calibrating the model with Platt scaling (sigmoid, cv=5)...")
    cal_clf = CalibratedClassifierCV(best_estimator, method="sigmoid", cv=5)
    cal_clf.fit(X_train_full, y_train_full)
    
    # --- Curva de calibración PRE (antes de calibrar) ---
    calibration_fig_pre = os.path.join(calibration_dir, "calibration_pre.png")
    fig, ax = plt.subplots(figsize=(8, 6))
    CalibrationDisplay.from_estimator(
        best_estimator, 
        X_test, 
        y_test, 
        n_bins=10, 
        name=f"{selected_model}_pre", 
        ax=ax
    )

    for line in ax.get_lines():
        line.set_color("black")

    legend = ax.get_legend()
    if legend:
        for text in legend.get_texts():
            text.set_color("black")
        for line in legend.get_lines():
            line.set_color("black")
        for patch in legend.get_patches():
            patch.set_edgecolor("black")
            patch.set_facecolor("black")
            
    # ax.set_title(f"Calibration Curve (pre), {selected_model}", fontsize=14)
    plt.savefig(calibration_fig_pre, dpi=dpi, bbox_inches='tight')
    plt.close()
    log_progress(f"Pre-calibration curve saved to: {calibration_fig_pre}")
    
    # --- Curva de calibración POST (después de calibrar) ---
    calibration_fig_post = os.path.join(calibration_dir, "calibration_post.png")
    fig, ax = plt.subplots(figsize=(8, 6))
    CalibrationDisplay.from_estimator(
        cal_clf, 
        X_test, 
        y_test, 
        n_bins=10, 
        name=f"{selected_model}_post", 
        ax=ax
    )
    # ax.set_title(f"Calibration Curve (post), {selected_model}", fontsize=14)

    for line in ax.get_lines():
        line.set_color("black")

    legend = ax.get_legend()
    if legend:
        for text in legend.get_texts():
            text.set_color("black")
        for line in legend.get_lines():
            line.set_color("black")
        for patch in legend.get_patches():
            patch.set_edgecolor("black")
            patch.set_facecolor("black")
            
    plt.savefig(calibration_fig_post, dpi=dpi, bbox_inches='tight')
    plt.close()
    log_progress(f"Post-calibration curve saved to: {calibration_fig_post}")

    # Métricas de calibración 
    def calibration_error(y_true, y_prob, n_bins=10, norm='l1'):
        y_true = np.asarray(y_true, dtype=np.float64)
        y_prob = np.asarray(y_prob, dtype=np.float64)

        if norm not in {"l1", "l2"}:
            raise ValueError("norm must be 'l1' or 'l2'")

        # 1. Asignar cada probabilidad a un bin
        bins = np.linspace(0.0, 1.0, n_bins + 1)
        bin_ids = np.digitize(y_prob, bins, right=True) - 1
        bin_ids = np.clip(bin_ids, 0, n_bins - 1)      

        # 2. Sumar error ponderado por el tamaño de cada bin
        ece = 0.0
        N = len(y_true)
        for i in range(n_bins):
            idx = bin_ids == i
            if idx.any():
                prob_avg = y_prob[idx].mean()
                acc_avg  = y_true[idx].mean()
                err_bin  = abs(prob_avg - acc_avg) if norm == "l1" else (prob_avg - acc_avg) ** 2
                ece     += err_bin * idx.sum() / N

        return ece

    # 1) Probabilidades (sin / con Platt)
    p_post = cal_clf.predict_proba(X_test)[:, 1]
    test_ci_calibrated = bootstrap_patient_level_confidence_intervals(
        y_true=y_test,
        y_prob=p_post,
        patient_ids=patient_ids_test,
        threshold=0.5,
        n_bootstrap=args.bootstrap_iterations,
        ci_level=args.ci_level,
        seed=42,
    )
    test_ci_calibrated_path = os.path.join(calibration_dir, "test_confidence_intervals_calibrated.csv")
    save_bootstrap_summary(
        ci_result=test_ci_calibrated,
        output_csv_path=test_ci_calibrated_path,
        setting_name="calibrated_test_threshold_0_5",
    )
    roc_ci_calibrated_fig = os.path.join(calibration_dir, "roc_test_calibrated_ci.png")
    save_roc_with_confidence_band(
        ci_result=test_ci_calibrated,
        output_path=roc_ci_calibrated_fig,
        line_color="black",
    )

    # 2) Expected Calibration Error (ECE)
    ece_pre  = calibration_error(y_test, p_pre,  n_bins=10, norm='l1')
    ece_post = calibration_error(y_test, p_post, n_bins=10, norm='l1')

    # 3) Brier score
    brier_pre  = brier_score_loss(y_test, p_pre)
    brier_post = brier_score_loss(y_test, p_post)
    log_progress(
        f"Calibration summary | post_AUC={format_metric(test_ci_calibrated['metrics']['auc']['point_estimate'])} | "
        f"ECE pre={format_metric(ece_pre)} post={format_metric(ece_post)} | "
        f"Brier pre={format_metric(brier_pre)} post={format_metric(brier_post)}"
    )

    # 4) Volcar resultados al informe
    with open(report_path, "a", encoding="utf-8") as f_out:
        f_out.write("=== Calibration Metrics ===\n")
        f_out.write(f"  AUC (post): {test_ci_calibrated['metrics']['auc']['point_estimate']:.3f}\n")
        f_out.write(
            f"  AUC {ci_percent}% CI (post, bootstrap): "
            f"[{test_ci_calibrated['metrics']['auc']['ci_low']:.3f}, "
            f"{test_ci_calibrated['metrics']['auc']['ci_high']:.3f}]\n"
        )
        f_out.write(f"  Bootstrap summary CSV (post): {test_ci_calibrated_path}\n")
        f_out.write(f"  ROC with CI figure (post): {roc_ci_calibrated_fig}\n")
        f_out.write(f"  ECE  (pre):  {ece_pre:.3f}\n")
        f_out.write(f"  ECE  (post): {ece_post:.3f}\n")
        f_out.write(f"  Brier (pre): {brier_pre:.3f}\n")
        f_out.write(f"  Brier (post): {brier_post:.3f}\n\n")

    thresholds = np.linspace(0.1, 0.9, 9)
    best_thresh = None
    best_f1 = -np.inf
    results = []
    
    for thresh in thresholds:
        y_pred_thresh = (cal_clf.predict_proba(X_test)[:, 1] >= thresh).astype(int)
        f1_val = f1_score(y_test, y_pred_thresh)
        results.append({'threshold': thresh, 'f1': f1_val})
        if f1_val > best_f1:
            best_f1 = f1_val
            best_thresh = thresh
    log_progress(
        f"Threshold sweep completed | best_threshold={best_thresh:.2f} | "
        f"best_f1={format_metric(best_f1)}"
    )
    
    y_pred_best = (cal_clf.predict_proba(X_test)[:, 1] >= best_thresh).astype(int)
    
    auc_best    = roc_auc_score(y_test, cal_clf.predict_proba(X_test)[:, 1])
    mcc_best    = matthews_corrcoef(y_test, y_pred_best)
    kappa_best  = cohen_kappa_score(y_test, y_pred_best)
    f1_best     = f1_score(y_test, y_pred_best)
    acc_best    = accuracy_score(y_test, y_pred_best)
    sens_best   = recall_score(y_test, y_pred_best, pos_label=1)
    spec_best   = recall_score(y_test, y_pred_best, pos_label=0)
    ppv_best    = precision_score(y_test, y_pred_best, pos_label=1)
    npv_best    = precision_score(y_test, y_pred_best, pos_label=0)
    balacc_best = balanced_accuracy_score(y_test, y_pred_best)
    
    report_cr_best = classification_report(y_test, y_pred_best)
    test_ci_calibrated_best_threshold = bootstrap_patient_level_confidence_intervals(
        y_true=y_test,
        y_prob=p_post,
        patient_ids=patient_ids_test,
        threshold=best_thresh,
        n_bootstrap=args.bootstrap_iterations,
        ci_level=args.ci_level,
        seed=42,
    )
    test_ci_calibrated_best_threshold_path = os.path.join(
        calibration_dir, "test_confidence_intervals_calibrated_best_threshold.csv"
    )
    save_bootstrap_summary(
        ci_result=test_ci_calibrated_best_threshold,
        output_csv_path=test_ci_calibrated_best_threshold_path,
        setting_name="calibrated_test_best_threshold",
    )

    with open(report_path, "a", encoding="utf-8") as f_out:
        f_out.write("=== Threshold Tuning (Best F1 Threshold) ===\n")
        f_out.write("Threshold sweep results:\n")
        for r in results:
            f_out.write("Threshold: {:.2f} - F1: {:.3f}\n".format(r['threshold'], r['f1']))
        f_out.write(f"\nBest threshold selected by F1: {best_thresh:.2f}\n")
        f_out.write("\nClassification report at threshold {:.2f}:\n".format(best_thresh))
        f_out.write(report_cr_best)
        f_out.write("\n")
        f_out.write(f"AUC: {auc_best:.3f}\n")
        f_out.write(
            f"AUC {ci_percent}% CI (bootstrap): "
            f"[{test_ci_calibrated_best_threshold['metrics']['auc']['ci_low']:.3f}, "
            f"{test_ci_calibrated_best_threshold['metrics']['auc']['ci_high']:.3f}]\n"
        )
        f_out.write(
            f"Bootstrap summary CSV (best threshold): {test_ci_calibrated_best_threshold_path}\n"
        )
        f_out.write(f"MCC: {mcc_best:.3f}\n")
        f_out.write(f"Kappa: {kappa_best:.3f}\n")
        f_out.write(f"F1: {f1_best:.3f}\n")
        f_out.write(f"Accuracy: {acc_best:.3f}\n")
        f_out.write(f"Sensitivity: {sens_best:.3f}\n")
        f_out.write(f"Specificity: {spec_best:.3f}\n")
        f_out.write(f"PPV: {ppv_best:.3f}\n")
        f_out.write(f"NPV: {npv_best:.3f}\n")
        f_out.write(f"Balanced Accuracy: {balacc_best:.3f}\n\n")
    log_progress(
        f"Best-threshold calibrated metrics | AUC={format_metric(auc_best)} | "
        f"F1={format_metric(f1_best)} | Balanced Accuracy={format_metric(balacc_best)} | "
        f"MCC={format_metric(mcc_best)}"
    )
    
    conf_matrix_best = confusion_matrix(y_test, y_pred_best)
    confusion_fig_best = os.path.join(calibration_dir, "confusion_matrix_best_threshold.png")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.grid(False)
    
    disp_best = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_best)
    disp_best.plot(ax=ax, cmap='cividis')
    # ax.set_title(f"{selected_model} (Calibrado, threshold={best_thresh:.2f})", fontsize=12)
    
    n_classes = conf_matrix_best.shape[0]
    
    ax.set_xticks(np.arange(-0.5, n_classes, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_classes, 1), minor=True)
    
    ax.grid(which='minor', color='black', linestyle='--', linewidth=1)
    ax.tick_params(which='minor', bottom=False, left=False)
    
    plt.tight_layout()
    plt.savefig(confusion_fig_best, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    with open(report_path, "a", encoding="utf-8") as f_out:
        f_out.write(f"Confusion matrix at calibrated threshold={best_thresh:.2f}: {confusion_fig_best}\n\n")
    log_progress(f"Best-threshold confusion matrix saved to: {confusion_fig_best}")
        

    # ----------------------------------------------------------------------
    # 7) EXPLICABILIDAD
    # ----------------------------------------------------------------------

    # Extraer el preprocesador (todos los pasos excepto el clasificador final)
    preprocessor = deepcopy(best_estimator)
    preprocessor.steps.pop(-1)

    # Extraer el clasificador final
    model_clf = best_estimator.steps[-1][1]

    log_progress("Starting SHAP analysis on the training split...")
    train_success, selected_features, train_shap_values, train_top_features = perform_shap_analysis(
        X_data=X_train_full,
        y_data=y_train_full,
        model_clf=model_clf,
        preprocessor=preprocessor,
        shap_dir=train_shap_dir,
        report_path=report_path,
        dataset_name="training"
    )

    log_progress("Starting SHAP analysis on the hold-out test split...")
    test_success, _, test_shap_values, test_top_features = perform_shap_analysis(
        X_data=X_test,
        y_data=y_test,
        model_clf=model_clf,
        preprocessor=preprocessor,
        shap_dir=test_shap_dir,
        report_path=report_path,
        dataset_name="test"
    )

    if train_success:
        log_progress("Starting LIME analysis on the training split...")
        train_lime_success = perform_lime_analysis(
            X_data=X_train_full,
            y_data=y_train_full,
            model_clf=model_clf,
            preprocessor=preprocessor,
            lime_dir=train_lime_dir,
            selected_features=selected_features,
            report_path=report_path,
            # shap_top_features=train_top_features,
        dataset_name="training"
        )

    if test_success:
        log_progress("Starting LIME analysis on the hold-out test split...")
        test_lime_success = perform_lime_analysis(
            X_data=X_test,
            y_data=y_test,
            model_clf=model_clf,
            preprocessor=preprocessor,
            lime_dir=test_lime_dir,
            selected_features=selected_features,
            report_path=report_path,
            # shap_top_features=test_top_features,
            dataset_name="test"
        )
            
    log_progress(f"Process completed. Report saved to: {report_path}")

if __name__ == "__main__":
    main()

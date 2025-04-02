#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap

import matplotlib as mpl
mpl.use('Agg')
import scienceplots
plt.style.use(['science', 'grid'])
dpi = 300

import shap
from lime.lime_tabular import LimeTabularExplainer

from copy import deepcopy
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GroupShuffleSplit, StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
from sklearn.metrics import (
    roc_auc_score, matthews_corrcoef, cohen_kappa_score, f1_score,
    accuracy_score, recall_score, precision_score, balanced_accuracy_score,
    confusion_matrix, ConfusionMatrixDisplay, classification_report
)

from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

import joblib

from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests

def main():
    parser = argparse.ArgumentParser(
        description="Entrena y afina un modelo utilizando un conjunto de test hold-out definitivo y validación cruzada en el resto de los datos. Luego calibra el modelo y aplica SHAP si es posible."
    )
    parser.add_argument("--csv", type=str, default="features_all_gland.csv",
                        choices=["features_all_gland.csv", "features_all_full.csv"],
                        help="Ruta al CSV con las características (por defecto 'features_all_gland.csv').")
    parser.add_argument("--model", type=str, required=True,
                        choices=["SVM", "LogisticRegression", "RandomForest", 
                                 "NaiveBayes", "KNN", "GradientBoosting"],
                        help="Modelo a entrenar/optimizar.")
    parser.add_argument("--n_folds", type=int, default=5,
                        help="Número de folds para la validación cruzada en BayesSearchCV")
    parser.add_argument("--variables", type=str, required=True,
                        help="Ruta al archivo variables_usadas.txt con las variables a utilizar.")
    args = parser.parse_args()
    
    print("\nIniciando fine-tuning del modelo.")
    print(f"  --> Modelo seleccionado: {args.model}")
    print(f"  --> CSV utilizado: {args.csv}")
    print(f"  --> Archivo de variables: {args.variables}")
    
    selected_model = args.model
    base_dir = os.path.dirname(os.path.abspath(args.variables))
    output_parent_dir = os.path.join(base_dir, "best_results")
    calibration_dir = os.path.join(output_parent_dir, "calibration")
    shap_dir = os.path.join(output_parent_dir, "SHAP_analysis")
    
    os.makedirs(output_parent_dir, exist_ok=True)
    os.makedirs(calibration_dir, exist_ok=True)
    os.makedirs(shap_dir, exist_ok=True)
    
    print(f"\nCarpeta de salida creada/ubicada en: {os.path.relpath(output_parent_dir)}")
    
    # ----------------------------------------------------------------------
    # 1) CARGAR CSV E IDENTIFICAR X, y, groups
    # ----------------------------------------------------------------------
    pre_path = "../../../../../data/radiomic_data/"
    data_filename = str(args.csv) if args.csv else "features_all_gland.csv"
    data_path = os.path.join(pre_path, "concatenated_data", data_filename)
    
    print(f"\nCargando datos desde: {data_path}")
    df = pd.read_csv(data_path)
    df['patient_id_study_id'] = df['patient_id'].astype(str) + '_' + df['study_id'].astype(str)
    df = df.set_index('patient_id_study_id')
    print(f"Datos cargados. Dimensiones: {df.shape}")
    
    y = df["label"].values
    groups = df["patient_id"].values
    X = df.drop(columns=['patient_id', 'study_id', 'label'])
    
    # ----------------------------------------------------------------------
    # 1.1) FILTRAR LAS VARIABLES USADAS (variables_usadas.txt)
    # ----------------------------------------------------------------------
    print(f"\nFiltrando variables usando el archivo: {args.variables}")
    with open(args.variables, "r", encoding="utf-8") as f_vars:
        used_vars = [line.strip() for line in f_vars if line.strip()]
    X = X[used_vars]
    
    # ----------------------------------------------------------------------
    # 2) SEPARAR HOLD-OUT TEST SET Y CONJUNTO DE ENTRENAMIENTO
    # ----------------------------------------------------------------------
    gss = GroupShuffleSplit(test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))
    X_train_full, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train_full, y_test = y[train_idx], y[test_idx]
    groups_train_full = groups[train_idx]
    
    # ----------------------------------------------------------------------
    # 3) DEFINIR PIPELINE Y ESPACIO DE BÚSQUEDA CON OPTIMIZACIÓN BAYESIANA
    # ----------------------------------------------------------------------
    number_folds = args.n_folds
    score_group = {
        'roc_auc': 'roc_auc',
        'f1': 'f1',
        'balanced_accuracy': 'balanced_accuracy'
    }
    score_refit_str = 'roc_auc'
    random_state_value = 42
    
    if selected_model == 'SVM':
        pipe = make_pipeline(StandardScaler(),
                             VarianceThreshold(),
                             SVC(random_state=random_state_value, probability=True))
        param_grid = {
            'svc__C': Real(1e-4, 1e3, prior='log-uniform'),
            'svc__kernel': Categorical(['linear', 'rbf', 'poly']),
            'svc__gamma': Real(1e-4, 1e3, prior='log-uniform'),
            'svc__coef0': Real(0, 1)
        }
        
    elif selected_model == 'LogisticRegression':
        pipe = make_pipeline(StandardScaler(),
                             VarianceThreshold(),
                             LogisticRegression(
                                 class_weight='balanced', 
                                 random_state=random_state_value,
                                 solver='saga',  
                                 max_iter=10000
                             ))
        param_grid = {
            'logisticregression__C': Real(1e-4, 1e3, prior='log-uniform'),
            'logisticregression__penalty': Categorical(['l1', 'l2', 'elasticnet']),
            'logisticregression__l1_ratio': Real(0.1, 0.9)  
        }
        
    elif selected_model == 'RandomForest':
        pipe = make_pipeline(StandardScaler(),
                             VarianceThreshold(),
                             RandomForestClassifier(n_jobs=-1, 
                                                    class_weight="balanced_subsample", 
                                                    random_state=random_state_value))
        param_grid = {
            'randomforestclassifier__n_estimators': Integer(50, 1024),
            'randomforestclassifier__max_depth': Integer(1, 10),
            'randomforestclassifier__max_features': Categorical(['sqrt', 'log2', None]),
            'randomforestclassifier__min_samples_split': Integer(2, 20)
        }
        
    elif selected_model == 'NaiveBayes':
        pipe = make_pipeline(StandardScaler(),
                             VarianceThreshold(),
                             GaussianNB())
        param_grid = {}  # Sin hiperparámetros a tunear de forma clásica
        
    elif selected_model == 'KNN':
        pipe = make_pipeline(StandardScaler(),
                             VarianceThreshold(),
                             KNeighborsClassifier(n_jobs=-1))
        param_grid = {
            'kneighborsclassifier__n_neighbors': Integer(2, 8),
            'kneighborsclassifier__weights': Categorical(['uniform', 'distance'])
        }
        
    elif selected_model == 'GradientBoosting':
        pipe = make_pipeline(StandardScaler(),
                             VarianceThreshold(),
                             GradientBoostingClassifier(random_state=random_state_value))
        param_grid = {
            'gradientboostingclassifier__n_estimators': Integer(50, 1024),
            'gradientboostingclassifier__learning_rate': Real(1e-4, 0.1, prior='log-uniform'),
            'gradientboostingclassifier__max_depth': Integer(1, 10),
            'gradientboostingclassifier__subsample': Real(0.5, 1.0),
            'gradientboostingclassifier__max_features': Categorical(['sqrt', 'log2', None])
        }
    else:
        raise ValueError(f"Modelo '{selected_model}' no reconocido.")
    
    # ----------------------------------------------------------------------
    # 4) AJUSTAR CON BayesSearchCV (OPTIMIZACIÓN BAYESIANA) SOBRE EL CONJUNTO DE ENTRENAMIENTO
    # ----------------------------------------------------------------------
    cv = StratifiedGroupKFold(n_splits=number_folds, shuffle=True, random_state=random_state_value)
    print("\nIniciando optimización bayesiana con BayesSearchCV...")
    search = BayesSearchCV(
        estimator=pipe,
        search_spaces=param_grid,
        scoring=score_group,
        refit=score_refit_str,
        cv=cv,
        n_jobs=-1,
        random_state=random_state_value
    )
    search.fit(X_train_full, y_train_full, groups=groups_train_full)
    best_estimator = search.best_estimator_
    print("\nOptimización completada.")
    print(f"  --> Mejores parámetros: {search.best_params_}")

    estimator_path = os.path.join(output_parent_dir, "best_estimator.pkl")
    joblib.dump(best_estimator, estimator_path)
    print(f"  --> Mejor estimador guardado en: {os.path.relpath(estimator_path)}")

    # best_estimator = joblib.load(os.path.join(output_parent_dir, "best_estimator.pkl"))
    
    # ----------------------------------------------------------------------
    # 5) GUARDAR REPORTE EN "report.txt"
    # ----------------------------------------------------------------------
    report_path = os.path.join(output_parent_dir, "report.txt")
    with open(report_path, "w", encoding="utf-8") as f_out:
        f_out.write(f"=== Fine-tuning del modelo {selected_model} ===\n\n")
        f_out.write(f"Mejores parámetros (según {score_refit_str}): {search.best_params_}\n\n")
        f_out.write("=== Resultados CV (BayesSearch) ===\n")
        idx_best = search.best_index_
        for key in score_group:
            mean_test = search.cv_results_[f'mean_test_{key}'][idx_best]
            std_test  = search.cv_results_[f'std_test_{key}'][idx_best]
            f_out.write(f"  CV {key}: {mean_test:.3f} +/- {std_test:.3f}\n")
        f_out.write("\n")
    
    # ----------------------------------------------------------------------
    # 6) EVALUAR EN TEST
    # ----------------------------------------------------------------------
    print("\nEvaluando modelo en el conjunto de test (sin calibrar)...")
    y_pred_test = best_estimator.predict(X_test)
    
    # Matriz de confusión sin calibrar en el directorio padre
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
    ax.set_title(f"{selected_model} (NO calibrado)", fontsize=12)
    
    n_classes = len(disp.display_labels)
    
    ax.set_xticks(np.arange(-0.5, n_classes, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_classes, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='--', linewidth=1)
    ax.tick_params(which='minor', bottom=False, left=False)
    plt.tight_layout()
    plt.savefig(confusion_fig, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix guardada en: {confusion_fig}")
    
    if hasattr(best_estimator, "predict_proba"):
        auc_ = roc_auc_score(y_test, best_estimator.predict_proba(X_test)[:, 1])
    elif hasattr(best_estimator, "decision_function"):
        auc_ = roc_auc_score(y_test, best_estimator.decision_function(X_test))
    else:
        auc_ = np.nan
    
    mcc_    = matthews_corrcoef(y_test, y_pred_test)
    kappa_  = cohen_kappa_score(y_test, y_pred_test)
    f1_     = f1_score(y_test, y_pred_test)
    acc_    = accuracy_score(y_test, y_pred_test)
    sens_   = recall_score(y_test, y_pred_test, pos_label=1)
    spec_   = recall_score(y_test, y_pred_test, pos_label=0)
    ppv_    = precision_score(y_test, y_pred_test, pos_label=1)
    npv_    = precision_score(y_test, y_pred_test, pos_label=0)
    balacc_ = balanced_accuracy_score(y_test, y_pred_test)
    
    report_cr = classification_report(y_test, y_pred_test)
    
    with open(report_path, "a", encoding="utf-8") as f_out:
        f_out.write("=== Evaluación en Test (NO calibrado) ===\n")
        f_out.write(f"  Figura de Confusion Matrix: {confusion_fig}\n")
        f_out.write(f"  AUC: {auc_:.3f}\n")
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
    
    # --- Calibrar con Platt scaling (sigmoid, cv=5) ---
    print("\nCalibrando el modelo con Platt scaling (sigmoid, cv=5)...")
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
            
    ax.set_title(f"Calibration Curve (pre), {selected_model}", fontsize=14)
    plt.savefig(calibration_fig_pre, dpi=dpi, bbox_inches='tight')
    plt.close()
    print(f"  --> Calibration curve (pre) guardada en: {calibration_fig_pre}")
    
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
    ax.set_title(f"Calibration Curve (post), {selected_model}", fontsize=14)

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
    print(f"  --> Calibration curve (post) guardada en: {calibration_fig_post}")
    
    # --- Ajuste de umbral ---
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
    
    with open(report_path, "a", encoding="utf-8") as f_out:
        f_out.write("=== Ajuste de Umbral (Resultados con el mejor threshold) ===\n")
        f_out.write("Resultados para cada threshold:\n")
        for r in results:
            f_out.write("Threshold: {:.2f} - F1: {:.3f}\n".format(r['threshold'], r['f1']))
        f_out.write(f"\nMejor threshold seleccionado (según F1): {best_thresh:.2f}\n")
        f_out.write("\nClassification Report (con threshold {:.2f}):\n".format(best_thresh))
        f_out.write(report_cr_best)
        f_out.write("\n")
        f_out.write(f"AUC: {auc_best:.3f}\n")
        f_out.write(f"MCC: {mcc_best:.3f}\n")
        f_out.write(f"Kappa: {kappa_best:.3f}\n")
        f_out.write(f"F1: {f1_best:.3f}\n")
        f_out.write(f"Accuracy: {acc_best:.3f}\n")
        f_out.write(f"Sensitivity: {sens_best:.3f}\n")
        f_out.write(f"Specificity: {spec_best:.3f}\n")
        f_out.write(f"PPV: {ppv_best:.3f}\n")
        f_out.write(f"NPV: {npv_best:.3f}\n")
        f_out.write(f"Balanced Accuracy: {balacc_best:.3f}\n\n")
    
    # --- Matriz de confusión calibrada ---
    conf_matrix_best = confusion_matrix(y_test, y_pred_best)
    confusion_fig_best = os.path.join(calibration_dir, "confusion_matrix_best_threshold.png")
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.grid(False)
    
    disp_best = ConfusionMatrixDisplay(confusion_matrix=conf_matrix_best)
    disp_best.plot(ax=ax, cmap='cividis')
    ax.set_title(f"{selected_model} (Calibrado, threshold={best_thresh:.2f})", fontsize=12)
    
    n_classes = conf_matrix_best.shape[0]
    
    ax.set_xticks(np.arange(-0.5, n_classes, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, n_classes, 1), minor=True)
    
    ax.grid(which='minor', color='black', linestyle='--', linewidth=1)
    ax.tick_params(which='minor', bottom=False, left=False)
    
    plt.tight_layout()
    plt.savefig(confusion_fig_best, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    with open(report_path, "a", encoding="utf-8") as f_out:
        f_out.write(f"Confusion Matrix (Calibrado con threshold={best_thresh:.2f}) fig: {confusion_fig_best}\n\n")
        
    # ------------------
    # 7) SHAP ANALYSIS 
    # ------------------
    print("\nRealizando análisis SHAP...")
    try:
        preprocessor = deepcopy(best_estimator)
        preprocessor.steps.pop(-1) 
        
        # Aplicar StandardScaler conservando nombres
        scaler = preprocessor.steps[0][1]
        X_scaled = pd.DataFrame(scaler.transform(X_train_full),
                                index=X_train_full.index,
                                columns=X_train_full.columns)
        
        # Aplicar VarianceThreshold y recuperar columnas seleccionadas
        vt = preprocessor.steps[1][1]
        mask = vt.get_support()
        selected_features = X_train_full.columns[mask]
        X_transformed_array = vt.transform(X_scaled.values)
        X_transformed = pd.DataFrame(X_transformed_array,
                                     index=X_train_full.index,
                                     columns=selected_features)
        
        model_clf = best_estimator.steps[-1][1]
    
        # Seleccionar el explainer según el tipo de modelo
        if isinstance(model_clf, (RandomForestClassifier, GradientBoostingClassifier)):
            explainer = shap.TreeExplainer(model_clf)
        elif isinstance(model_clf, LogisticRegression):
            try:
                explainer = shap.LinearExplainer(model_clf, X_transformed)
            except Exception:
                background = shap.kmeans(X_transformed, 50)
                explainer = shap.KernelExplainer(model_clf.predict_proba, background)
        else:
            background = shap.kmeans(X_transformed, 50)
            explainer = shap.KernelExplainer(model_clf.predict_proba, background)
        
        # Calcular los valores SHAP
        shap_values = explainer(X_transformed)
        joblib.dump(shap_values, os.path.join(shap_dir, 'shap_values.pkl'))

        # shap_values = joblib.load(os.path.join(shap_dir, 'shap_values.pkl'))
        
        if shap_values.values.ndim > 2:
            shap_values = shap_values[:,:,1]
        
        # --------------------------------------------------------------
        # PARTE 1: TEST ESTADÍSTICO entre valores SHAP y la clase
        # --------------------------------------------------------------
        print(" - Realizando test estadístico (Mann-Whitney U) para cada feature con corrección Holm...")
        
        # Construimos DataFrame con valores SHAP
        shap_matrix = pd.DataFrame(
            shap_values.values,
            index=X_transformed.index,
            columns=X_transformed.columns
        )
        
        features_test = []
        pvalues_raw = []
        
        for feat in shap_matrix.columns:
            shap_class0 = shap_matrix.loc[y_train_full == 0, feat]
            shap_class1 = shap_matrix.loc[y_train_full == 1, feat]
            
            stat, pval = mannwhitneyu(shap_class0, shap_class1, alternative='two-sided')
            features_test.append(feat)
            pvalues_raw.append(pval)
        
        # Corrección por comparaciones múltiples
        alpha = 0.05
        reject, pvals_corr, _, _ = multipletests(pvalues_raw, alpha=alpha, method='holm')
        
        lines_output = []
        lines_output.append("=================================")
        lines_output.append("TEST DE MANN-WHITNEY U (SHAP por feature) con corrección 'Holm'")
        lines_output.append("Comparación: Clase 0 vs Clase 1")
        lines_output.append(f"alpha = {alpha}")
        lines_output.append(f"Features totales: {len(features_test)}") 
        lines_output.append("=================================\n")
        
        lines_output.append(f"Resultados por feature (p-value crudo y corregido):")
        significant_feats = []
        
        for feat, pval_raw, pval_corr, rej_bool in zip(features_test, pvalues_raw, pvals_corr, reject):
            if rej_bool:
                result_str = "=> DIFERENCIA SIGNIFICATIVA"
                significant_feats.append((feat, pval_raw, pval_corr))
            else:
                result_str = "=> sin diferencia significativa"
            
            lines_output.append(
                f"    {feat}: p-value crudo={pval_raw:.4e}, p-value corregido={pval_corr:.4e} {result_str}"
            )
        
        lines_output.append("")
        lines_output.append(f" Total comparaciones con diferencia significativa: {len(significant_feats)}. Comparaciones:")
        
        if not significant_feats:
            lines_output.append("    No se encontraron diferencias significativas.")
        else:
            for feat, pval_raw, pval_corr in significant_feats:
                lines_output.append(
                    f"    {feat}: p-value crudo={pval_raw:.4e}, p-value corregido={pval_corr:.4e} => DIFERENCIA SIGNIFICATIVA"
                )
        
        lines_output.append("\n")
        
        test_txt_path = os.path.join(shap_dir, "shap_statistical_test.txt")
        with open(test_txt_path, "w", encoding="utf-8") as f_out:
            for line in lines_output:
                f_out.write(line + "\n")
        
        print(f"  --> Test estadístico guardado en: {test_txt_path}")
    
        # --------------------------------------------------------------
        # PARTE 2: HEATMAP (clase 0 primero, luego clase 1)
        # --------------------------------------------------------------
        print(" - Generando Heatmap con muestras ordenadas por clase...")

        idx_class0 = np.where(y_train_full == 0)[0]
        idx_class1 = np.where(y_train_full == 1)[0]
        
        idx_order = np.concatenate([idx_class0, idx_class1])
        
        heatmap_path = os.path.join(shap_dir, "shap_heatmap.png")
        
        # Generamos el heatmap indicando el orden de las instancias
        shap.plots.heatmap(
            shap_values, 
            show=False,
            instance_order=idx_order
        )
        
        fig = plt.gcf()
        ax = plt.gca()
        
        split_position = len(idx_class0)
        ax.axvline(split_position - 0.5, color='black', linewidth=1, zorder=10)

        n_total = len(idx_order)
        mid_class0 = (split_position / 2) / n_total
        mid_class1 = (split_position + len(idx_class1)/2) / n_total
        
        # Añadimos etiquetas sobre la parte superior del heatmap
        ax.text(mid_class0, 1.01, 'Clase 0', ha='center', va='bottom', transform=ax.transAxes)
        ax.text(mid_class1, 1.01, 'Clase 1', ha='center', va='bottom', transform=ax.transAxes)
        
        fig.set_size_inches(10, 6)
        plt.tight_layout()
        plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"  --> Heatmap reordenado guardado en: {heatmap_path}")

    
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
        print(f"  --> Beeswarm plot guardado en: {shap_fig_path}")
    
        # --------------------------------------------------------------
        # Scatter plots de las top features
        # --------------------------------------------------------------
        scatter_dir = os.path.join(shap_dir, "scatter_plots")
        os.makedirs(scatter_dir, exist_ok=True)
    
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
        
        print(f"  --> Scatter plots de las variables más relevantes guardados en: {scatter_dir}")

    except Exception as e:
        with open(report_path, "a", encoding="utf-8") as f_out:
            f_out.write("=== SHAP Analysis ===\n")
            f_out.write(" No se pudo generar SHAP (modelo no soportado o error):\n")
            f_out.write(f"  {repr(e)}\n\n")
        print("Error en SHAP analysis:", e)

    # ------------------
    # 8) LIME ANALYSIS 
    # ------------------
    print("\nRealizando análisis LIME...")
    def extraer_nombre(feat_str: str) -> str:
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
        instance_label="instancia"
    ):
        exp = explainer.explain_instance(
            data_row=X_data[index],
            predict_fn=model_clf.predict_proba,
            num_features=10
        )
        
        explanation_txt_path = os.path.join(lime_dir, f"lime_explanation_{instance_label}_{index}.txt")
        fig_path = os.path.join(lime_dir, f"lime_explanation_{instance_label}_{index}.png")
        
        with open(explanation_txt_path, "w", encoding="utf-8") as f:
            f.write(f"=== LIME Explanation para {instance_label} (índice: {index}) ===\n\n")
            f.write(f"Clase real: {y_true[index]}\n")
            f.write(f"Predicción del modelo: {y_pred[index]}\n")
            f.write(f"Probabilidades: {model_clf.predict_proba([X_data[index]])}\n\n")
            f.write("Importancia local de las features:\n")
            for feat_info in exp.as_list():
                f.write("  {}: {:.4f}\n".format(feat_info[0], feat_info[1]))
        
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
            
            plt.title(f"LIME Explanation - {instance_label} (index={index})")
            plt.savefig(fig_path, dpi=dpi, bbox_inches='tight')
            plt.close(lime_fig)
        
        print(f"  -> LIME para {instance_label} (índice {index}) guardado en:\n"
              f"     {explanation_txt_path}\n"
              f"     {fig_path}")
    
    
    def generate_lime_explanations_for_misclassifications(
        X_test_lime,
        y_test,
        model_clf, 
        explainer,
        lime_dir
    ):
        y_pred = model_clf.predict(X_test_lime)
        
        # True Negative (TN): real=0, pred=0
        tn_indices = np.where((y_test == 0) & (y_pred == 0))[0]
        # True Positive (TP): real=1, pred=1
        tp_indices = np.where((y_test == 1) & (y_pred == 1))[0]
        # False Positive (FP): real=0, pred=1
        fp_indices = np.where((y_test == 0) & (y_pred == 1))[0]
        # False Negative (FN): real=1, pred=0
        fn_indices = np.where((y_test == 1) & (y_pred == 0))[0]
        
        def explain_first_if_any(indices, label):
            if len(indices) > 0:
                idx = indices[0]
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
                print(f"No hay instancias para {label}.")
        
        explain_first_if_any(tn_indices, "TN")
        explain_first_if_any(tp_indices, "TP")
        explain_first_if_any(fp_indices, "FP")
        explain_first_if_any(fn_indices, "FN")
        
    try:
        lime_dir = os.path.join(output_parent_dir, "LIME_analysis")
        os.makedirs(lime_dir, exist_ok=True)
    
        preprocessor_lime = deepcopy(best_estimator)
        preprocessor_lime.steps.pop(-1)
    
        X_train_lime = preprocessor_lime.transform(X_train_full)
        X_test_lime = preprocessor_lime.transform(X_test)
    
        model_clf = best_estimator.steps[-1][1]
    
        # 4) Definimos el explainer
        explainer_lime = LimeTabularExplainer(
            training_data=X_train_lime,
            feature_names=selected_features,  
            class_names=["0", "1"],
            discretize_continuous=True,
            random_state=42
        )

        resultados = []
        
        # 2) Iteramos en múltiples instancias
        for i in range(len(X_train_lime)):
            exp = explainer_lime.explain_instance(
                data_row=X_train_lime[i],
                predict_fn=model_clf.predict_proba
            )
            # Supongamos que tu modelo es binario y la clase de interés es 1
            lime_list = exp.as_list(label=1)  
            
            # LIME da pares (feature_str, weight). Te hará falta parsear feature_str
            # si LIME la "corta" con rangos, etc. O si la variable es binaria, etc.
            for (feat_str, peso) in lime_list:
                feature_name = extraer_nombre(feat_str) 
                col_idx = selected_features.get_loc(feature_name)
                valor_feature = X_train_lime[i, col_idx]
                
                resultados.append({
                    'instancia': i,
                    'feature': feature_name,
                    'peso': peso,
                    'valor_feature': valor_feature
                })
        
        df_lime = pd.DataFrame(resultados)
        
        df_lime['abs_peso'] = df_lime['peso'].abs()
        
        # 2) Seleccionar las 15 features con mayor |peso| promedio
        top_features = (
            df_lime.groupby('feature')['abs_peso']
            .mean()
            .sort_values(ascending=False)
            .head(15)
            .index
            .tolist()
        )
        
        df_lime_top15 = df_lime[df_lime['feature'].isin(top_features)].copy()
        
        # 2) Para *cada feature* en df_lime_top15, normalizamos valor_feature entre su min y su max local.
        df_lime_top15['valor_min'] = df_lime_top15.groupby('feature')['valor_feature'].transform('min')
        df_lime_top15['valor_max'] = df_lime_top15.groupby('feature')['valor_feature'].transform('max')
        
        df_lime_top15['valor_feature_norm'] = (
            (df_lime_top15['valor_feature'] - df_lime_top15['valor_min'])
            / (df_lime_top15['valor_max'] - df_lime_top15['valor_min'])
        )

        shap_order = list(top_features_shap)

        # Extraemos las variables únicas de LIME (orden en que aparecieron, por ejemplo)
        lime_features = df_lime_top15['feature'].unique().tolist()
        
        # Creamos la lista final:
        #  - Primero las que están en SHAP y también en LIME
        #  - Luego, las que están en LIME y no en SHAP (se añaden al final)
        final_order = [feat for feat in shap_order if feat in lime_features] + \
                      [feat for feat in lime_features if feat not in shap_order]

        # 3) Hacemos el stripplot usando hue='valor_feature_norm' y una paleta continua
        fig, ax = plt.subplots(figsize=(10,8))

        # Graficamos stripplot con hue numérico
        colors = [
            (0.0,  "#008afb"),
            (0.2, "#008afb"),  
            (0.7, "#ff0052"),  
            (1.0,  "#ff0052")  
        ]
        
        # 2) Creamos el colormap con esas secciones
        the_cmap = LinearSegmentedColormap.from_list("my_cmap", colors)

        sns.stripplot(
            data=df_lime_top15,
            x='peso',
            y='feature',
            hue='valor_feature_norm',    
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
        ax.set_title("Distribución de pesos LIME - Top 15 features")
        
        # Creamos un 'ScalarMappable' con la misma norma y colormap
        norm = mpl.colors.Normalize(vmin=0, vmax=1)
        sm = mpl.cm.ScalarMappable(cmap=the_cmap, norm=norm)
        sm.set_array([])
        
        # Añadimos la colorbar en la misma figura (robamos el espacio del ax)
        cbar = fig.colorbar(sm, ax=ax)
        cbar.set_label("Valor_feature normalizado [0..1]")
        
        plt.tight_layout()
        fig_path = os.path.join(lime_dir, "lime_pseudo_beeswarm.png")
        plt.savefig(fig_path, dpi=dpi, bbox_inches='tight')
        plt.close()


        
        # 5) Generamos explicaciones locales para algunas instancias.
        ind_lime_dir = os.path.join(lime_dir, "individual_analysis")
        os.makedirs(ind_lime_dir, exist_ok=True)
        
        generate_lime_explanations_for_misclassifications(
            X_test_lime=X_test_lime,
            y_test=y_test,
            model_clf=model_clf,
            explainer=explainer_lime,
            lime_dir=ind_lime_dir
        )

    
    except Exception as e:
        with open(report_path, "a", encoding="utf-8") as f_out:
            f_out.write("\n=== LIME Analysis ===\n")
            f_out.write("No se pudo generar LIME (modelo no soportado o error):\n")
            f_out.write(f"  {repr(e)}\n\n")
        print("Error en LIME analysis:", e)        
        
    print(f"\nProceso finalizado. Report guardado en: {report_path}")

if __name__ == "__main__":
    main()
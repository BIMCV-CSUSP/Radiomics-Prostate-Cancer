import argparse
import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import shapiro, mannwhitneyu, ttest_ind
from statsmodels.stats.multitest import multipletests
from sklearn import metrics
from sklearn.model_selection import StratifiedGroupKFold, GroupShuffleSplit
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.feature_selection import VarianceThreshold
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score, precision_score,
                            recall_score, balanced_accuracy_score, cohen_kappa_score,
                            matthews_corrcoef, confusion_matrix, ConfusionMatrixDisplay,
                            classification_report, roc_curve, auc)
from sklearn.calibration import CalibratedClassifierCV, CalibrationDisplay
import matplotlib as mpl
mpl.use('Agg')
import scienceplots

from skopt import BayesSearchCV
from skopt.space import Real, Integer, Categorical

from copy import deepcopy
import joblib
import shap

# Configuración de estilo para las gráficas
plt.style.use(['science', 'grid'])
dpi = 300

# ================================
# Función para definir los modelos
# ================================
def get_models(random_state=42):
    """
    Define pipelines para cada clasificador.
    """
    pipe_svc = make_pipeline(
        StandardScaler(),
        VarianceThreshold(),
        SVC(random_state=random_state, class_weight="balanced", probability=True)
    )
    
    pipe_lr = make_pipeline(
        StandardScaler(),
        VarianceThreshold(),
        LogisticRegression(
            penalty='elasticnet', l1_ratio=0.5,
            class_weight="balanced",
            random_state=random_state,
            solver='saga', max_iter=10000
        )
    )
    
    pipe_rf = make_pipeline(
        StandardScaler(),
        VarianceThreshold(),
        RandomForestClassifier(n_jobs=-1, class_weight="balanced_subsample", random_state=random_state)
    )
    
    pipe_nb = make_pipeline(
        StandardScaler(),
        VarianceThreshold(),
        GaussianNB()
    )
    
    pipe_knn = make_pipeline(
        StandardScaler(),
        VarianceThreshold(),
        KNeighborsClassifier(n_jobs=-1)
    )
    
    pipe_gb = make_pipeline(
        StandardScaler(),
        VarianceThreshold(),
        GradientBoostingClassifier(random_state=random_state)
    )
    
    models = [
        ("SVM", pipe_svc),
        ("Logistic Regression", pipe_lr),
        ("Random Forest", pipe_rf),
        ("Naive Bayes", pipe_nb),
        ("KNN", pipe_knn),
        ("Gradient Boosting", pipe_gb),
    ]
    return models

# ========================================
# Función de selección de características
# ========================================
def feature_selection_with_graphs(X, y, outdir, df_original, dpi=300):
    """
    Selección de características "most_discriminant" con generación de gráficos.
    Se guarda un CSV con los p-valores y AUC, y para las TOP 20 features se generan:
      - Violinplot de la distribución
      - Curva ROC
    Devuelve X filtrado y la lista de features seleccionadas.
    """
    os.makedirs(outdir, exist_ok=True)
    images_dir = os.path.join(outdir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    feature_names = []
    sensitivity_list, specificity_list = [], []
    auc_list, threshold_list = [], []
    test_type_list, pvalue_list, pos_vs_neg_list = [], [], []
    
    for column in X.columns:
        stat, p = shapiro(X[column])
        a_dist = X[column][y == 0]
        b_dist = X[column][y == 1]
        
        feature_names.append(column)
        alpha = 0.05
        if p > alpha:
            test_type_list.append('t-test')
            _, pval = ttest_ind(a_dist, b_dist)
        else:
            test_type_list.append('mann-whitney U-test')
            _, pval = mannwhitneyu(a_dist, b_dist)
        pvalue_list.append(pval)
        
        fpr, tpr, thresholds = metrics.roc_curve(y, X[column], pos_label=1)
        auc_val = metrics.auc(fpr, tpr)
        pos_vs_neg = ">"
        if auc_val < 0.5:
            fpr, tpr, thresholds = metrics.roc_curve(y, X[column], pos_label=0)
            auc_val = metrics.auc(fpr, tpr)
            pos_vs_neg = "<"
        auc_list.append(auc_val)
        pos_vs_neg_list.append(pos_vs_neg)
        
        roc_df = pd.DataFrame({
            'fpr': fpr,
            'tpr': tpr,
            '1-fpr': 1 - fpr,
            'tf': tpr - (1 - fpr),
            'thresholds': thresholds
        })
        cutoff_df = roc_df.iloc[(roc_df.tf - 0).abs().argsort()[:1]]
        sensitivity_list.append(cutoff_df['tpr'].values[0])
        specificity_list.append(cutoff_df['1-fpr'].values[0])
        threshold_list.append(cutoff_df['thresholds'].values[0])
    
    train_auc_pvals_df = pd.DataFrame(
        list(zip(auc_list, pos_vs_neg_list, threshold_list,
                 sensitivity_list, specificity_list, 
                 test_type_list, pvalue_list)),
        index=feature_names,
        columns=['AUC', 'Pos.vs.Neg.', 'Cutoff-Threshold', 'Sensitivity',
                 'Specificity', 'Test', 'p-value']
    ).sort_values(by='p-value', ascending=True)
    
    num_features_model = max(1, round(X.shape[0] / 15))
    train_df = train_auc_pvals_df.sort_values(by='p-value', ascending=True)
    selected_features = train_df.index[:num_features_model]
    print(f"  --> Seleccionadas {len(selected_features)} características más relevantes.")
    
    X_filtered = X[selected_features]
    
    csv_path = os.path.join(outdir, "train_auc_pvals_df.csv")
    train_auc_pvals_df.to_csv(csv_path)
    print(f"  --> Guardado CSV: {csv_path}\n")
    
    top_20 = train_auc_pvals_df.index[:20]
    for rank, feature_name in enumerate(top_20, start=1):
        safe_feat_name = feature_name.replace("/", "_")
        feat_folder_name = f"{rank}_{safe_feat_name}"
        feat_folder_path = os.path.join(images_dir, feat_folder_name)
        os.makedirs(feat_folder_path, exist_ok=True)
        
        # Violinplot
        plt.figure(figsize=(9, 9))
        sns.violinplot(x=y, y=df_original[feature_name], color='grey')
        plt.title(f"Distribución de {feature_name}\n(no-csPCa vs csPCa)", fontsize=14)
        plt.xlabel("Clases")
        plt.xticks([0, 1], ["no-csPCa", "csPCa"], fontsize=12)
        violin_plot_path = os.path.join(feat_folder_path, f"{safe_feat_name}_violinplot.png")
        plt.savefig(violin_plot_path, dpi=dpi)
        plt.close()
        
        # Curva ROC
        fpr, tpr, _ = metrics.roc_curve(y, df_original[feature_name], pos_label=1)
        auc_val = metrics.auc(fpr, tpr)
        plt.figure(figsize=(6, 6))
        plt.plot(fpr, tpr, marker='.', color='black', markersize=3, label=f"{feature_name} (AUC={auc_val:.3f})")
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend()
        plt.title(f"Curva ROC: {feature_name}")
        roc_plot_path = os.path.join(feat_folder_path, f"{safe_feat_name}_ROC.png")
        plt.savefig(roc_plot_path, dpi=dpi)
        plt.close()
    
    return X_filtered, selected_features

# ============================================================
# Función para evaluar un modelo dado un conjunto de índices (fold)
# ============================================================
def evaluate_model(model, X, y, train_idx, val_idx):
    """
    Evalúa un modelo dado usando los índices de entrenamiento y validación.
    Devuelve un diccionario con todas las métricas y predicciones.
    """
    X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
    y_train, y_val = y[train_idx], y[val_idx]
    
    model.fit(X_train, y_train)
    
    # Predicciones en entrenamiento
    y_train_pred = model.predict(X_train)
    if hasattr(model, "predict_proba"):
        y_train_prob = model.predict_proba(X_train)[:, 1]
    elif hasattr(model, "decision_function"):
        y_train_prob = model.decision_function(X_train)
    else:
        y_train_prob = None
    try:
        train_auc = roc_auc_score(y_train, y_train_prob) if y_train_prob is not None else np.nan
    except Exception:
        train_auc = np.nan
    train_f1 = f1_score(y_train, y_train_pred, average="binary")
    
    # Predicciones en validación
    y_val_pred = model.predict(X_val)
    if hasattr(model, "predict_proba"):
        y_val_prob = model.predict_proba(X_val)[:, 1]
    elif hasattr(model, "decision_function"):
        y_val_prob = model.decision_function(X_val)
    else:
        y_val_prob = None
    try:
        val_auc = roc_auc_score(y_val, y_val_prob) if y_val_prob is not None else np.nan
    except Exception:
        val_auc = np.nan
    mcc = matthews_corrcoef(y_val, y_val_pred)
    kappa = cohen_kappa_score(y_val, y_val_pred)
    f1_bin = f1_score(y_val, y_val_pred, average="binary")
    f1_macro = f1_score(y_val, y_val_pred, average="macro")
    acc = accuracy_score(y_val, y_val_pred)
    bal_acc = balanced_accuracy_score(y_val, y_val_pred)
    sens = recall_score(y_val, y_val_pred, pos_label=1)
    spec = recall_score(y_val, y_val_pred, pos_label=0)
    ppv = precision_score(y_val, y_val_pred, pos_label=1)
    
    cm = confusion_matrix(y_val, y_val_pred)
    if (cm[0, 0] + cm[1, 0]) > 0:
        npv = cm[0, 0] / (cm[0, 0] + cm[1, 0])
    else:
        npv = np.nan
    
    per_class_precision = precision_score(y_val, y_val_pred, average=None)
    per_class_recall = recall_score(y_val, y_val_pred, average=None)
    per_class_f1 = f1_score(y_val, y_val_pred, average=None)
    per_class_accuracy = []
    for i in range(len(cm)):
        row_sum = np.sum(cm[i, :])
        if row_sum > 0:
            per_class_accuracy.append(cm[i, i] / row_sum)
        else:
            per_class_accuracy.append(np.nan)
    
    return {
        "y_val": y_val,
        "y_val_pred": y_val_pred,
        "y_val_prob": y_val_prob,
        "train_auc": train_auc,
        "train_f1": train_f1,
        "val_auc": val_auc,
        "val_mcc": mcc,
        "val_kappa": kappa,
        "val_f1_binary": f1_bin,
        "val_f1_macro": f1_macro,
        "val_accuracy": acc,
        "val_balanced_accuracy": bal_acc,
        "val_sensitivity": sens,
        "val_specificity": spec,
        "val_ppv": ppv,
        "val_npv": npv,
        "per_class_precision": per_class_precision.tolist(),
        "per_class_recall": per_class_recall.tolist(),
        "per_class_f1": per_class_f1.tolist(),
        "per_class_accuracy": per_class_accuracy
    }

# ============================================================
# Función para ejecutar validación cruzada en un ensamble dado
# ============================================================
def run_cv_ensemble(model_t2, model_adc, model_dwi,
                    X_t2, X_adc, X_dwi, y, groups,
                    n_splits=5, n_repeats=2, base_random_state=42):
    """
    Ejecuta validación cruzada para un ensamble de tres modelos (uno por secuencia).
    Devuelve los resultados por fold y los datos necesarios para las curvas ROC.
    """
    fold_results = []
    roc_data = []  # Lista para almacenar datos para curvas ROC
    global_fold = 0
    
    for rep in range(n_repeats):
        current_state = base_random_state + rep
        splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=current_state)
        
        for train_idx, val_idx in splitter.split(X_t2, y, groups=groups):
            global_fold += 1
            # Evaluar cada modelo en su secuencia
            res_t2 = evaluate_model(model_t2, X_t2, y, train_idx, val_idx)
            res_adc = evaluate_model(model_adc, X_adc, y, train_idx, val_idx)
            res_dwi = evaluate_model(model_dwi, X_dwi, y, train_idx, val_idx)
            
            # Promediar métricas de entrenamiento de las tres secuencias
            ensemble_train_auc = np.nanmean([res_t2["train_auc"], res_adc["train_auc"], res_dwi["train_auc"]])
            ensemble_train_f1  = np.nanmean([res_t2["train_f1"], res_adc["train_f1"], res_dwi["train_f1"]])
            
            # Ensemble (votación suave): promediamos las probabilidades en validación
            probs = []
            for res in [res_t2, res_adc, res_dwi]:
                if res["y_val_prob"] is not None:
                    probs.append(res["y_val_prob"])
                else:
                    probs.append(np.zeros_like(res_t2["y_val_pred"], dtype=float))
            ensemble_prob = np.mean(probs, axis=0)
            ensemble_pred = (ensemble_prob >= 0.5).astype(int)
            
            y_val = y[val_idx]
            try:
                ensemble_val_auc = roc_auc_score(y_val, ensemble_prob)
            except Exception:
                ensemble_val_auc = np.nan
            ensemble_mcc = matthews_corrcoef(y_val, ensemble_pred)
            ensemble_kappa = cohen_kappa_score(y_val, ensemble_pred)
            ensemble_f1_bin = f1_score(y_val, ensemble_pred, average="binary")
            ensemble_f1_macro = f1_score(y_val, ensemble_pred, average="macro")
            ensemble_acc = accuracy_score(y_val, ensemble_pred)
            ensemble_bal_acc = balanced_accuracy_score(y_val, ensemble_pred)
            ensemble_sens = recall_score(y_val, ensemble_pred, pos_label=1)
            ensemble_spec = recall_score(y_val, ensemble_pred, pos_label=0)
            ensemble_ppv = precision_score(y_val, ensemble_pred, pos_label=1)
            cm = confusion_matrix(y_val, ensemble_pred)
            if (cm[0, 0] + cm[1, 0]) > 0:
                ensemble_npv = cm[0, 0] / (cm[0, 0] + cm[1, 0])
            else:
                ensemble_npv = np.nan
            
            per_class_precision = precision_score(y_val, ensemble_pred, average=None)
            per_class_recall = recall_score(y_val, ensemble_pred, average=None)
            per_class_f1 = f1_score(y_val, ensemble_pred, average=None)
            per_class_accuracy = []
            for i in range(len(cm)):
                row_sum = np.sum(cm[i, :])
                if row_sum > 0:
                    per_class_accuracy.append(cm[i, i] / row_sum)
                else:
                    per_class_accuracy.append(np.nan)
            
            fold_metrics = {
                "Fold": global_fold,
                "Repeat": rep + 1,
                "train_auc": ensemble_train_auc,
                "train_f1": ensemble_train_f1,
                "val_auc": ensemble_val_auc,
                "val_mcc": ensemble_mcc,
                "val_kappa": ensemble_kappa,
                "val_f1_binary": ensemble_f1_bin,
                "val_f1_macro": ensemble_f1_macro,
                "val_accuracy": ensemble_acc,
                "val_balanced_accuracy": ensemble_bal_acc,
                "val_sensitivity": ensemble_sens,
                "val_specificity": ensemble_spec,
                "val_ppv": ensemble_ppv,
                "val_npv": ensemble_npv,
                "per_class_precision": per_class_precision.tolist(),
                "per_class_recall": per_class_recall.tolist(),
                "per_class_f1": per_class_f1.tolist(),
                "per_class_accuracy": per_class_accuracy
            }
            
            fold_results.append(fold_metrics)
            
            # Guardar datos para curvas ROC
            try:
                # Calculamos curva ROC para este fold
                fpr, tpr, _ = roc_curve(y_val, ensemble_prob)
                roc_auc = auc(fpr, tpr)
                
                roc_data.append({
                    "fold": global_fold,
                    "fpr": fpr,
                    "tpr": tpr,
                    "auc": roc_auc,
                    "y_true": y_val,
                    "y_prob": ensemble_prob
                })
            except Exception as e:
                print(f"Error al calcular ROC para fold {global_fold}: {e}")
    
    return fold_results, roc_data

# ============================================================
# Función para generar curvas ROC óptimas y medianas
# ============================================================
def generate_roc_curves(results_df, roc_data_all, outdir, top_n=6):
    """
    Genera curvas ROC para los mejores N ensambles.
    Muestra la curva óptima y mediana para cada ensamble.
    """
    os.makedirs(outdir, exist_ok=True)
    
    # Paleta de colores
    my_colors = [
        "#0072B2",  # Azul oscuro
        "#009E73",  # Verde
        "#D55E00",  # Naranja rojizo
        "#CC78BC",  # Morado
        "#DE8F05",  # Marrón/naranja
        "#56B4E9"   # Azul claro
    ]
    
    # Seleccionar los mejores N ensambles
    top_ensembles = results_df.sort_values(by="Avg_Val_AUC", ascending=False).head(top_n)["Ensemble"].tolist()
    
    # Filtrar datos ROC para estos ensambles
    roc_data_filtered = {ensemble: [] for ensemble in top_ensembles}
    
    for ensemble in top_ensembles:
        ensemble_data = [data for data in roc_data_all if data["ensemble"] == ensemble]
        roc_data_filtered[ensemble] = ensemble_data
    
    # 1. ROC óptimas (mejores folds por AUC)
    optimal_curves = []
    for ensemble in top_ensembles:
        ensemble_data = roc_data_filtered[ensemble]
        if ensemble_data:
            # Encontrar el fold con mejor AUC
            best_fold = max(ensemble_data, key=lambda x: x["auc"])
            optimal_curves.append({
                "ensemble": ensemble,
                "fold": best_fold["fold"],
                "fpr": best_fold["fpr"],
                "tpr": best_fold["tpr"],
                "auc": best_fold["auc"]
            })
    
    # 2. ROC medianas (folds con AUC más cercano a la mediana)
    median_curves = []
    for ensemble in top_ensembles:
        ensemble_data = roc_data_filtered[ensemble]
        if ensemble_data:
            # Calcular AUC mediano para este ensamble
            aucs = [data["auc"] for data in ensemble_data]
            median_auc = np.median(aucs)
            # Encontrar el fold con AUC más cercano a la mediana
            median_fold = min(ensemble_data, key=lambda x: abs(x["auc"] - median_auc))
            median_curves.append({
                "ensemble": ensemble,
                "fold": median_fold["fold"],
                "fpr": median_fold["fpr"],
                "tpr": median_fold["tpr"],
                "auc": median_fold["auc"]
            })
    
    # Crear gráficas
    # 1. ROC óptimas
    fig_opt, ax_opt = plt.subplots(figsize=(10, 8))
    for i, curve in enumerate(optimal_curves):
        color = my_colors[i % len(my_colors)]
        ensemble_short = curve["ensemble"].replace(" + ", "+")
        ax_opt.plot(curve["fpr"], curve["tpr"], 
                   label=f"{ensemble_short} (AUC={curve['auc']:.3f})",
                   color=color, linewidth=2)
    
    ax_opt.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1.5)
    ax_opt.set_xlabel("False Positive Rate", fontsize=12)
    ax_opt.set_ylabel("True Positive Rate", fontsize=12)
    ax_opt.set_title("Curvas ROC: Folds óptimos para los mejores ensambles", fontsize=14)
    ax_opt.grid(True, linestyle='--', alpha=0.7)
    ax_opt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "roc_optimal_ensembles.png"), dpi=dpi, bbox_inches='tight')
    plt.close()
    
    # 2. ROC medianas
    fig_med, ax_med = plt.subplots(figsize=(10, 8))
    for i, curve in enumerate(median_curves):
        color = my_colors[i % len(my_colors)]
        ensemble_short = curve["ensemble"].replace(" + ", "+")
        ax_med.plot(curve["fpr"], curve["tpr"], 
                   label=f"{ensemble_short} (AUC={curve['auc']:.3f})",
                   color=color, linewidth=2)
    
    ax_med.plot([0, 1], [0, 1], linestyle='--', color='gray', linewidth=1.5)
    ax_med.set_xlabel("False Positive Rate", fontsize=12)
    ax_med.set_ylabel("True Positive Rate", fontsize=12)
    ax_med.set_title("Curvas ROC: Folds medianos para los mejores ensambles", fontsize=14)
    ax_med.grid(True, linestyle='--', alpha=0.7)
    ax_med.legend(loc="lower right", fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, "roc_median_ensembles.png"), dpi=dpi, bbox_inches='tight')
    plt.close()
    
    return optimal_curves, median_curves

# ============================================================
# Función para optimizar modelos mediante BayesSearch
# ============================================================
def optimize_models(X_t2, X_adc, X_dwi, y, groups, best_ensemble, outdir, n_folds=5, random_state=42):
    """
    Optimiza los parámetros de los tres modelos del mejor ensamble usando BayesSearchCV.
    """
    print(f"\nOptimizando modelos para el ensamble: {best_ensemble}")
    os.makedirs(outdir, exist_ok=True)
    
    # Desempaquetar nombres de modelos
    model_names = best_ensemble.split(" + ")
    model_t2_name, model_adc_name, model_dwi_name = model_names
    
    # Mapeo de nombres a espacios de búsqueda
    param_spaces = {}
    optimized_models = {}
    
    # Función auxiliar para crear el espacio de parámetros basado en el tipo de modelo
    def create_param_space(model_name):
        if "SVM" in model_name:
            pipe = make_pipeline(
                StandardScaler(),
                VarianceThreshold(),
                SVC(random_state=random_state, probability=True)
            )
            param_space = {
                'svc__C': Real(1e-4, 1e3, prior='log-uniform'),
                'svc__kernel': Categorical(['linear', 'rbf', 'poly']),
                'svc__gamma': Real(1e-4, 1e3, prior='log-uniform'),
                'svc__class_weight': Categorical(['balanced', None])
            }
            return pipe, param_space
                 
        elif "Logistic Regression" in model_name:
            pipe = make_pipeline(
                StandardScaler(),
                VarianceThreshold(),
                LogisticRegression(random_state=random_state, max_iter=10000, solver='saga')
            )
            param_space = {
                'logisticregression__C': Real(1e-4, 1e3, prior='log-uniform'),
                'logisticregression__penalty': Categorical(['l1', 'l2', 'elasticnet']),
                'logisticregression__l1_ratio': Real(0.1, 0.9),
                'logisticregression__class_weight': Categorical(['balanced', None])
            }
            return pipe, param_space
            
        elif "Random Forest" in model_name:
            pipe = make_pipeline(
                StandardScaler(),
                VarianceThreshold(),
                RandomForestClassifier(random_state=random_state, n_jobs=-1)
            )
            param_space = {
                'randomforestclassifier__n_estimators': Integer(50, 500),
                'randomforestclassifier__max_depth': Integer(3, 20),
                'randomforestclassifier__min_samples_split': Integer(2, 20),
                'randomforestclassifier__max_features': Categorical(['sqrt', 'log2', None]),
                'randomforestclassifier__class_weight': Categorical(['balanced', 'balanced_subsample', None])
            }
            return pipe, param_space
            
        elif "Naive Bayes" in model_name:
            pipe = make_pipeline(
                StandardScaler(),
                VarianceThreshold(),
                GaussianNB()
            )
            param_space = {
                'gaussiannb__var_smoothing': Real(1e-10, 1e-8, prior='log-uniform')
            }
            return pipe, param_space
            
        elif "KNN" in model_name:
            pipe = make_pipeline(
                StandardScaler(),
                VarianceThreshold(),
                KNeighborsClassifier(n_jobs=-1)
            )
            param_space = {
                'kneighborsclassifier__n_neighbors': Integer(3, 15),
                'kneighborsclassifier__weights': Categorical(['uniform', 'distance']),
                'kneighborsclassifier__p': Integer(1, 2)
            }
            return pipe, param_space
            
        elif "Gradient Boosting" in model_name:
            pipe = make_pipeline(
                StandardScaler(),
                VarianceThreshold(),
                GradientBoostingClassifier(random_state=random_state)
            )
            param_space = {          
                'gradientboostingclassifier__n_estimators': Integer(50, 500),
                'gradientboostingclassifier__learning_rate': Real(0.01, 0.2, prior='log-uniform'),
                'gradientboostingclassifier__max_depth': Integer(3, 10),
                'gradientboostingclassifier__subsample': Real(0.5, 1.0),
                'gradientboostingclassifier__min_samples_split': Integer(2, 20)
            }
            return pipe, param_space
    
    # Crear y optimizar cada modelo
    # T2
    print(f"Optimizando modelo para secuencia T2: {model_t2_name}")
    pipe_t2, param_space_t2 = create_param_space(model_t2_name)
    cv_t2 = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    search_t2 = BayesSearchCV(
        estimator=pipe_t2,
        search_spaces=param_space_t2,
        scoring='roc_auc',
        cv=cv_t2,
        n_jobs=-1,
        n_iter=30,
        random_state=random_state
    )
    search_t2.fit(X_t2, y, groups=groups)
    optimized_models['t2'] = search_t2.best_estimator_
    
    # ADC
    print(f"Optimizando modelo para secuencia ADC: {model_adc_name}")
    pipe_adc, param_space_adc = create_param_space(model_adc_name)
    cv_adc = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    search_adc = BayesSearchCV(
        estimator=pipe_adc,
        search_spaces=param_space_adc,
        scoring='roc_auc',
        cv=cv_adc,
        n_jobs=-1,
        n_iter=30,
        random_state=random_state
    )
    search_adc.fit(X_adc, y, groups=groups)
    optimized_models['adc'] = search_adc.best_estimator_
    
    # DWI
    print(f"Optimizando modelo para secuencia DWI: {model_dwi_name}")
    pipe_dwi, param_space_dwi = create_param_space(model_dwi_name)
    cv_dwi = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    
    search_dwi = BayesSearchCV(
        estimator=pipe_dwi,
        search_spaces=param_space_dwi,
        scoring='roc_auc',
        cv=cv_dwi,
        n_jobs=-1,
        n_iter=30,
        random_state=random_state
    )
    search_dwi.fit(X_dwi, y, groups=groups)
    optimized_models['dwi'] = search_dwi.best_estimator_
    
    # Guardar modelos y mejores parámetros
    for key, model in optimized_models.items():
        model_path = os.path.join(outdir, f"best_{key}_model.pkl")
        joblib.dump(model, model_path)
        print(f"  Modelo {key} guardado en: {model_path}")
    
    # Guardar informe de mejores parámetros
    params_report = {
        't2': search_t2.best_params_,
        'adc': search_adc.best_params_,
        'dwi': search_dwi.best_params_
    }
    
    scores_report = {
        't2': search_t2.best_score_,
        'adc': search_adc.best_score_,
        'dwi': search_dwi.best_score_
    }
    
    report_path = os.path.join(outdir, "optimization_report.txt")
    with open(report_path, "w") as f:
        f.write(f"==== Optimización de Modelos para Ensamble: {best_ensemble} ====\n\n")
        
        f.write("=== Modelo T2 ===\n")
        f.write(f"Mejor puntuación (AUC): {scores_report['t2']:.4f}\n")
        f.write("Mejores parámetros:\n")
        for param, value in params_report['t2'].items():
            f.write(f"  {param}: {value}\n")
        f.write("\n")
        
        f.write("=== Modelo ADC ===\n")
        f.write(f"Mejor puntuación (AUC): {scores_report['adc']:.4f}\n")
        f.write("Mejores parámetros:\n")
        for param, value in params_report['adc'].items():
            f.write(f"  {param}: {value}\n")
        f.write("\n")
        
        f.write("=== Modelo DWI ===\n")
        f.write(f"Mejor puntuación (AUC): {scores_report['dwi']:.4f}\n")
        f.write("Mejores parámetros:\n")
        for param, value in params_report['dwi'].items():
            f.write(f"  {param}: {value}\n")
        
    print(f"Informe de optimización guardado en: {report_path}")
    
    return optimized_models

# ============================================================
# Función para evaluar el ensemble optimizado en conjunto de prueba
# ============================================================
def evaluate_optimized_ensemble(optimized_models, X_t2_test, X_adc_test, X_dwi_test, y_test, outdir):
    """
    Evalúa el ensamble optimizado en un conjunto de prueba y genera métricas y gráficos.
    """
    print("\nEvaluando ensamble optimizado en conjunto de prueba...")
    os.makedirs(outdir, exist_ok=True)
    
    # Obtener probabilidades de cada modelo
    if hasattr(optimized_models['t2'], "predict_proba"):
        y_pred_prob_t2 = optimized_models['t2'].predict_proba(X_t2_test)[:, 1]
    elif hasattr(optimized_models['t2'], "decision_function"):
        y_pred_prob_t2 = optimized_models['t2'].decision_function(X_t2_test)
        y_pred_prob_t2 = (y_pred_prob_t2 - y_pred_prob_t2.min()) / (y_pred_prob_t2.max() - y_pred_prob_t2.min())
    else:
        y_pred_prob_t2 = optimized_models['t2'].predict(X_t2_test).astype(float)
    
    if hasattr(optimized_models['adc'], "predict_proba"):
        y_pred_prob_adc = optimized_models['adc'].predict_proba(X_adc_test)[:, 1]
    elif hasattr(optimized_models['adc'], "decision_function"):
        y_pred_prob_adc = optimized_models['adc'].decision_function(X_adc_test)
        y_pred_prob_adc = (y_pred_prob_adc - y_pred_prob_adc.min()) / (y_pred_prob_adc.max() - y_pred_prob_adc.min())
    else:
        y_pred_prob_adc = optimized_models['adc'].predict(X_adc_test).astype(float)
    
    if hasattr(optimized_models['dwi'], "predict_proba"):
        y_pred_prob_dwi = optimized_models['dwi'].predict_proba(X_dwi_test)[:, 1]
    elif hasattr(optimized_models['dwi'], "decision_function"):
        y_pred_prob_dwi = optimized_models['dwi'].decision_function(X_dwi_test)
        y_pred_prob_dwi = (y_pred_prob_dwi - y_pred_prob_dwi.min()) / (y_pred_prob_dwi.max() - y_pred_prob_dwi.min())
    else:
        y_pred_prob_dwi = optimized_models['dwi'].predict(X_dwi_test).astype(float)
    
    # Ensamble por promedio (votación suave)
    y_pred_prob_ensemble = np.mean([y_pred_prob_t2, y_pred_prob_adc, y_pred_prob_dwi], axis=0)
    y_pred_ensemble = (y_pred_prob_ensemble >= 0.5).astype(int)
    
    # Calcular métricas
    try:
        ensemble_auc = roc_auc_score(y_test, y_pred_prob_ensemble)
    except Exception as e:
        print(f"Error al calcular AUC: {e}")
        ensemble_auc = np.nan
    
    ensemble_mcc = matthews_corrcoef(y_test, y_pred_ensemble)
    ensemble_kappa = cohen_kappa_score(y_test, y_pred_ensemble)
    ensemble_f1 = f1_score(y_test, y_pred_ensemble)
    ensemble_accuracy = accuracy_score(y_test, y_pred_ensemble)
    ensemble_balanced_acc = balanced_accuracy_score(y_test, y_pred_ensemble)
    ensemble_sensitivity = recall_score(y_test, y_pred_ensemble, pos_label=1)
    ensemble_specificity = recall_score(y_test, y_pred_ensemble, pos_label=0)
    ensemble_ppv = precision_score(y_test, y_pred_ensemble, pos_label=1)
    
    cm = confusion_matrix(y_test, y_pred_ensemble)
    if (cm[0, 0] + cm[1, 0]) > 0:
        ensemble_npv = cm[0, 0] / (cm[0, 0] + cm[1, 0])
    else:
        ensemble_npv = np.nan
    
    # Guardar informe de métricas
    report_path = os.path.join(outdir, "test_performance.txt")
    with open(report_path, "w") as f:
        f.write("==== Evaluación del Ensamble Optimizado en Conjunto de Prueba ====\n\n")
        
        f.write("=== Métricas Generales ===\n")
        f.write(f"AUC: {ensemble_auc:.4f}\n")
        f.write(f"MCC: {ensemble_mcc:.4f}\n")
        f.write(f"Kappa: {ensemble_kappa:.4f}\n")
        f.write(f"F1-Score: {ensemble_f1:.4f}\n")
        f.write(f"Accuracy: {ensemble_accuracy:.4f}\n")
        f.write(f"Balanced Accuracy: {ensemble_balanced_acc:.4f}\n")
        f.write(f"Sensitivity/Recall: {ensemble_sensitivity:.4f}\n")
        f.write(f"Specificity: {ensemble_specificity:.4f}\n")
        f.write(f"PPV/Precision: {ensemble_ppv:.4f}\n")
        f.write(f"NPV: {ensemble_npv:.4f}\n\n")
        
        f.write("=== Informe de Clasificación ===\n")
        f.write(classification_report(y_test, y_pred_ensemble))
        
        f.write("\n=== Matriz de Confusión ===\n")
        f.write(f"{cm}\n")
    
    print(f"Informe de evaluación guardado en: {report_path}")
    
    # Generar gráficos
    
    # 1. Matriz de confusión
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title('Matriz de Confusión - Ensamble Optimizado')
    plt.tight_layout()
    cm_path = os.path.join(outdir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    # 2. Curva ROC
    plt.figure(figsize=(8, 6))
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob_ensemble)
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, color='blue', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], color='gray', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Curva ROC - Ensamble Optimizado')
    plt.legend(loc="lower right")
    plt.grid(True, linestyle='--', alpha=0.7)
    roc_path = os.path.join(outdir, "roc_curve.png")
    plt.savefig(roc_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    # 3. Calibración
    plt.figure(figsize=(10, 8))
    disp = CalibrationDisplay.from_predictions(
        y_test, y_pred_prob_ensemble, n_bins=10, name='Ensamble'
    )
    plt.title('Curva de Calibración - Ensamble Optimizado')
    plt.grid(True, linestyle='--', alpha=0.7)
    calibration_path = os.path.join(outdir, "calibration_curve.png")
    plt.savefig(calibration_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    # 4. Distribución de probabilidades
    plt.figure(figsize=(10, 6))
    class_0_probs = y_pred_prob_ensemble[y_test == 0]
    class_1_probs = y_pred_prob_ensemble[y_test == 1]
    
    plt.hist(class_0_probs, bins=20, alpha=0.5, label='Clase 0 (no-csPCa)', color='blue')
    plt.hist(class_1_probs, bins=20, alpha=0.5, label='Clase 1 (csPCa)', color='red')
    
    plt.axvline(x=0.5, color='gray', linestyle='--', label='Umbral (0.5)')
    plt.xlabel('Probabilidad de la clase positiva')
    plt.ylabel('Frecuencia')
    plt.title('Distribución de Probabilidades - Ensamble Optimizado')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    hist_path = os.path.join(outdir, "probability_distribution.png")
    plt.savefig(hist_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    
    return {
        'auc': ensemble_auc,
        'mcc': ensemble_mcc,
        'f1': ensemble_f1,
        'accuracy': ensemble_accuracy,
        'sensitivity': ensemble_sensitivity,
        'specificity': ensemble_specificity,
        'confusion_matrix': cm,
        'y_pred': y_pred_ensemble,
        'y_prob': y_pred_prob_ensemble
    }

# ============================================================
# Función para análisis de interpretabilidad con SHAP
# ============================================================
def interpretability_analysis(optimized_models, X_t2, X_adc, X_dwi, y, outdir, max_samples=100):
    """
    Realiza análisis de interpretabilidad usando SHAP para cada modelo en el ensamble optimizado.
    """
    print("\nRealizando análisis de interpretabilidad con SHAP...")
    os.makedirs(outdir, exist_ok=True)
    
    sequence_names = ['T2', 'ADC', 'DWI']
    model_keys = ['t2', 'adc', 'dwi']
    feature_sets = [X_t2, X_adc, X_dwi]
    
    for seq_name, model_key, X in zip(sequence_names, model_keys, feature_sets):
        seq_dir = os.path.join(outdir, seq_name)
        os.makedirs(seq_dir, exist_ok=True)
        
        model = optimized_models[model_key]
        
        # Extraer el modelo final del pipeline
        preprocessor = deepcopy(model)
        # Extraer el último paso (el clasificador)
        classifier = preprocessor.steps.pop()[1]
        
        # Aplicar el preprocesador a los datos
        X_processed = preprocessor.transform(X)
        
        # Para conjuntos de datos grandes, limitamos el número de muestras
        if len(X) > max_samples:
            indices = np.random.choice(len(X), max_samples, replace=False)
            X_sample = X.iloc[indices]
            X_processed_sample = X_processed[indices]
            y_sample = y[indices]
        else:
            X_sample = X
            X_processed_sample = X_processed
            y_sample = y
        
        # Crear y aplicar el explainer adecuado según el tipo de modelo
        try:
            if isinstance(classifier, RandomForestClassifier) or isinstance(classifier, GradientBoostingClassifier):
                # Para modelos basados en árboles, podemos usar TreeExplainer
                explainer = shap.TreeExplainer(classifier)
                feature_names = X.columns
                
                # Calcular valores SHAP
                shap_values = explainer.shap_values(X_processed_sample)
                
                # Para clasificación binaria, tomamos los valores de la clase positiva
                if isinstance(shap_values, list):
                    shap_values = shap_values[1]  # Valores para la clase positiva
                
                # Gráfico de resumen
                plt.figure(figsize=(12, 8))
                shap.summary_plot(shap_values, X_processed_sample, feature_names=feature_names, show=False)
                plt.title(f'SHAP Summary Plot - {seq_name}')
                summary_path = os.path.join(seq_dir, "shap_summary.png")
                plt.tight_layout()
                plt.savefig(summary_path, dpi=dpi, bbox_inches='tight')
                plt.close()
                
                # Gráfico de dependencia para las top 5 features
                mean_abs_shap = np.abs(shap_values).mean(axis=0)
                top_indices = np.argsort(mean_abs_shap)[-5:]
                
                for i, idx in enumerate(top_indices):
                    feature_name = feature_names[idx] if idx < len(feature_names) else f"Feature {idx}"
                    plt.figure(figsize=(10, 6))
                    shap.dependence_plot(idx, shap_values, X_processed_sample, feature_names=feature_names, show=False)
                    plt.title(f'SHAP Dependence Plot - {seq_name} - {feature_name}')
                    dep_path = os.path.join(seq_dir, f"shap_dependence_{i+1}_{feature_name.replace('/', '_')}.png")
                    plt.tight_layout()
                    plt.savefig(dep_path, dpi=dpi, bbox_inches='tight')
                    plt.close()
                
            elif isinstance(classifier, LogisticRegression):
                # Para regresión logística podemos usar LinearExplainer
                explainer = shap.LinearExplainer(classifier, X_processed_sample)
                feature_names = X.columns
                
                # Calcular valores SHAP
                shap_values = explainer.shap_values(X_processed_sample)
                
                # Gráfico de resumen
                plt.figure(figsize=(12, 8))
                shap.summary_plot(shap_values, X_processed_sample, feature_names=feature_names, show=False)
                plt.title(f'SHAP Summary Plot - {seq_name}')
                summary_path = os.path.join(seq_dir, "shap_summary.png")
                plt.tight_layout()
                plt.savefig(summary_path, dpi=dpi, bbox_inches='tight')
                plt.close()
                
                # Gráfico de coeficientes
                plt.figure(figsize=(12, 8))
                shap.plots.bar(shap.Explanation(
                    values=shap_values, 
                    base_values=explainer.expected_value,
                    data=X_processed_sample,
                    feature_names=feature_names
                ), show=False)
                plt.title(f'SHAP Coefficients - {seq_name}')
                coef_path = os.path.join(seq_dir, "shap_coefficients.png")
                plt.tight_layout()
                plt.savefig(coef_path, dpi=dpi, bbox_inches='tight')
                plt.close()
                
            else:
                # Para otros modelos, podemos usar KernelExplainer, pero es más lento
                # Tomamos una muestra más pequeña
                background = shap.kmeans(X_processed_sample, 10)
                
                if hasattr(classifier, "predict_proba"):
                    predict_fn = lambda x: classifier.predict_proba(x)[:, 1]
                else:
                    predict_fn = classifier.predict
                
                explainer = shap.KernelExplainer(predict_fn, background)
                feature_names = X.columns
                
                # Limitamos aún más para KernelExplainer
                n_explain = min(50, len(X_processed_sample))
                X_sample_small = X_processed_sample[:n_explain]
                
                # Calcular valores SHAP
                shap_values = explainer.shap_values(X_sample_small)
                
                # Gráfico de resumen
                plt.figure(figsize=(12, 8))
                shap.summary_plot(shap_values, X_sample_small, feature_names=feature_names, show=False)
                plt.title(f'SHAP Summary Plot - {seq_name}')
                summary_path = os.path.join(seq_dir, "shap_summary.png")
                plt.tight_layout()
                plt.savefig(summary_path, dpi=dpi, bbox_inches='tight')
                plt.close()
            
            print(f"  Análisis SHAP para {seq_name} completado.")
            
        except Exception as e:
            print(f"  Error en el análisis SHAP para {seq_name}: {e}")
            with open(os.path.join(seq_dir, "shap_error.txt"), "w") as f:
                f.write(f"Error en el análisis SHAP: {str(e)}")
    
    return True

# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Ensamble multisecuencia con todas las combinaciones de modelos")
    parser.add_argument("--region", type=str, default="gland",
                        choices=["gland", "full"],
                        help="Tipo de features: 'gland' o 'full'")
    parser.add_argument("--n_splits", type=int, default=2,  # 10
                        help="Número de folds para StratifiedGroupKFold")
    parser.add_argument("--n_repeats", type=int, default=1,  # 5 
                        help="Número de repeticiones de validación cruzada")
    parser.add_argument("--feature_strategy", type=str, default="most_discriminant",
                        choices=["all", "most_discriminant"],
                        help="Estrategia de selección de features")
    parser.add_argument("--optimize", action="store_true", default=True,
                        help="Realizar optimización bayesiana del mejor ensamble")
    args = parser.parse_args()
    
    # Configurar directorios de resultados
    base_results_dir = "results"
    os.makedirs(base_results_dir, exist_ok=True)
    
    # Ruta a los archivos CSV
    pre_path = "../../../../../data/radiomic_data/"
    
    t2_csv = f"features_t2_{args.region}.csv"
    adc_csv = f"features_adc_{args.region}.csv"
    dwi_csv = f"features_dwi_{args.region}.csv"
    
    t2_path = os.path.join(pre_path, t2_csv)
    adc_path = os.path.join(pre_path, adc_csv)
    dwi_path = os.path.join(pre_path, dwi_csv)
    
    print("\n==== Cargando datos ====")
    print(f"T2:  {t2_path}")
    print(f"ADC: {adc_path}")
    print(f"DWI: {dwi_path}")
    
    df_t2 = pd.read_csv(t2_path)
    df_adc = pd.read_csv(adc_path)
    df_dwi = pd.read_csv(dwi_path)
    
    # Crear IDs de paciente+estudio para asegurar correspondencia
    for df in [df_t2, df_adc, df_dwi]:
        df['patient_id_study_id'] = df['patient_id'].astype(str) + '_' + df['study_id'].astype(str)
        df.set_index('patient_id_study_id', inplace=True)
    
    # Asegurar que todos los dataframes tienen los mismos pacientes
    common_index = df_t2.index.intersection(df_adc.index).intersection(df_dwi.index)
    df_t2 = df_t2.loc[common_index]
    df_adc = df_adc.loc[common_index]
    df_dwi = df_dwi.loc[common_index]
    
    print(f"Número de pacientes en común: {len(common_index)}")
    
    # Obtener las variables de destino (y) y los grupos de pacientes
    y = df_t2["label"].values
    groups = df_t2["patient_id"].values
    
    # Eliminar columnas no deseadas
    X_t2 = df_t2.drop(columns=["patient_id", "study_id", "label", "mask_type"] + 
                     [col for col in df_t2.columns if col.startswith("diagnostics")])
    X_adc = df_adc.drop(columns=["patient_id", "study_id", "label", "mask_type"] + 
                       [col for col in df_adc.columns if col.startswith("diagnostics")])
    X_dwi = df_dwi.drop(columns=["patient_id", "study_id", "label", "mask_type"] + 
                       [col for col in df_dwi.columns if col.startswith("diagnostics")])
    
    # Codificar las etiquetas
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    # Crear un conjunto de prueba para la evaluación final
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    train_idx, test_idx = next(gss.split(X_t2, y, groups=groups))
    
    X_t2_train, X_t2_test = X_t2.iloc[train_idx], X_t2.iloc[test_idx]
    X_adc_train, X_adc_test = X_adc.iloc[train_idx], X_adc.iloc[test_idx]
    X_dwi_train, X_dwi_test = X_dwi.iloc[train_idx], X_dwi.iloc[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_train = groups[train_idx]
    
    print(f"Conjunto de entrenamiento: {len(X_t2_train)} muestras")
    print(f"Conjunto de prueba: {len(X_t2_test)} muestras")
    
    # Si se selecciona "most_discriminant", se realiza la selección con gráficos para cada secuencia
    if args.feature_strategy == "most_discriminant":
        print("\n==== Realizando selección de características ====")
        fs_dir_t2 = os.path.join(base_results_dir, "feature_selection", "T2")
        X_t2_train, selected_t2 = feature_selection_with_graphs(X_t2_train, y_train, fs_dir_t2, df_t2.iloc[train_idx], dpi=dpi)
        X_t2_test = X_t2_test[selected_t2]
        
        fs_dir_adc = os.path.join(base_results_dir, "feature_selection", "ADC")
        X_adc_train, selected_adc = feature_selection_with_graphs(X_adc_train, y_train, fs_dir_adc, df_adc.iloc[train_idx], dpi=dpi)
        X_adc_test = X_adc_test[selected_adc]
        
        fs_dir_dwi = os.path.join(base_results_dir, "feature_selection", "DWI")
        X_dwi_train, selected_dwi = feature_selection_with_graphs(X_dwi_train, y_train, fs_dir_dwi, df_dwi.iloc[train_idx], dpi=dpi)
        X_dwi_test = X_dwi_test[selected_dwi]
    else:
        print("\n>> Usando TODAS las características (sin selección).")
    
    # Guardar las variables usadas por secuencia
    with open(os.path.join(base_results_dir, "variables_T2.txt"), "w") as f:
        for feat in X_t2_train.columns:
            f.write(str(feat) + "\n")
    with open(os.path.join(base_results_dir, "variables_ADC.txt"), "w") as f:
        for feat in X_adc_train.columns:
            f.write(str(feat) + "\n")
    with open(os.path.join(base_results_dir, "variables_DWI.txt"), "w") as f:
        for feat in X_dwi_train.columns:
            f.write(str(feat) + "\n")
    
    # ====================================================
    # Probar todas las combinaciones de modelos
    # ====================================================
    all_models = get_models(random_state=42)
    model_combinations = list(itertools.product(all_models, repeat=3))
    print(f"\n==== Evaluando {len(model_combinations)} combinaciones de modelos ====")
    
    results_list = []  
    roc_data_all = []  # Para las curvas ROC
    
    for (name_t2, model_t2), (name_adc, model_adc), (name_dwi, model_dwi) in model_combinations:
        ensemble_name = f"{name_t2} + {name_adc} + {name_dwi}"
        print(f"Evaluando combinación: {ensemble_name}")
        
        fold_results, roc_data = run_cv_ensemble(model_t2, model_adc, model_dwi,
                                               X_t2_train, X_adc_train, X_dwi_train, y_train, groups_train,
                                               n_splits=args.n_splits, n_repeats=args.n_repeats,
                                               base_random_state=42)
        
        # Añadir nombre del ensemble a los datos ROC
        for data in roc_data:
            data["ensemble"] = ensemble_name
        roc_data_all.extend(roc_data)
        
        # Calcular métricas promedio (promediando sobre todos los folds)
        avg_train_auc = np.nanmean([fr["train_auc"] for fr in fold_results])
        avg_train_f1  = np.nanmean([fr["train_f1"]  for fr in fold_results])
        avg_val_auc   = np.nanmean([fr["val_auc"]   for fr in fold_results])
        avg_val_mcc   = np.nanmean([fr["val_mcc"]   for fr in fold_results])
        avg_val_kappa = np.nanmean([fr["val_kappa"] for fr in fold_results])
        avg_val_f1_binary = np.nanmean([fr["val_f1_binary"] for fr in fold_results])
        avg_val_f1_macro  = np.nanmean([fr["val_f1_macro"]  for fr in fold_results])
        avg_val_accuracy  = np.nanmean([fr["val_accuracy"]  for fr in fold_results])
        avg_val_bal_acc   = np.nanmean([fr["val_balanced_accuracy"] for fr in fold_results])
        avg_val_sens    = np.nanmean([fr["val_sensitivity"] for fr in fold_results])
        avg_val_spec    = np.nanmean([fr["val_specificity"] for fr in fold_results])
        avg_val_ppv     = np.nanmean([fr["val_ppv"] for fr in fold_results])
        avg_val_npv     = np.nanmean([fr["val_npv"] for fr in fold_results])
        
        # Calcular métricas per clase
        avg_per_class_precision = np.nanmean([np.array(fr["per_class_precision"]) for fr in fold_results], axis=0)
        avg_per_class_recall    = np.nanmean([np.array(fr["per_class_recall"]) for fr in fold_results], axis=0)
        avg_per_class_f1        = np.nanmean([np.array(fr["per_class_f1"]) for fr in fold_results], axis=0)
        avg_per_class_accuracy  = np.nanmean([np.array(fr["per_class_accuracy"]) for fr in fold_results], axis=0)
        
        results_list.append({
            "Ensemble": ensemble_name,
            "Folds": len(fold_results),
            "Avg_Train_AUC": avg_train_auc,
            "Avg_Train_F1": avg_train_f1,
            "Avg_Val_AUC": avg_val_auc,
            "Avg_Val_MCC": avg_val_mcc,
            "Avg_Val_Kappa": avg_val_kappa,
            "Avg_Val_F1_Binary": avg_val_f1_binary,
            "Avg_Val_F1_Macro": avg_val_f1_macro,
            "Avg_Val_Accuracy": avg_val_accuracy,
            "Avg_Val_BalancedAccuracy": avg_val_bal_acc,
            "Avg_Val_Sensitivity": avg_val_sens,
            "Avg_Val_Specificity": avg_val_spec,
            "Avg_Val_PPV": avg_val_ppv,
            "Avg_Val_NPV": avg_val_npv,
            "Avg_Per_Class_Precision": ", ".join([f"{v:.3f}" for v in avg_per_class_precision]),
            "Avg_Per_Class_Recall": ", ".join([f"{v:.3f}" for v in avg_per_class_recall]),
            "Avg_Per_Class_F1": ", ".join([f"{v:.3f}" for v in avg_per_class_f1]),
            "Avg_Per_Class_Accuracy": ", ".join([f"{v:.3f}" for v in avg_per_class_accuracy])
        })
    
    # Guardar el resumen promedio de cada combinación
    df_results = pd.DataFrame(results_list)
    df_results = df_results.sort_values(by="Avg_Val_AUC", ascending=False)
    results_csv_path = os.path.join(base_results_dir, "ensemble_results.csv")
    df_results.to_csv(results_csv_path, index=False)
    print(f"\n>> Resultados de combinaciones guardados en: {results_csv_path}")
    
    # ====================================================
    # Generar curvas ROC para los mejores ensambles
    # ====================================================
    roc_dir = os.path.join(base_results_dir, "roc_curves")
    os.makedirs(roc_dir, exist_ok=True)
    
    optimal_curves, median_curves = generate_roc_curves(df_results, roc_data_all, roc_dir, top_n=6)
    print(f"\n>> Curvas ROC generadas en: {roc_dir}")
    
    # ====================================================
    # Optimizar el mejor ensamble
    # ====================================================
    if args.optimize:
        best_ensemble = df_results.iloc[0]["Ensemble"]
        print(f"\n==== Optimizando el mejor ensamble: {best_ensemble} ====")
        
        opt_dir = os.path.join(base_results_dir, "optimized_ensemble")
        os.makedirs(opt_dir, exist_ok=True)
        
        # Extraer los nombres de los modelos
        model_names = best_ensemble.split(" + ")
        model_t2_name, model_adc_name, model_dwi_name = model_names
        
        # Optimizar cada modelo
        optimized_models = optimize_models(
            X_t2_train, X_adc_train, X_dwi_train, y_train, groups_train,
            best_ensemble, opt_dir, n_folds=5, random_state=42
        )
        
        # Evaluar el ensemble optimizado en el conjunto de prueba
        test_dir = os.path.join(opt_dir, "test_evaluation")
        os.makedirs(test_dir, exist_ok=True)
        
        test_results = evaluate_optimized_ensemble(
            optimized_models, X_t2_test, X_adc_test, X_dwi_test, y_test, test_dir
        )
        
        # Realizar análisis de interpretabilidad
        interpretability_dir = os.path.join(opt_dir, "interpretability")
        os.makedirs(interpretability_dir, exist_ok=True)
        
        interpretability_analysis(
            optimized_models, X_t2_train, X_adc_train, X_dwi_train, y_train, interpretability_dir
        )
        
        print("\n==== Análisis completado ====")
        print(f"Resultados guardados en: {base_results_dir}")
        print(f"Mejor ensamble: {best_ensemble}")
        print(f"Métricas en test AUC: {test_results['auc']:.4f}, MCC: {test_results['mcc']:.4f}, F1: {test_results['f1']:.4f}")
    
    else:
        print("\n==== Optimización desactivada ====")
        print("Para optimizar el mejor ensamble, use la opción --optimize")
        
    print("\nProceso completado con éxito.")

if __name__ == "__main__":
    main()
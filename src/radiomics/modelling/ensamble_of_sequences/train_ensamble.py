import argparse
import os
import itertools
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from scipy.stats import shapiro, mannwhitneyu, ttest_ind
from sklearn import metrics
from sklearn.model_selection import StratifiedGroupKFold
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
                             matthews_corrcoef, confusion_matrix)


import matplotlib as mpl
mpl.use('Agg')
import scienceplots

dpi = 300
plt.style.use(['science', 'grid'])


# ================================
# Función para definir los modelos
# ================================
def get_models(random_state=42):
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
    fold_results = []
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
    return fold_results

# ============================================================
# MAIN
# ============================================================
def main():
    parser = argparse.ArgumentParser(description="Ensamble multisecuencia con todas las combinaciones de modelos")
    parser.add_argument("--region", type=str, default="gland",
                        choices=["gland", "full"],
                        help="Tipo de features: 'gland' o 'full'")
    parser.add_argument("--n_splits", type=int, default=2, # 10
                        help="Número de folds para StratifiedGroupKFold")
    parser.add_argument("--n_repeats", type=int, default=1, # 5 
                        help="Número de repeticiones de validación cruzada")
    parser.add_argument("--feature_strategy", type=str, default="most_discriminant",
                        choices=["all", "most_discriminant"],
                        help="Estrategia de selección de features")
    args = parser.parse_args()
    
    pre_path = "../../../../../data/radiomic_data/"
    
    t2_csv = f"features_t2_{args.region}.csv"
    adc_csv = f"features_adc_{args.region}.csv"
    dwi_csv = f"features_dwi_{args.region}.csv"
    
    t2_path = os.path.join(pre_path, t2_csv)
    adc_path = os.path.join(pre_path, adc_csv)
    dwi_path = os.path.join(pre_path, dwi_csv)
    
    df_t2 = pd.read_csv(t2_path)
    df_adc = pd.read_csv(adc_path)
    df_dwi = pd.read_csv(dwi_path)
    
    for df in [df_t2, df_adc, df_dwi]:
        df['patient_id_study_id'] = df['patient_id'].astype(str) + '_' + df['study_id'].astype(str)
        df.set_index('patient_id_study_id', inplace=True)
    common_index = df_t2.index.intersection(df_adc.index).intersection(df_dwi.index)
    df_t2 = df_t2.loc[common_index]
    df_adc = df_adc.loc[common_index]
    df_dwi = df_dwi.loc[common_index]
    
    y = df_t2["label"].values
    groups = df_t2["patient_id"].values
    
    X_t2 = df_t2.drop(columns=["patient_id", "study_id", "label", "mask_type"] + [col for col in df_t2.columns if col.startswith("diagnostics")])
    X_adc = df_adc.drop(columns=["patient_id", "study_id", "label", "mask_type"] + [col for col in df_adc.columns if col.startswith("diagnostics")])
    X_dwi = df_dwi.drop(columns=["patient_id", "study_id", "label", "mask_type"] + [col for col in df_dwi.columns if col.startswith("diagnostics")])
    
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    base_results_dir = "results"
    os.makedirs(base_results_dir, exist_ok=True)
    
    # Si se selecciona "most_discriminant", se realiza la selección con gráficos para cada secuencia
    if args.feature_strategy == "most_discriminant":
        print(">> Realizando selección de características...")
        fs_dir_t2 = os.path.join(base_results_dir, "feature_selection", "T2")
        os.makedirs(fs_dir_t2, exist_ok=True)
        X_t2, selected_t2 = feature_selection_with_graphs(X_t2, y, fs_dir_t2, df_t2, dpi=dpi)
        
        fs_dir_adc = os.path.join(base_results_dir, "feature_selection", "ADC")
        os.makedirs(fs_dir_adc, exist_ok=True)
        X_adc, selected_adc = feature_selection_with_graphs(X_adc, y, fs_dir_adc, df_adc, dpi=dpi)
        
        fs_dir_dwi = os.path.join(base_results_dir, "feature_selection", "DWI")
        os.makedirs(fs_dir_dwi, exist_ok=True)
        X_dwi, selected_dwi = feature_selection_with_graphs(X_dwi, y, fs_dir_dwi, df_dwi, dpi=dpi)
    else:
        print(">> Usando TODAS las características (sin selección).")
    
    # Guardar las variables usadas por secuencia
    with open(os.path.join(base_results_dir, "variables_T2.txt"), "w") as f:
        for feat in X_t2.columns:
            f.write(str(feat) + "\n")
    with open(os.path.join(base_results_dir, "variables_ADC.txt"), "w") as f:
        for feat in X_adc.columns:
            f.write(str(feat) + "\n")
    with open(os.path.join(base_results_dir, "variables_DWI.txt"), "w") as f:
        for feat in X_dwi.columns:
            f.write(str(feat) + "\n")
    
    # ====================================================
    # Probar todas las combinaciones de modelos
    # ====================================================
    all_models = get_models(random_state=42)
    model_combinations = list(itertools.product(all_models, repeat=3))
    print(f">> Se evaluarán {len(model_combinations)} combinaciones de modelos.")
    
    results_list = []  
    folds_all = []     
    
    for (name_t2, model_t2), (name_adc, model_adc), (name_dwi, model_dwi) in model_combinations:
        ensemble_name = f"{name_t2} + {name_adc} + {name_dwi}"
        print(f"Evaluando combinación: {ensemble_name}")
        
        fold_results = run_cv_ensemble(model_t2, model_adc, model_dwi,
                                       X_t2, X_adc, X_dwi, y, groups,
                                       n_splits=args.n_splits, n_repeats=args.n_repeats,
                                       base_random_state=42)
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
        
        # Promediar métricas per clase
        avg_per_class_precision = np.nanmean(np.stack([np.array(fr["per_class_precision"]) for fr in fold_results]), axis=0)
        avg_per_class_recall    = np.nanmean(np.stack([np.array(fr["per_class_recall"]) for fr in fold_results]), axis=0)
        avg_per_class_f1        = np.nanmean(np.stack([np.array(fr["per_class_f1"]) for fr in fold_results]), axis=0)
        avg_per_class_accuracy  = np.nanmean(np.stack([np.array(fr["per_class_accuracy"]) for fr in fold_results]), axis=0)
        
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
        
        for fr in fold_results:
            fr["Ensemble"] = ensemble_name
            folds_all.append(fr)
    
    # Guardar el resumen promedio de cada combinación
    df_results = pd.DataFrame(results_list)
    df_results = df_results.sort_values(by="Avg_Val_AUC", ascending=False)
    results_csv_path = os.path.join(base_results_dir, "ensemble_results.csv")
    df_results.to_csv(results_csv_path, index=False)
    print(f">> Resultados de combinaciones guardados en: {results_csv_path}")
    
    # Guardar resultados de cada fold con TODAS las métricas
    df_folds = pd.DataFrame(folds_all)
    folds_csv_path = os.path.join(base_results_dir, "ensemble_folds.csv")
    df_folds.to_csv(folds_csv_path, index=False)
    print(f">> Resultados por fold guardados en: {folds_csv_path}")
    
if __name__ == "__main__":
    main()
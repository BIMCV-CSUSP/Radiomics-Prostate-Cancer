import argparse
import pandas as pd
import numpy as np
import os

from scipy.stats import shapiro, mannwhitneyu, ttest_ind
from statsmodels.stats.multitest import multipletests
from sklearn import metrics

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline

from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from sklearn.metrics import (roc_auc_score, accuracy_score, f1_score, precision_score,
                             recall_score, balanced_accuracy_score, cohen_kappa_score,
                             matthews_corrcoef, confusion_matrix)
from sklearn.feature_selection import VarianceThreshold

import matplotlib.pyplot as plt
import seaborn as sns

import matplotlib as mpl
mpl.use('Agg')
import scienceplots

plt.style.use(['science', 'grid'])
dpi = 300

def get_models(random_state=42):
    """
    Define los pipelines para cada clasificador (SIN GridSearch).
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

def evaluate_model(model, X, y, groups, n_splits=5, n_repeats=1, base_random_state=42):
    """
    Realiza validación cruzada repetida usando StratifiedGroupKFold en cada repetición.
    
    n_splits : int -> número de folds en cada repetición
    n_repeats: int -> cuántas veces repetamos la división en folds
    base_random_state: semilla base, se suma con el número de repetición.
    
    Devuelve:
    - fold_results: lista de métricas (dict) por fold global
    - pred_vals: dict con información (y_val, y_pred, y_val_prob, etc.) para cada fold global
    """
    fold_results = []
    folds_data = []  

    global_fold_index = 0
    for rep in range(n_repeats):
        # Cambiamos la semilla en cada repetición
        # para obtener divisiones distintas
        current_random_state = base_random_state + rep
        
        splitter = StratifiedGroupKFold(
            n_splits=n_splits, shuffle=True, random_state=current_random_state
        )
        
        for train_idx, val_idx in splitter.split(X, y, groups=groups):
            global_fold_index += 1
            
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            model.fit(X_train, y_train)
            
            # Métricas en entrenamiento
            y_train_pred = model.predict(X_train)
            if hasattr(model, "predict_proba"):
                y_train_prob = model.predict_proba(X_train)[:, 1]
            elif hasattr(model, "decision_function"):
                y_train_prob = model.decision_function(X_train)
            else:
                y_train_prob = None
            
            try:
                train_auc = roc_auc_score(y_train, y_train_prob) if y_train_prob is not None else np.nan
            except:
                train_auc = np.nan
            train_f1 = f1_score(y_train, y_train_pred, average="binary")
            
            # Métricas en validación
            y_val_pred = model.predict(X_val)
            
            if hasattr(model, "predict_proba"):
                y_val_prob = model.predict_proba(X_val)[:, 1]
            elif hasattr(model, "decision_function"):
                y_val_prob = model.decision_function(X_val)
            else:
                y_val_prob = None
            
            try:
                val_auc = roc_auc_score(y_val, y_val_prob) if y_val_prob is not None else np.nan
            except:
                val_auc = np.nan
            
            val_mcc = matthews_corrcoef(y_val, y_val_pred)
            val_kappa = cohen_kappa_score(y_val, y_val_pred)
            val_f1_binary = f1_score(y_val, y_val_pred, average="binary")
            val_f1_macro = f1_score(y_val, y_val_pred, average="macro")
            val_accuracy = accuracy_score(y_val, y_val_pred)
            val_balanced_accuracy = balanced_accuracy_score(y_val, y_val_pred)
            val_sensitivity = recall_score(y_val, y_val_pred, pos_label=1)
            val_specificity = recall_score(y_val, y_val_pred, pos_label=0)
            val_ppv = precision_score(y_val, y_val_pred, pos_label=1)
            
            cm = confusion_matrix(y_val, y_val_pred)
            if (cm[0, 0] + cm[1, 0]) > 0:
                val_npv = cm[0, 0] / (cm[0, 0] + cm[1, 0])
            else:
                val_npv = np.nan
            
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
            
            fold_metrics = {
                "Fold": global_fold_index,  
                "Repeat": rep + 1,          
                "train_auc": train_auc,
                "train_f1": train_f1,
                "val_auc": val_auc,
                "val_mcc": val_mcc,
                "val_kappa": val_kappa,
                "val_f1_binary": val_f1_binary,
                "val_f1_macro": val_f1_macro,
                "val_accuracy": val_accuracy,
                "val_sensitivity": val_sensitivity,
                "val_specificity": val_specificity,
                "val_ppv": val_ppv,
                "val_npv": val_npv,
                "val_balanced_accuracy": val_balanced_accuracy,
                "per_class_precision": per_class_precision.tolist(),
                "per_class_recall": per_class_recall.tolist(),
                "per_class_f1": per_class_f1.tolist(),
                "per_class_accuracy": per_class_accuracy
            }
            
            fold_results.append(fold_metrics)
    
            # Guardar la info de este fold para la curva ROC
            folds_data.append({
                "fold_index": global_fold_index,
                "Repeat": rep + 1,
                "y_val": y_val,
                "y_val_pred": y_val_pred,
                "y_val_prob": y_val_prob 
            })
            
    pred_vals = {
        "folds": folds_data
    }
    return fold_results, pred_vals

def main():
    parser = argparse.ArgumentParser(
        description="Evaluación de modelos con validación cruzada repetida"
    )
    parser.add_argument(
        "--csv", type=str,
        choices=["features_all_gland.csv", "features_all_full.csv"],
        default="features_all_gland.csv",
        help="Nombre del CSV con las características."
    )
    parser.add_argument(
        "--data_pre", type=str,
        default="../../../artifacts/radiomics",
        help="Directorio raíz donde se encuentran los datos radiomics."
    )
    parser.add_argument(
        "--results_base", type=str, default="../../../results/radiomics",
        help="Directorio base donde se crearán los resultados."
    )
    parser.add_argument(
        "--n_splits", type=int, default=10,
        help="Número de particiones para StratifiedGroupKFold (por repetición)."
    )
    parser.add_argument(
        "--n_repeats", type=int, default=5,
        help="Número de repeticiones de la validación cruzada."
    )
    parser.add_argument(
        "--feature_strategy", type=str,
        choices=["all", "most_discriminant"],
        default="most_discriminant",
        help="Estrategia de selección de features: 'all' o 'most_discriminant'."
    )
    parser.add_argument(
        "--calculate_differences", action="store_true", default=True,
        help="Si se habilita, ejecuta model_differences.py."
    )
    parser.add_argument(
        "--fine_tune_best_model", action="store_true", default=False,
        help="Si se habilita, realiza fine-tuning del mejor modelo."
    )

    args = parser.parse_args()

    # Ruta al CSV
    data_path = os.path.join(
        args.data_pre,
        "concatenated_data",
        args.csv
    )
    df = pd.read_csv(data_path)
    df['patient_id_study_id'] = df['patient_id'].astype(str) + '_' + df['study_id'].astype(str)
    df = df.set_index('patient_id_study_id')

    # Variables para CV
    y = df['label'].values
    groups = df['patient_id'].values
    X = df.drop(columns=['patient_id', 'study_id', 'label'])

    le = LabelEncoder()
    y = le.fit_transform(y)

    # Estructura de carpetas de resultados
    base_dir = args.results_base
    strat_dir = os.path.join(base_dir, args.feature_strategy)
    os.makedirs(strat_dir, exist_ok=True)

    csv_stem = os.path.splitext(args.csv)[0]
    mode = csv_stem.rsplit("_", 1)[-1]
    experiment_dir = os.path.join(strat_dir, mode)
    os.makedirs(experiment_dir, exist_ok=True)

    print(f"Creada carpeta de resultados: {experiment_dir}")
    # =====================================================================
    
    selected_features = X.columns
    
    # === Selección de características ===
    if args.feature_strategy == "most_discriminant":
        print(">> Realizando selección de características...")
        
        fs_dir = os.path.join(experiment_dir, "feature_selection")
        os.mkdir(fs_dir)
        
        images_dir = os.path.join(fs_dir, "images")
        os.mkdir(images_dir)
        
        feature_names, sensitivity_list, specificity_list = ([] for _ in range(3))
        auc_list, threshold_list, test_type_list, pvalue_list, pos_vs_neg_list = ([] for _ in range(5))
        
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
        
        num_features_model = round(X.shape[0] / 15)
        train_df = train_auc_pvals_df.sort_values(by='p-value', ascending=True)
        
        selected_features = train_df.index[0:num_features_model]
        print(f"  --> Seleccionadas {len(selected_features)} características más relevantes.")
        
        X = X[selected_features]
        
        df_path_1 = os.path.join(fs_dir, "train_auc_pvals_df.csv")
        train_auc_pvals_df.to_csv(df_path_1)
        print(f"  --> Guardado CSV: {df_path_1}\n")
        
        # ====== Generamos imágenes para las TOP 20 features ======
        top_20 = train_auc_pvals_df.index[:20]
        
        for rank, feature_name in enumerate(top_20, start=1):
            safe_feat_name = feature_name.replace("/", "_")
            feat_folder_name = f"{rank}_{safe_feat_name}"
            feat_folder_path = os.path.join(images_dir, feat_folder_name)
            os.mkdir(feat_folder_path)
            
            # Violinplot
            plt.figure(figsize=(9, 9))
            sns.violinplot(x=y, y=df[feature_name], color='grey')
            plt.title(f"Distribución de {feature_name} en no-csPCa vs csPCa", fontsize=14)
            plt.xlabel("Clases")
            plt.xticks([0, 1], ["no-csPCa", "csPCa"], fontsize=12)
            violin_plot_path = os.path.join(feat_folder_path, f"{safe_feat_name}_violinplot.png")
            plt.savefig(violin_plot_path, dpi=dpi)
            plt.close()
            
            # ROC curve
            fpr, tpr, _ = metrics.roc_curve(y, df[feature_name], pos_label=1)
            auc_val = metrics.auc(fpr, tpr)
            
            plt.figure(figsize=(6, 6))
            plt.plot(fpr, tpr, marker='.', color='black', markersize=3, label=f"{feature_name} (AUC={auc_val:.3f})")
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.legend()
            plt.title(f"Curva ROC: Capacidad discriminativa de {feature_name}")
            roc_plot_path = os.path.join(feat_folder_path, f"{safe_feat_name}_ROC.png")
            plt.savefig(roc_plot_path, dpi=dpi)
            plt.close()
    else:
        print(">> Usando TODAS las características (sin selección).")
    
    # == Definimos y evaluamos los modelos ==
    models = get_models(random_state=42)
    
    all_results = []
    preds_data = []
    
    # Aquí llamamos a evaluate_model con n_repeats
    for model_name, model in models:
        print(f"Evaluando {model_name}...")
        fold_metrics_list, pred_vals = evaluate_model(
            model, X, y, groups, 
            n_splits=args.n_splits, 
            n_repeats=args.n_repeats,
            base_random_state=42
        )
        
        for fold_metrics in fold_metrics_list:
            fold_metrics["Classifier"] = model_name
            all_results.append(fold_metrics)
        
        preds_data.append({
            "Classifier": model_name,
            "folds": pred_vals["folds"]
        })
    
    df_resultados = pd.DataFrame(all_results)

    fixed_cols = ["Classifier", "Fold", "Repeat"]
    other_cols = [c for c in df_resultados.columns if c not in fixed_cols]
    df_resultados = df_resultados[fixed_cols + other_cols]
    df_resultados.sort_values(by=["Classifier", "Fold"], inplace=True)
    
    csv_basename = os.path.splitext(args.csv)[0]
    feat_str = args.feature_strategy
    resultados_filename = f"resultados_{csv_basename}_{feat_str}.csv"
    
    resultados_filepath = os.path.join(experiment_dir, resultados_filename)
    df_resultados.to_csv(resultados_filepath, index=False)
    print(f"\nResultados guardados en '{resultados_filepath}'")
    
    records_for_csv = []
    for item in preds_data:
        clf_name = item["Classifier"]
        folds_info = item["folds"]
        for fold_info in folds_info:
            fold_idx = fold_info["fold_index"]
            rep_idx = fold_info["Repeat"]
            
            y_val_list = fold_info["y_val"].tolist()
            y_pred_list = fold_info["y_val_pred"].tolist()
            if fold_info["y_val_prob"] is not None:
                y_prob_list = fold_info["y_val_prob"].tolist()
            else:
                y_prob_list = []
            
            records_for_csv.append({
                "Classifier": clf_name,
                "Fold": fold_idx,
                "Repeat": rep_idx,
                "y_val": y_val_list,
                "y_pred": y_pred_list,
                "y_prob": y_prob_list
            })

    df_preds = pd.DataFrame(records_for_csv)
    preds_filename = f"preds_{csv_basename}_{feat_str}.csv"
    preds_filepath = os.path.join(experiment_dir, preds_filename)
    df_preds.to_csv(preds_filepath, index=False)
    print(f"Predicciones guardadas en '{preds_filepath}'")
    
    # === Guardar txt con las variables usadas ===
    variables_txt_path = os.path.join(experiment_dir, "variables_usadas.txt")
    with open(variables_txt_path, "w") as f:
        for feat in selected_features:
            f.write(str(feat) + "\n")
    print(f"Archivo con variables usadas: {variables_txt_path}")


    # <--- Curvas ROC: Generamos dos archivos, uno para el fold óptimo y otro para el fold mediano por clasificador
    print("\nGenerando curvas ROC: fold óptimo y mediano por clasificador...")
    
    roc_dir = os.path.join(experiment_dir, "ROC_curves")
    os.makedirs(roc_dir, exist_ok=True)
    
    curves_info_optimal = []
    curves_info_median = []
    
    classifiers = df_resultados["Classifier"].unique()
    for clf_name in classifiers:
        df_clf = df_resultados[df_resultados["Classifier"] == clf_name]
        
        # Fold óptimo
        best_fold_idx = df_clf["val_auc"].idxmax()
        best_fold_num = df_clf.loc[best_fold_idx, "Fold"]
        
        # Fold mediano
        median_auc = df_clf["val_auc"].median()
        median_fold_idx = (df_clf["val_auc"] - median_auc).abs().idxmin()
        median_fold_num = df_clf.loc[median_fold_idx, "Fold"]
        
        # Predicciones óptimas
        df_clf_preds_best = df_preds[
            (df_preds["Classifier"] == clf_name) & 
            (df_preds["Fold"] == best_fold_num)
        ]
        if len(df_clf_preds_best) > 0:
            y_val_list_best = df_clf_preds_best.iloc[0]["y_val"]
            y_prob_list_best = df_clf_preds_best.iloc[0]["y_prob"]
            if y_prob_list_best:
                fpr_best, tpr_best, _ = metrics.roc_curve(y_val_list_best, y_prob_list_best, pos_label=1)
                auc_val_best = metrics.auc(fpr_best, tpr_best)
                curves_info_optimal.append({
                    "classifier": clf_name,
                    "fold": best_fold_num,
                    "fpr": fpr_best,
                    "tpr": tpr_best,
                    "auc": auc_val_best
                })
        
        # Predicciones medianas
        df_clf_preds_median = df_preds[
            (df_preds["Classifier"] == clf_name) & 
            (df_preds["Fold"] == median_fold_num)
        ]
        if len(df_clf_preds_median) > 0:
            y_val_list_median = df_clf_preds_median.iloc[0]["y_val"]
            y_prob_list_median = df_clf_preds_median.iloc[0]["y_prob"]
            if y_prob_list_median:
                fpr_median, tpr_median, _ = metrics.roc_curve(y_val_list_median, y_prob_list_median, pos_label=1)
                auc_val_median = metrics.auc(fpr_median, tpr_median)
                curves_info_median.append({
                    "classifier": clf_name,
                    "fold": median_fold_num,
                    "fpr": fpr_median,
                    "tpr": tpr_median,
                    "auc": auc_val_median
                })
    
    # Ordenamos las curvas de cada tipo por AUC descendente
    curves_info_optimal.sort(key=lambda x: x["auc"], reverse=True)
    curves_info_median.sort(key=lambda x: x["auc"], reverse=True)
    
    my_colors = [
        "#0072B2",  # Azul oscuro
        "#009E73",  # Verde
        "#D55E00",  # Naranja rojizo
        "#CC78BC",  # Morado
        "#DE8F05",  # Marrón/naranja
        "#56B4E9"   # Azul claro
    ]
    
    my_palette = sns.color_palette(my_colors)
    
    fixed_classifiers = ["SVM", "Logistic Regression", "Random Forest", 
                         "Naive Bayes", "KNN", "Gradient Boosting"]
    color_mapping = {clf: my_palette[i] for i, clf in enumerate(fixed_classifiers)}
    
    # --- ROC óptimo ---
    fig_opt, ax_opt = plt.subplots(figsize=(8, 6))
    for info in curves_info_optimal:
        clf_name = info["classifier"]
        fold_num = info["fold"]
        fpr = info["fpr"]
        tpr = info["tpr"]
        auc_val = info["auc"]
        ax_opt.plot(fpr, tpr, label=f"{clf_name} (Fold={fold_num}, AUC={auc_val:.3f})", 
                    color=color_mapping[clf_name])
    
    ax_opt.plot([0, 1], [0, 1], linestyle='--', color='gray', label="_nolegend_")
    ax_opt.set_xlabel("False Positive Rate", fontsize=12, labelpad=10)
    ax_opt.set_ylabel("True Positive Rate", fontsize=12, labelpad=10)
    ax_opt.set_title("Curvas ROC: Folds óptimos por modelo", fontsize=14)
    ax_opt.tick_params(axis='both', which='major', labelsize=10)
    ax_opt.legend(fontsize=10)
    leg = ax_opt.get_legend()
    for line in leg.get_lines():
        line.set_linewidth(2.5)
    fig_opt.tight_layout()
    
    roc_plot_path_opt = os.path.join(roc_dir, "roc_optimal_folds.png")
    plt.savefig(roc_plot_path_opt, dpi=dpi, bbox_inches='tight')
    plt.close(fig_opt)
    print(f"Gráfico ROC (fold óptimo) guardado en: {roc_plot_path_opt}")
    
    # --- ROC mediano ---
    fig_med, ax_med = plt.subplots(figsize=(8, 6))
    for info in curves_info_median:
        clf_name = info["classifier"]
        fold_num = info["fold"]
        fpr = info["fpr"]
        tpr = info["tpr"]
        auc_val = info["auc"]
        ax_med.plot(fpr, tpr, label=f"{clf_name} (Fold={fold_num}, AUC={auc_val:.3f})", 
                    color=color_mapping[clf_name])
    
    ax_med.plot([0, 1], [0, 1], linestyle='--', color='gray', label="_nolegend_")
    ax_med.set_xlabel("False Positive Rate", fontsize=12, labelpad=10)
    ax_med.set_ylabel("True Positive Rate", fontsize=12, labelpad=10)
    ax_med.set_title("Curvas ROC: Folds medianos por modelo", fontsize=14)
    ax_med.tick_params(axis='both', which='major', labelsize=10)
    ax_med.legend(fontsize=10)
    leg = ax_med.get_legend()
    for line in leg.get_lines():
        line.set_linewidth(2.5)
    fig_med.tight_layout()
    
    roc_plot_path_med = os.path.join(roc_dir, "roc_median_folds.png")
    plt.savefig(roc_plot_path_med, dpi=dpi, bbox_inches='tight')
    plt.close(fig_med)
    print(f"Gráfico ROC (fold mediano) guardado en: {roc_plot_path_med}")

    
    # <--- Si se habilita calculate_differences, ejecutar model_differences
    if args.calculate_differences:
        print("\nEjecutando comparaciones de modelos (model_differences.py)...")
        import subprocess

        model_diff_dir = os.path.join(experiment_dir, "model_differences")
        os.mkdir(model_diff_dir)

        postprocess_cmd = [
            "python3",
            "2_model_differences.py",
            "--csv_preds", preds_filepath,
            "--csv_results", resultados_filepath,
            "--metric", "val_auc",
            "--alpha", "0.05",
            "--outdir", model_diff_dir
        ]
        
        subprocess.call(postprocess_cmd)
    else:
        print("\nOmitiendo cálculo de diferencias (model_differences.py)...")

    # <--- Si se habilita fine_tune_best_model, ejecutar retrain_best_model_and_evaluate
    if args.fine_tune_best_model:
        if len(curves_info_optimal) > 0:
            best_model = curves_info_optimal[0]["classifier"]

            model_mapping = {
                "SVM": "SVM",
                "Logistic Regression": "LogisticRegression",
                "Random Forest": "RandomForest",
                "Naive Bayes": "NaiveBayes",
                "KNN": "KNN",
                "Gradient Boosting": "GradientBoosting"
            }
            best_model_finetune = model_mapping.get(best_model, best_model)
            
            print(f"Fine-tuning del mejor modelo: {best_model_finetune}")
            
            fine_tune_cmd = [
                "python3",
                "3_retrain_best_model_and_evaluate.py",
                "--csv", args.csv,                       
                "--model", best_model_finetune,           
                "--variables", variables_txt_path          
            ]
            
            subprocess.call(fine_tune_cmd)
        else:
            print("No se encontró información de curvas óptimas para determinar el mejor modelo.")
    else:
        print("Omitiendo fine-tuning del mejor modelo.")

if __name__ == "__main__":
    main()
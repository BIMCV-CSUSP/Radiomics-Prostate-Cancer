#!/usr/bin/env python
import argparse
import json
import importlib
import logging
import os

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (roc_auc_score, f1_score, cohen_kappa_score, accuracy_score,
                             balanced_accuracy_score, recall_score, precision_score, matthews_corrcoef,
                             confusion_matrix)

from monai.data import Dataset, DataLoader

import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')

def dynamic_import(class_path):
    """Importa dinámicamente una clase dado su path completo."""
    module_name, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def setup_logger(log_file):
    """Configura un logger que sobrescribe el fichero de log en cada ejecución."""
    logger = logging.getLogger("train_logger")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    if logger.hasHandlers():
        logger.handlers.clear()

    fh = logging.FileHandler(log_file, mode='w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    return logger


def main():
    parser = argparse.ArgumentParser(description="Entrena un modelo según una configuración especificada.")
    parser.add_argument("--config_key", type=str, required=True,
                        help="Clave de configuración definida en el fichero JSON.")
    parser.add_argument("--config_file", type=str, default="config.json",
                        help="Ruta al fichero JSON de configuración.")
    
    parser.add_argument("--mode", type=str, choices=["full", "gland"], required=True,
                        help="Modo de carga de datos: 'full' para imagen completa o 'gland' para ROI de glándula.")
    
    parser.add_argument("--csv_path", type=str, default="../../../artifacts/data.csv",
                        help="Ruta al CSV de datos.")
    parser.add_argument("--input_shape", type=int, nargs=3, default=[128, 128, 32],
                        help="Dimensiones de la imagen de entrada.")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Número de épocas de entrenamiento.")
    parser.add_argument("--n_splits", type=int, default=2,
                        help="Número de splits para validación cruzada.")
    
    args = parser.parse_args()
    
    with open(args.config_file, "r") as f:
        configs = json.load(f)
    if args.config_key not in configs:
        raise ValueError(f"La clave {args.config_key} no se encuentra en el fichero de configuración.")
    config = configs[args.config_key]

    if args.mode == "full":
        loader_module = "data_loaders.data_loader_for_cv_org"
    else:
        loader_module = "data_loaders.data_loader_for_cv_roi"

    MyDataLoader = dynamic_import(f"{loader_module}.MyDataLoader")
    
    current_file = os.path.abspath(__file__)
    project_root = os.path.abspath(
        os.path.join(current_file,
                    os.pardir,
                    os.pardir,   
                    os.pardir,  
                    os.pardir)  
    )
    base_dir = os.path.join(project_root, "artifacts", "deep_learning", args.mode)

    logs_dir = os.path.join(base_dir, "logs")
    models_dir = os.path.join(base_dir, "models", args.config_key)
    results_dir = os.path.join(base_dir, "results", args.config_key)

    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    log_file = os.path.join(logs_dir, f"{args.config_key}.log")
    logger = setup_logger(log_file)
    logger.info(f"Configuración cargada: {config}")
    
    csv_path = args.csv_path
    input_shape = tuple(args.input_shape)
    epochs = args.epochs
    n_splits = args.n_splits
    
    extra_transforms = []
    extra_transforms_list = config.get("extra_transforms", [])
    if extra_transforms_list:
        for transform_item in extra_transforms_list:
            try:
                if isinstance(transform_item, dict):
                    transform_class_str = transform_item.get("class")
                    transform_args = transform_item.get("args", {})
                else:
                    transform_class_str = transform_item
                    transform_args = config.get("extra_transform_args", {})
                TransformClass = dynamic_import(transform_class_str)
                extra_transforms.append(TransformClass(**transform_args))
                logger.info(f"Transformación extra añadida: {transform_class_str} con argumentos {transform_args}")
            except Exception as e:
                logger.error(f"Error al importar la transformación {transform_class_str}: {e}")
                raise e
    else:
        logger.info("No se han definido transformaciones extra.")
    
    data_loader = MyDataLoader(
        csv_path=csv_path,
        input_shape=input_shape,
        config={"batch_size": 2, "num_workers": 4},
        transformations=extra_transforms,
        num_classes=config.get("model_args", {}).get("num_classes", 2)
    )
    
    all_data = data_loader.get_all_data()
    all_labels = [int(torch.argmax(item["label"]).item()) for item in all_data]
    patient_ids = [item["patient_id"] for item in all_data]
    
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    best_overall_model = None
    best_overall_score = -np.inf
    best_split_info = (None, None)
    
    for split_index, (train_idx, val_idx) in enumerate(splitter.split(all_data, all_labels, groups=patient_ids), start=1):
        logger.info(f"=== Split {split_index}/{n_splits} ===")
        train_subset = [all_data[i] for i in train_idx]
        val_subset = [all_data[i] for i in val_idx]
        
        num_classes = config.get("model_args", {}).get("num_classes", 2)
        class_counts = torch.zeros(num_classes, dtype=torch.long)
        for item in train_subset:
            cls_idx = torch.argmax(item["label"]).item()
            class_counts[cls_idx] += 1
        train_len = len(train_subset)
        weights_list = [
            train_len / (class_counts[c].item() if class_counts[c] > 0 else 1e-6)
            for c in range(num_classes)
        ]
        weights_tensor = torch.tensor(weights_list, dtype=torch.float32)
        class_weights = weights_tensor / weights_tensor.sum()
        logger.info(f"Pesos de clase: {class_weights.tolist()}")
        
        train_dataset = Dataset(data=train_subset, transform=data_loader.get_transforms(augment=True))
        val_dataset = Dataset(data=val_subset, transform=data_loader.get_transforms(augment=False))
        train_loader = DataLoader(train_dataset, batch_size=2, num_workers=4, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=2, num_workers=4, shuffle=False)
        
        try:
            ModelClass = dynamic_import(config["model"])
            model = ModelClass(**config.get("model_args", {}))
            logger.info(f"Modelo instanciado: {config['model']} con argumentos {config.get('model_args', {})}")
        except Exception as e:
            logger.error(f"Error al importar el modelo {config['model']}: {e}")
            raise e
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        optimizer = optim.Adam(model.parameters(), lr=1e-5)
        
        best_split_model_state = None
        best_split_val_auc = -np.inf  
        best_split_epoch = 0
        split_results = []

        patience = 10
        no_improve_count = 0
        
        for epoch in range(1, epochs + 1):
            # --- Entrenamiento ---
            model.train()
            train_loss_accum = 0.0
            preds_list = []
            labels_list = []
            train_probs = []  
            for batch in train_loader:
                inputs = batch["image"].to(device)
                label_hot = batch["label"].to(device)
                label_cls = torch.argmax(label_hot, dim=1)
                optimizer.zero_grad()
                outputs = model(inputs)
                if isinstance(outputs, (tuple, list)):
                    outputs = outputs[0]
                loss = criterion(outputs, label_cls)
                loss.backward()
                optimizer.step()
                train_loss_accum += loss.item()
                preds_list.append(torch.argmax(outputs, dim=1).cpu())
                labels_list.append(label_cls.cpu())
                probs = torch.softmax(outputs, dim=1)
                train_probs.append(probs[:, 1].detach().cpu())
            train_loss = train_loss_accum / len(train_loader)
            train_labels_np = torch.cat(labels_list).numpy()
            train_preds_np = torch.cat(preds_list).numpy()
            train_probs_np = torch.cat(train_probs).numpy()
            try:
                train_auc = roc_auc_score(train_labels_np, train_probs_np)
            except Exception as e:
                logger.error(f"Error calculando AUC en train: {e}")
                train_auc = np.nan
            train_f1 = f1_score(train_labels_np, train_preds_np, average='binary')
            
            # --- Validación ---
            model.eval()
            val_loss_accum = 0.0
            val_preds = []
            val_labels = []
            val_probs = []  
            with torch.no_grad():
                for batch in val_loader:
                    inputs = batch["image"].to(device)
                    label_hot = batch["label"].to(device)
                    label_cls = torch.argmax(label_hot, dim=1)
                    outputs = model(inputs)
                    if isinstance(outputs, (tuple, list)):
                        outputs = outputs[0]
                    loss = criterion(outputs, label_cls)
                    val_loss_accum += loss.item()
                    probs = torch.softmax(outputs, dim=1)
                    val_probs.append(probs[:, 1].cpu())
                    val_preds.append(torch.argmax(outputs, dim=1).cpu())
                    val_labels.append(label_cls.cpu())
            val_loss = val_loss_accum / len(val_loader)
            val_labels_np = torch.cat(val_labels).numpy()
            val_preds_np = torch.cat(val_preds).numpy()
            val_probs_np = torch.cat(val_probs).numpy()
            try:
                val_auc = roc_auc_score(val_labels_np, val_probs_np)
            except Exception as e:
                logger.error(f"Error calculando AUC en validación: {e}")
                val_auc = np.nan
                
            val_mcc = matthews_corrcoef(val_labels_np, val_preds_np)
            val_kappa = cohen_kappa_score(val_labels_np, val_preds_np)
            val_f1_binary = f1_score(val_labels_np, val_preds_np, average='binary')
            val_f1_macro = f1_score(val_labels_np, val_preds_np, average='macro')
            val_accuracy = accuracy_score(val_labels_np, val_preds_np)
            val_sensitivity = recall_score(val_labels_np, val_preds_np, pos_label=1)
            tn, fp, fn, tp = confusion_matrix(val_labels_np, val_preds_np).ravel()
            val_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            val_ppv = precision_score(val_labels_np, val_preds_np, pos_label=1)
            val_npv = tn / (tn + fn) if (tn + fn) > 0 else 0
            val_balanced_accuracy = balanced_accuracy_score(val_labels_np, val_preds_np)
            
            # Cálculo de métricas por clase
            per_class_precision = precision_score(val_labels_np, val_preds_np, average=None)
            per_class_recall = recall_score(val_labels_np, val_preds_np, average=None)
            per_class_f1 = f1_score(val_labels_np, val_preds_np, average=None)
            cm = confusion_matrix(val_labels_np, val_preds_np)
            per_class_accuracy = (cm.diagonal() / cm.sum(axis=1)).tolist()
            
            # Registro en logger
            logger.info(
                f"Split {split_index}, Epoch [{epoch}/{epochs}] | "
                f"Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}, Train F1: {train_f1:.4f} || "
                f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}, Val MCC: {val_mcc:.4f}, Val Kappa: {val_kappa:.4f}, "
                f"Val F1 (binary): {val_f1_binary:.4f}, Val F1 (macro): {val_f1_macro:.4f}, Val Accuracy: {val_accuracy:.4f}, "
                f"Val Sensitivity: {val_sensitivity:.4f}, Val Specificity: {val_specificity:.4f}, "
                f"Val PPV: {val_ppv:.4f}, Val NPV: {val_npv:.4f}, Val BalancedAcc: {val_balanced_accuracy:.4f}"
            )
            
            # Guardamos todas las métricas en el diccionario de resultados (incluyendo las métricas por clase)
            split_results.append({
                "split": split_index,
                "epoch": epoch,
                "train_loss": train_loss,
                "train_auc": train_auc,
                "train_f1": train_f1,
                "val_loss": val_loss,
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
            })
            
            # Early stopping basado en el AUC en validación
            if val_auc > best_split_val_auc:
                best_split_val_auc = val_auc
                best_split_epoch = epoch
                best_split_model_state = model.state_dict()
                no_improve_count = 0
            else:
                no_improve_count += 1
                
            if no_improve_count >= patience:
                logger.info(f"Early stopping en el split {split_index} en la epoch {epoch} por falta de mejora en AUC.")
                break
        
        results_csv_path = os.path.join(results_dir, f"split_{split_index}_results.csv")
        pd.DataFrame(split_results).to_csv(results_csv_path, index=False)
        logger.info(f"Resultados del split {split_index} guardados en {results_csv_path}")
        
        model_save_path = os.path.join(models_dir, f"best_model_split_{split_index}.pth")
        torch.save(best_split_model_state, model_save_path)
        logger.info(f"Mejor modelo del split {split_index} guardado en {model_save_path}")
        
        if best_split_val_auc > best_overall_score:
            best_overall_score = best_split_val_auc
            best_split_info = (split_index, best_split_epoch)
            best_overall_model = best_split_model_state
    
    overall_model_path = os.path.join(models_dir, "best_overall_model.pth")
    torch.save(best_overall_model, overall_model_path)
    logger.info(f"Mejor modelo global (split {best_split_info[0]}, epoch {best_split_info[1]} con Val AUC: {best_overall_score:.4f}) guardado en {overall_model_path}")


if __name__ == "__main__":
    main()
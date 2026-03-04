#!/usr/bin/env python
"""
Script para entrenamiento de modelos de deep learning.

Este script implementa un pipeline completo para entrenamiento y evaluación de modelos
de redes neuronales con validación cruzada estratificada por paciente. Características principales:

- Carga dinámica de configuraciones desde un archivo JSON
- Soporte para dos modos de análisis: imágenes completas ('full') o regiones de interés ('gland')
- Validación cruzada estratificada por paciente para evitar contaminación de datos
- Ponderación de clases para manejar desbalance
- Sistema completo de registro (logging)
- Métricas de evaluación
- Early stopping para evitar sobreajuste
- Guardado de modelos y resultados
"""

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

from monai.data import Dataset, DataLoader, CacheDataset, PersistentDataset
from tqdm import tqdm

import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')

def dynamic_import(class_path):
    """
    Importa dinámicamente una clase dado su path completo.
    
    Esta función permite cargar clases (modelos, transformaciones) especificadas
    en la configuración JSON sin necesidad de importarlas explícitamente en el código.
    
    Args:
        class_path (str): Ruta completa a la clase en formato 'module.submodule.ClassName'
        
    Returns:
        class: La clase importada (no una instancia)
    """
    
    module_name, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def setup_logger(log_file):
    """
    Configura un logger que sobrescribe el fichero de log en cada ejecución.
    
    Configura un sistema de logging que envía mensajes tanto a la consola como
    a un archivo, sobrescribiendo el archivo en cada ejecución.
    
    Args:
        log_file (str): Ruta al archivo de log
        
    Returns:
        logger: Objeto logger configurado
    """
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
    """
    Función principal que coordina todo el proceso de entrenamiento y evaluación.
    
    Flujo de trabajo:
    1. Procesa argumentos de línea de comandos
    2. Carga configuración del modelo desde JSON
    3. Configura directorios y sistema de logging
    4. Prepara cargador de datos y transformaciones
    5. Ejecuta validación cruzada estratificada por paciente
    6. Para cada split entrena el modelo y evalúa métricas
    7. Guarda resultados, modelos y selecciona el mejor modelo global
    """
    
    # ================== Configuración inicial y argumentos ==================    
    parser = argparse.ArgumentParser(description="Entrena un modelo según una configuración especificada.")
    parser.add_argument("--config_key", type=str, required=True,
                        help="Clave de configuración definida en el fichero JSON.")
    parser.add_argument("--config_file", type=str, default="config.json",
                        help="Ruta al fichero JSON de configuración.")
    
    parser.add_argument("--mode", type=str, choices=["full", "gland"], required=True,
                        help="Modo de carga de datos: 'full' para imagen completa o 'gland' para ROI de glándula.")
    
    parser.add_argument("--csv_path", type=str, default="../../../artifacts/data.csv",
                        help="Ruta al CSV de datos.")
    parser.add_argument("--input_shape", type=int, nargs=3, default=[256, 256, 32],
                        help="Dimensiones de la imagen de entrada.")
    parser.add_argument("--epochs", type=int, default=1,
                        help="Número de épocas de entrenamiento.")
    parser.add_argument("--n_splits", type=int, default=2,
                        help="Número de splits para validación cruzada.")
    
    args = parser.parse_args()
    
    # ================== Carga de configuración ==================

    # Cargamos el archivo de configuración JSON
    with open(args.config_file, "r") as f:
        configs = json.load(f)

    # Verificamos que la clave especificada existe en el archivo
    if args.config_key not in configs:
        raise ValueError(f"La clave {args.config_key} no se encuentra en el fichero de configuración.")
    
    # Extraemos la configuración específica
    config = configs[args.config_key]

    
    # ================== Preparación de módulos y directorios ==================
    
    # Seleccionamos el módulo de carga de datos según el modo elegido
    if args.mode == "full":
        loader_module = "data_loaders.data_loader_for_cv_org"
    else:
        loader_module = "data_loaders.data_loader_for_cv_roi"

    # Importamos dinámicamente la clase de carga de datos
    MyDataLoader = dynamic_import(f"{loader_module}.MyDataLoader")
    
    # Determinamos la ruta absoluta al directorio raíz del proyecto
    current_file = os.path.abspath(__file__)
    project_root = os.path.abspath(
        os.path.join(current_file,
                    os.pardir,
                    os.pardir,   
                    os.pardir,  
                    os.pardir)  
    )
    
    # Configuramos directorios para logs, modelos y resultados
    base_dir = os.path.join(project_root, "artifacts", "deep_learning", args.mode)
    logs_dir = os.path.join(base_dir, "logs")
    models_dir = os.path.join(base_dir, "models", args.config_key)
    results_dir = os.path.join(base_dir, "results", args.config_key)

    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)
    
    # Configuramos el sistema de logging
    log_file = os.path.join(logs_dir, f"{args.config_key}.log")
    logger = setup_logger(log_file)
    logger.info(f"Configuración cargada: {config}")
    
    # ================== Extracción de parámetros ==================

    # Extraemos parámetros de los argumentos
    csv_path = args.csv_path
    input_shape = tuple(args.input_shape)
    epochs = args.epochs
    n_splits = args.n_splits
    
    # ================== Configuración de transformaciones ==================

    # Lista para almacenar transformaciones adicionales
    extra_transforms = []

    # Obtenemos transformaciones extra desde la configuración (si existen)
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
                
                # Importamos dinámicamente la clase de transformación
                TransformClass = dynamic_import(transform_class_str)
                # Instanciamos la transformación con sus argumentos
                extra_transforms.append(TransformClass(**transform_args))
                logger.info(f"Transformación extra añadida: {transform_class_str} con argumentos {transform_args}")
            except Exception as e:
                logger.error(f"Error al importar la transformación {transform_class_str}: {e}")
                raise e
    else:
        logger.info("No se han definido transformaciones extra.")

    # ================== Preparación de datos ==================

    # Instanciamos el cargador de datos
    data_loader = MyDataLoader(
        csv_path=csv_path,
        input_shape=input_shape,
        config={"batch_size": 2, "num_workers": 4},     # Configuración para los DataLoaders
        transformations=extra_transforms,               # Transformaciones adicionales
        num_classes=config.get("model_args", {}).get("num_classes", 2) # Por defecto, clasificación binaria
    )
    
    # Obtenemos todos los datos
    all_data = data_loader.get_all_data()

    # Extraemos etiquetas y IDs de paciente para validación cruzada estratificada
    all_labels = [int(torch.argmax(item["label"]).item()) for item in all_data]
    patient_ids = [item["patient_id"] for item in all_data]
    
    # Creamos el objeto de validación cruzada estratificada por grupos (pacientes)
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Variables para seguimiento del mejor modelo global
    best_overall_model = None
    best_overall_score = -np.inf
    best_split_info = (None, None)
    
    # ================== Bucle de validación cruzada ==================

    # Iteramos por cada split de validación cruzada con progress bar
    split_pbar = tqdm(enumerate(splitter.split(all_data, all_labels, groups=patient_ids), start=1), 
                      total=n_splits, 
                      desc="Cross-validation splits",
                      position=0,
                      leave=True)

    for split_index, (train_idx, val_idx) in split_pbar:
        split_pbar.set_description(f"Split {split_index}/{n_splits}")
        logger.info(f"=== Split {split_index}/{n_splits} ===")

        # Dividimos los datos en subconjuntos de entrenamiento y validación
        train_subset = [all_data[i] for i in train_idx]
        val_subset = [all_data[i] for i in val_idx]

        # ================== Cálculo de pesos de clase para manejo de desbalance ==================
        
        # Obtenemos el número de clases desde la configuración (por defecto 2)
        num_classes = config.get("model_args", {}).get("num_classes", 2)
        
        # Contador para cada clase
        class_counts = torch.zeros(num_classes, dtype=torch.long)

        # Contamos ejemplos por clase en el conjunto de entrenamiento
        for item in train_subset:
            cls_idx = torch.argmax(item["label"]).item()
            class_counts[cls_idx] += 1

        # Calculamos pesos de clase inversamente proporcionales a la frecuencia
        train_len = len(train_subset)
        weights_list = [
            train_len / (class_counts[c].item() if class_counts[c] > 0 else 1e-6)
            for c in range(num_classes)
        ]
        weights_tensor = torch.tensor(weights_list, dtype=torch.float32)

        # Normalizamos los pesos para que sumen 1
        class_weights = weights_tensor / weights_tensor.sum()
        logger.info(f"Pesos de clase: {class_weights.tolist()}")
        
        # ================== Creación de datasets y dataloaders ==================

        dataset_type = config.get("data_args", {}).get("dataset_type", "regular")  # "regular", "cache", or "persistent"
        
        if dataset_type == "cache":
            cache_rate = config.get("data_args", {}).get("cache_rate", 0.5)
            train_dataset = CacheDataset(
                data=train_subset, 
                transform=data_loader.get_transforms(augment=True),
                cache_rate=cache_rate,
                num_workers=4
            )
            val_dataset = CacheDataset(
                data=val_subset, 
                transform=data_loader.get_transforms(augment=False),
                cache_rate=cache_rate,
                num_workers=4
            )
            logger.info(f"Using CacheDataset with cache_rate={cache_rate}")
            
        elif dataset_type == "persistent":
            # Use PersistentDataset - caches to disk instead of RAM
            cache_dir = os.path.join(base_dir, "persistent_cache", f"split_{split_index}")
            os.makedirs(cache_dir, exist_ok=True)
            
            train_dataset = PersistentDataset(
                data=train_subset,
                transform=data_loader.get_transforms(augment=True),
                cache_dir=os.path.join(cache_dir, "train"),
                pickle_protocol=2
            )
            val_dataset = PersistentDataset(
                data=val_subset,
                transform=data_loader.get_transforms(augment=False),
                cache_dir=os.path.join(cache_dir, "val"),
                pickle_protocol=2
            )
            logger.info(f"Using PersistentDataset with cache_dir={cache_dir}")
            
        else:
            # Use regular Dataset
            train_dataset = Dataset(data=train_subset, transform=data_loader.get_transforms(augment=True))
            val_dataset = Dataset(data=val_subset, transform=data_loader.get_transforms(augment=False))
            logger.info("Using regular Dataset")

        # Creamos dataloaders para alimentar los datos en lotes (batches)
        train_loader = DataLoader(train_dataset, batch_size=32, num_workers=8, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=8, num_workers=0, shuffle=False, pin_memory=False, persistent_workers=False)
        
        # ================== Instanciación del modelo ==================

        try:
            # Importamos dinámicamente la clase del modelo desde la configuración
            ModelClass = dynamic_import(config["model"])
            # Instanciamos el modelo con los argumentos especificados
            model = ModelClass(**config.get("model_args", {}))
            logger.info(f"Modelo instanciado: {config['model']} con argumentos {config.get('model_args', {})}")
        except Exception as e:
            logger.error(f"Error al importar el modelo {config['model']}: {e}")
            raise e
        
        # Movemos el modelo a GPU si está disponible
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        
        # ================== Configuración de criterio y optimizador ==================

        criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
        optimizer = optim.Adam(model.parameters(), lr=1e-5)
        
        # ================== Variables para tracking durante entrenamiento ==================

        # Variables para almacenar el mejor modelo y métricas por split
        best_split_model_state = None
        best_split_val_auc = -np.inf  
        best_split_epoch = 0
        split_results = []

        # Parámetros para early stopping
        patience = 20
        no_improve_count = 0
        
        # ================== Bucle de entrenamiento (por época) ==================

        # Create epoch progress bar for this split
        epoch_pbar = tqdm(range(epochs), 
                          desc=f"Split {split_index}/{n_splits} - Epoch", 
                          position=1,
                          leave=False)

        best_split_score = -np.inf
        for epoch in epoch_pbar:
            # --- Fase de entrenamiento ---
            model.train()
            train_loss_accum = 0.0
            train_correct = 0
            train_total = 0
            train_preds = []
            train_labels = []
            train_probs = []
            
            # Create batch progress bar for this epoch
            batch_pbar = tqdm(train_loader, 
                              desc=f"Split {split_index}/{n_splits} - Epoch {epoch+1}/{epochs}", 
                              position=1,
                              leave=False)
            
            for batch_idx, batch in enumerate(batch_pbar):
                # Process training batch (your existing training code)
                inputs = batch["image"].to(device)
                label_hot = batch["label"].to(device)
                label_cls = torch.argmax(label_hot, dim=1)

                # Forward pass
                optimizer.zero_grad()
                outputs = model(inputs)
                if isinstance(outputs, (tuple, list)):
                    outputs = outputs[0]
                loss = criterion(outputs, label_cls)
                loss.backward()
                optimizer.step()

                # Accumulate metrics - detach tensors to avoid gradient issues
                train_loss_accum += loss.item()
                probs = torch.softmax(outputs, dim=1)
                train_probs.append(probs[:, 1].detach().cpu())
                train_preds.append(torch.argmax(outputs, dim=1).detach().cpu())
                train_labels.append(label_cls.detach().cpu())
                
                # Update batch progress bar
                current_train_loss = train_loss_accum / (batch_idx + 1)
                
                batch_pbar.set_postfix({
                    'Loss': f'{current_train_loss:.4f}',
                    'Batch': f'{batch_idx+1}/{len(train_loader)}'
                })
            
            # Close batch progress bar after epoch completes
            batch_pbar.close()
            
            # Calculate final training metrics for this epoch
            train_loss = train_loss_accum / len(train_loader)
            
            train_labels_np = torch.cat(train_labels).numpy()
            train_preds_np = torch.cat(train_preds).numpy()
            train_probs_np = torch.cat(train_probs).numpy()

            # Calculate training AUC
            try:
                train_auc = roc_auc_score(train_labels_np, train_probs_np)
            except Exception as e:
                logger.error(f"Error calculando AUC en entrenamiento: {e}")
                train_auc = np.nan

            train_f1 = f1_score(train_labels_np, train_preds_np, average='binary')

            # --- Validation phase ---
            model.eval()
            val_loss_accum = 0.0
            val_preds = []
            val_labels = []
            val_probs = []

            # Create batch progress bar for validation
            val_batch_pbar = tqdm(val_loader, 
                                desc=f"Split {split_index}/{n_splits} - Validation", 
                                position=1,
                                leave=False)

            with torch.no_grad():
                for batch_idx, batch in enumerate(val_batch_pbar):
                    inputs = batch["image"].to(device)
                    label_hot = batch["label"].to(device)
                    label_cls = torch.argmax(label_hot, dim=1)

                    # Forward pass
                    outputs = model(inputs)
                    if isinstance(outputs, (tuple, list)):
                        outputs = outputs[0]
                    val_loss = criterion(outputs, label_cls)

                    # Accumulate metrics
                    val_loss_accum += val_loss.item()
                    probs = torch.softmax(outputs, dim=1)
                    val_probs.append(probs[:, 1].detach().cpu())
                    val_preds.append(torch.argmax(outputs, dim=1).detach().cpu())
                    val_labels.append(label_cls.detach().cpu())
                    
                    # Update validation progress bar
                    current_val_loss = val_loss_accum / (batch_idx + 1)
                    
                    val_batch_pbar.set_postfix({
                        'ValLoss': f'{current_val_loss:.4f}',
                        'Batch': f'{batch_idx+1}/{len(val_loader)}'
                    })

            # Close validation progress bar
            val_batch_pbar.close()

            # Calculate final validation metrics
            val_loss = val_loss_accum / len(val_loader)
            val_labels_np = torch.cat(val_labels).numpy()
            val_preds_np = torch.cat(val_preds).numpy()
            val_probs_np = torch.cat(val_probs).numpy()

            # Calculamos AUC en validación
            try:
                val_auc = roc_auc_score(val_labels_np, val_probs_np)
            except Exception as e:
                logger.error(f"Error calculando AUC en validación: {e}")
                val_auc = np.nan
                
            # ================== Cálculo de métricas completas para validación ==================

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
            
            # Métricas por clase
            per_class_precision = precision_score(val_labels_np, val_preds_np, average=None)
            per_class_recall = recall_score(val_labels_np, val_preds_np, average=None)
            per_class_f1 = f1_score(val_labels_np, val_preds_np, average=None)
            cm = confusion_matrix(val_labels_np, val_preds_np)
            per_class_accuracy = (cm.diagonal() / cm.sum(axis=1)).tolist()
            
            # Update epoch progress bar with key metrics
            epoch_pbar.set_postfix({
                'TrLoss': f'{train_loss:.3f}',
                'TrAUC': f'{train_auc:.3f}',
                'VLoss': f'{val_loss:.3f}',
                'VAUC': f'{val_auc:.3f}',
                'VMCC': f'{val_mcc:.3f}',
                'VF1': f'{val_f1_binary:.3f}'
            })
            
            # ================== Registro de métricas ==================

            # Registramos métricas principales en el log
            logger.info(
                f"Split {split_index}, Epoch [{epoch+1}/{epochs}] | "
                f"Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}, Train F1: {train_f1:.4f} || "
                f"Val Loss: {val_loss:.4f}, Val AUC: {val_auc:.4f}, Val MCC: {val_mcc:.4f}, Val Kappa: {val_kappa:.4f}, "
                f"Val F1 (binary): {val_f1_binary:.4f}, Val F1 (macro): {val_f1_macro:.4f}, Val Accuracy: {val_accuracy:.4f}, "
                f"Val Sensitivity: {val_sensitivity:.4f}, Val Specificity: {val_specificity:.4f}, "
                f"Val PPV: {val_ppv:.4f}, Val NPV: {val_npv:.4f}, Val BalancedAcc: {val_balanced_accuracy:.4f}"
            )
            
            # Almacenamos todas las métricas calculadas en un diccionario
            split_results.append({
                "split": split_index,
                "epoch": epoch + 1,  # +1 for 1-based indexing in logs
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
            
            # Update progress bar with current metrics
            epoch_pbar.set_postfix({
                'Train Loss': f'{train_loss:.4f}',
                'Val Loss': f'{val_loss:.4f}',
                'Val Metric': f'{val_auc:.4f}',
                'Best': f'{best_split_score:.4f}'
            })
            
            # Update best score if current epoch is better
            if val_auc > best_split_score:
                best_split_score = val_auc
            
            # ================== Early stopping ==================

            # Comprobamos si hay mejora en AUC de validación
            if val_auc > best_split_val_auc:
                best_split_val_auc = val_auc
                best_split_epoch = epoch
                best_split_model_state = model.state_dict()
                no_improve_count = 0
            else:
                no_improve_count += 1
                
            # Si llevamos muchas épocas sin mejora, paramos entrenamiento
            if no_improve_count >= patience:
                logger.info(f"Early stopping en el split {split_index} en la epoch {epoch} por falta de mejora en AUC.")
                break
        
        # ================== Guardado de resultados y modelo por split ==================
        # Delete dataset objects
        del train_dataset
        del val_dataset
        del train_loader
        del val_loader
        
        # Force garbage collection
        import gc
        gc.collect()
        
        # If using CUDA, clear GPU cache as well
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info(f"Memory cleared after split {split_index}")
        # Guardamos resultados de este split en CSV
        results_csv_path = os.path.join(results_dir, f"split_{split_index}_results.csv")
        pd.DataFrame(split_results).to_csv(results_csv_path, index=False)
        logger.info(f"Resultados del split {split_index} guardados en {results_csv_path}")
        
        # Guardamos el mejor modelo de este split
        model_save_path = os.path.join(models_dir, f"best_model_split_{split_index}.pth")
        torch.save(best_split_model_state, model_save_path)
        logger.info(f"Mejor modelo del split {split_index} guardado en {model_save_path}")
        
        # Actualizamos mejor modelo global si este split es mejor
        if best_split_val_auc > best_overall_score:
            best_overall_score = best_split_val_auc
            best_split_info = (split_index, best_split_epoch)
            best_overall_model = best_split_model_state
    
    # ================== Guardado del mejor modelo global ==================

    # Guardamos el mejor modelo global (el mejor de todos los splits)
    overall_model_path = os.path.join(models_dir, "best_overall_model.pth")
    torch.save(best_overall_model, overall_model_path)
    logger.info(f"Mejor modelo global (split {best_split_info[0]}, epoch {best_split_info[1]} con Val AUC: {best_overall_score:.4f}) guardado en {overall_model_path}")

if __name__ == "__main__":
    main()
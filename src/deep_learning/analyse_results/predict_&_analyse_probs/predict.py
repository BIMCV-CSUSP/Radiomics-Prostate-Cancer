#!/usr/bin/env python
import argparse
import json
import importlib
import logging
import os
import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedGroupKFold
from monai.data import Dataset, DataLoader
import torch.multiprocessing as mp
mp.set_sharing_strategy('file_system')
from data_loader_for_cv_for_predict import MyDataLoader

def dynamic_import(class_path):
    """Importa dinámicamente una clase dado su path completo."""
    module_name, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)

def setup_logger(log_file):
    """Configura un logger que sobrescribe el fichero de log en cada ejecución."""
    logger = logging.getLogger("predictions_logger")
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
    parser = argparse.ArgumentParser(description="Genera predicciones usando modelos entrenados.")
    parser.add_argument("--models_root", type=str, default="../../../../../data/results/deep_learning/gland/models",
                      help="Directorio raíz donde se encuentran los modelos")
    parser.add_argument("--config_file", type=str, default="../../../../../data/results/deep_learning/config.json",
                      help="Ruta al fichero JSON de configuración.")
    parser.add_argument("--csv_path", type=str, default="../../../../../data/data.csv",
                      help="Ruta al CSV de datos.")
    parser.add_argument("--input_shape", type=int, nargs=3, default=[128, 128, 32],
                      help="Dimensiones de la imagen de entrada.")
    parser.add_argument("--n_splits", type=int, default=5,
                      help="Número de splits para validación cruzada.")
    parser.add_argument("--output_dir", type=str, default="predictions",
                      help="Directorio donde guardar las predicciones.")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    log_file = os.path.join(args.output_dir, "generate_predictions.log")
    logger = setup_logger(log_file)
    
    # Encontrar todas las carpetas de modelos
    model_folders = []
    for root, dirs, files in os.walk(args.models_root):
        if any(f.endswith(".pth") for f in files):
            model_folders.append(root)
    
    logger.info(f"Se encontraron {len(model_folders)} carpetas con modelos")
    
    # Cargar configuraciones
    with open(args.config_file, "r") as f:
        configs = json.load(f)
    
    # Para cada carpeta de modelos
    for model_folder in model_folders:
        model_name = os.path.basename(model_folder)
        logger.info(f"Procesando modelos en: {model_folder}")
        
        # Comprobar si el modelo tiene configuración
        if model_name not in configs:
            logger.warning(f"No se encontró configuración para el modelo {model_name}. Usando configuración por defecto.")
            config = {"model": "models.densenet.DenseNet", "model_args": {"num_classes": 2}}
        else:
            config = configs[model_name]
        
        # Cargar datos
        data_loader = MyDataLoader(
            csv_path=args.csv_path,
            input_shape=tuple(args.input_shape),
            config={"batch_size": 2, "num_workers": 4},
            transformations=[],
            num_classes=config.get("model_args", {}).get("num_classes", 2)
        )
        
        all_data = data_loader.get_all_data()
        all_labels = [int(torch.argmax(item["label"]).item()) for item in all_data]
        patient_ids = [item["patient_id"] for item in all_data]
        
        # Encontrar todos los archivos de modelo
        model_files = [f for f in os.listdir(model_folder) if f.endswith(".pth")]
        
        # Dividir los datos
        splitter = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=42)
        
        # Lista para almacenar todas las predicciones
        all_predictions = []
        
        # Para cada split
        for split_index, (train_idx, val_idx) in enumerate(splitter.split(all_data, all_labels, groups=patient_ids), start=1):
            logger.info(f"Procesando split {split_index}/{args.n_splits}")
            
            # Buscar el modelo correspondiente a este split
            split_model_file = None
            for model_file in model_files:
                if f"split_{split_index}" in model_file:
                    split_model_file = model_file
                    break
            
            if split_model_file is None:
                logger.warning(f"No se encontró modelo para el split {split_index} en {model_folder}")
                continue

            # Cargar el modelo
            try:
                ModelClass = dynamic_import(config["model"])
                model = ModelClass(**config.get("model_args", {}))
                model_path = os.path.join(model_folder, split_model_file)
                print(model_path)
                model.load_state_dict(torch.load(model_path))
                logger.info(f"Modelo cargado: {model_path}")
            except Exception as e:
                logger.error(f"Error al cargar el modelo {split_model_file}: {e}")
                continue
            
            # Preparar conjunto de test (en este caso usamos el conjunto de validación como test)
            test_subset = [all_data[i] for i in val_idx]
            test_dataset = Dataset(data=test_subset, transform=data_loader.get_transforms(augment=False))
            test_loader = DataLoader(test_dataset, batch_size=2, num_workers=4, shuffle=False)
            
            # Configurar dispositivo
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            model.eval()
            
            # Hacer predicciones
            predictions = []
            with torch.no_grad():
                for batch in test_loader:
                    inputs = batch["image"].to(device)
                    label_hot = batch["label"].to(device)
                    label_cls = torch.argmax(label_hot, dim=1)
                    patient_ids_batch = batch["patient_id"]
                    
                    outputs = model(inputs)
                    if isinstance(outputs, (tuple, list)):
                        outputs = outputs[0]
                    
                    probs = torch.softmax(outputs, dim=1)
                    preds = torch.argmax(outputs, dim=1)
                    
                    # Guardar resultados
                    for i in range(inputs.size(0)):
                        prediction_entry = {
                            "split": split_index,
                            "model": model_name,
                            "patient_id": patient_ids_batch[i],
                            "true_label": label_cls[i].item(),
                            "prediction": preds[i].item(),
                        }
                        
                        # Guardar probabilidades para cada clase
                        for class_idx in range(probs.size(1)):
                            prediction_entry[f"prob_class_{class_idx}"] = probs[i, class_idx].item()
                        
                        predictions.append(prediction_entry)
            
            all_predictions.extend(predictions)
            logger.info(f"Realizadas {len(predictions)} predicciones para el split {split_index}")
        
        # Guardar todas las predicciones para este modelo
        if all_predictions:
            predictions_df = pd.DataFrame(all_predictions)
            output_path = os.path.join(args.output_dir, f"{model_name}_predictions.csv")
            predictions_df.to_csv(output_path, index=False)
            logger.info(f"Predicciones guardadas en: {output_path}")
    
    logger.info("Proceso de generación de predicciones completado")

if __name__ == "__main__":
    main()
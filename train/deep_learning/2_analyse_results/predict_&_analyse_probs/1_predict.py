#!/usr/bin/env python
"""
Script para generar predicciones usando modelos previamente entrenados.

Este script toma modelos entrenados para diferentes folds de validación cruzada,
los carga, y genera predicciones sobre sus respectivos conjuntos de validación.
Los resultados se guardan en archivos CSV para su posterior análisis estadístico.

Flujo de funcionamiento:
1. Busca carpetas que contengan modelos guardados (.pth)
2. Carga la configuración correspondiente a cada modelo
3. Para cada modelo encuentra los archivos para cada split de validación cruzada
4. Carga los datos y realiza predicciones sobre el conjunto de validación
5. Guarda las predicciones junto con las probabilidades por clase
"""

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
from z_data_loader_for_cv_for_predict import MyDataLoader

def dynamic_import(class_path):
    """
    Importa dinámicamente una clase desde su ruta completa de módulo.
    
    Permite cargar clases (modelos) desde rutas especificadas en configuración
    sin tener que importarlas explícitamente.
    
    Args:
        class_path (str): Ruta completa de la clase en formato 'modulo.submodulo.Clase'
        
    Returns:
        class: La clase importada (no una instancia)
    """
    module_name, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)

def setup_logger(log_file):
    """
    Configura un sistema de registro que escribe tanto a archivo como a consola.
    
    Args:
        log_file (str): Ruta donde guardar el archivo de log
        
    Returns:
        logger: Objeto logger configurado
    """ 
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
    """
    Función principal que coordina el proceso completo de generación de predicciones.
    
    Proceso:
    1. Procesa argumentos de línea de comandos
    2. Configura sistema de logging y directorios
    3. Busca carpetas con modelos entrenados
    4. Carga configuraciones de cada modelo
    5. Para cada modelo y split de validación cruzada:
       - Carga el modelo entrenado
       - Prepara los datos de validación
       - Genera predicciones
       - Guarda resultados
    """

    # ============= Procesamiento de argumentos =============

    parser = argparse.ArgumentParser(
        description="Genera predicciones usando modelos entrenados."
    )
    parser.add_argument(
        "--mode", type=str, choices=["gland", "full"], default="gland",
        help="Modo de predicción: 'gland' o 'full'."
    )
    parser.add_argument(
        "--data_root", type=str, default="../../../../artifacts/deep_learning",
        help="Directorio raíz donde se ubican las carpetas de resultados y modelos."
    )
    parser.add_argument(
        "--config_file", type=str, default="../../1_modeling/config.json",
        help="Ruta al fichero JSON de configuración."
    )
    parser.add_argument(
        "--csv_path", type=str, default="../../../../artifacts/data.csv",
        help="Ruta al CSV de datos."
    )
    parser.add_argument(
        "--input_shape", type=int, nargs=3, default=[128, 128, 32],
        help="Dimensiones de la imagen de entrada."
    )
    parser.add_argument(
        "--n_splits", type=int, default=5,
        help="Número de splits para validación cruzada."
    )
    parser.add_argument(
        "--output_base", type=str, default="../../../../results/deep_learning/model_comparison/predict_&_analyse_probs/",
        help="Directorio base para guardar las predicciones."
    )

    args = parser.parse_args()

    # ============= Configuración de directorios y logging =============

    # Construcción dinámica de rutas según el modo
    models_root = os.path.join(args.data_root, args.mode, "models")
    output_dir = os.path.join(args.output_base, f"{args.mode}_analysis", "predictions")

    # Crear directorio de salida si no existe
    os.makedirs(output_dir, exist_ok=True)

    # Configurar sistema de logging
    log_file = os.path.join(output_dir, "generate_predictions.log")
    logger = setup_logger(log_file)

    # ============= Búsqueda de carpetas con modelos =============
    # Reunir carpetas que contengan archivos de modelo (.pth)
    model_folders = []
    for root, dirs, files in os.walk(models_root):
        if any(f.endswith(".pth") for f in files):
            model_folders.append(root)
    
    logger.info(f"Se encontraron {len(model_folders)} carpetas con modelos")
    
    # ============= Cargar configuraciones =============
    with open(args.config_file, "r") as f:
        configs = json.load(f)
    
    # ============= Procesamiento de cada modelo =============
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
        
        # ============= Cargar datos =============
        data_loader = MyDataLoader(
            csv_path=args.csv_path,
            input_shape=tuple(args.input_shape),
            config={"batch_size": 2, "num_workers": 4},
            transformations=[], # Sin transformaciones adicionales para predicción
            num_classes=config.get("model_args", {}).get("num_classes", 2)
        )

        # Obtener todos los datos
        all_data = data_loader.get_all_data()

        # Extraer etiquetas y IDs para validación cruzada
        all_labels = [int(torch.argmax(item["label"]).item()) for item in all_data]
        patient_ids = [item["patient_id"] for item in all_data]
        
        # Encontrar todos los archivos de modelo en esta carpeta
        model_files = [f for f in os.listdir(model_folder) if f.endswith(".pth")]
        
        # ============= Configurar validación cruzada =============
        # Dividir los datos usando la misma estrategia que en entrenamiento
        splitter = StratifiedGroupKFold(n_splits=args.n_splits, shuffle=True, random_state=42)
        
        # Lista para almacenar todas las predicciones
        all_predictions = []
        
        # ============= Procesar cada split =============
        # Para cada split de validación cruza
        for split_index, (train_idx, val_idx) in enumerate(splitter.split(all_data, all_labels, groups=patient_ids), start=1):
            logger.info(f"Procesando split {split_index}/{args.n_splits}")
            
            # Buscar el modelo correspondiente a este split
            split_model_file = None
            for model_file in model_files:
                if f"split_{split_index}" in model_file:
                    split_model_file = model_file
                    break
            
            # Si no encontramos modelo para este split, pasar al siguiente
            if split_model_file is None:
                logger.warning(f"No se encontró modelo para el split {split_index} en {model_folder}")
                continue

            # ============= Cargar modelo =============
            try:
                # Importar dinámicamente la clase del modelo
                ModelClass = dynamic_import(config["model"])

                # Instanciar el modelo con los argumentos de configuración
                model = ModelClass(**config.get("model_args", {}))
                model_path = os.path.join(model_folder, split_model_file)

                # Cargar los pesos del modelo entrenado
                model.load_state_dict(torch.load(model_path))
                logger.info(f"Modelo cargado: {model_path}")
            except Exception as e:
                logger.error(f"Error al cargar el modelo {split_model_file}: {e}")
                continue
            
            # ============= Preparar datos de validación =============
            # Tomamos el conjunto de validación como "test" para este split            
            test_subset = [all_data[i] for i in val_idx]
            test_dataset = Dataset(data=test_subset, transform=data_loader.get_transforms(augment=False))
            test_loader = DataLoader(test_dataset, batch_size=2, num_workers=4, shuffle=False)
            
            # ============= Configurar dispositivo (CPU/GPU) =============
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            model = model.to(device)
            model.eval()
            
            # ============= Generar predicciones =============
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
                    
                    # Calcular probabilidades y predicciones
                    probs = torch.softmax(outputs, dim=1)
                    preds = torch.argmax(outputs, dim=1)
                    
                    # Guardar resultados para cada muestra en el batch
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
            
            # Añadir predicciones de este split a la lista global
            all_predictions.extend(predictions)
            logger.info(f"Realizadas {len(predictions)} predicciones para el split {split_index}")
        
        # ============= Guardar predicciones =============
        # Guardar todas las predicciones para este modelo
        if all_predictions:
            predictions_df = pd.DataFrame(all_predictions)
            output_file = os.path.join(output_dir, f"{model_name}_predictions.csv")
            predictions_df.to_csv(output_file, index=False)
            logger.info(f"Predicciones guardadas en: {output_file}")
    
    logger.info("Proceso de generación de predicciones completado")

if __name__ == "__main__":
    main()
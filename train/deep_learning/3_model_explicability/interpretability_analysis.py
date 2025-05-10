#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script para análisis de interpretabilidad de modelos de aprendizaje profundo
para clasificación de cáncer de próstata.

Uso:
    python interpretability_analysis.py --model-type base-densenet --criteria correct_class1 --max-samples 3

Argumentos:
    --model-type: Tipo de modelo a analizar (ej. base-densenet, base-efficientnet)
    --criteria: Criterio de selección de muestras (correct_class0, correct_class1, incorrect, high_confidence, any)
    --max-samples: Número máximo de muestras a analizar
    --skip-gradcam: Si se especifica, no se calcula GradCAM
    --skip-occlusion: Si se especifica, no se calculan mapas de oclusión
    --skip-aggregated: Si se especifica, no se utilizan mapas agregados
    --max-attempts: Número máximo de intentos para encontrar muestras aleatorias
    --split: Split específico a usar (si no se especifica, se usa el mejor split)
    --project-root: Ruta raíz del proyecto (opcional)
    --output-dir: Directorio donde guardar los resultados (opcional)
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import monai
from tqdm import tqdm
import os
import glob
import json
import importlib
import sys
import argparse
from sklearn.model_selection import StratifiedGroupKFold

def parse_arguments():
    """Parsea los argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(description="Análisis de interpretabilidad de modelos de deep learning")
    
    parser.add_argument("--model-type", type=str, default="base-densenet",
                        help="Tipo de modelo a analizar (ej. base-densenet)")
    
    parser.add_argument("--criteria", type=str, default="correct_class1",
                        choices=["correct_class0", "correct_class1", "incorrect", "high_confidence", "any"],
                        help="Criterio para seleccionar muestras")
    
    parser.add_argument("--max-samples", type=int, default=3,
                        help="Número máximo de muestras a analizar")
    
    parser.add_argument("--skip-gradcam", action="store_true",
                        help="Si se especifica, no se calcula GradCAM")
    
    parser.add_argument("--skip-occlusion", action="store_true",
                        help="Si se especifica, no se calculan mapas de oclusión")
    
    parser.add_argument("--skip-aggregated", action="store_true",
                        help="Si se especifica, no se utilizan mapas agregados")
    
    parser.add_argument("--max-attempts", type=int, default=100,
                        help="Número máximo de intentos para encontrar muestras aleatorias")
    
    parser.add_argument("--split", type=int, default=None,
                        help="Split específico a usar (si no se especifica, se usa el mejor split)")
    
    parser.add_argument("--project-root", type=str, default=os.path.abspath("../../.."),
                        help="Ruta raíz del proyecto")
    
    parser.add_argument("--output-dir", type=str, default="../../../results/deep_learning/interpretability",
                        help="Directorio donde guardar los resultados")
    
    return parser.parse_args()

def dynamic_import(class_path):
    """Importa dinámicamente una clase desde una ruta."""
    module_name, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)

def load_model_and_test_data(model_dir, csv_path, split_to_use=None, project_root=None):
    """
    Carga un modelo y recupera los datos de test correspondientes a un split específico.
    
    Args:
        model_dir: Directorio donde se encuentra el modelo
        csv_path: Ruta al CSV de datos
        split_to_use: Split específico a usar como test (si es None, usa el mejor split)
        project_root: Ruta raíz del proyecto (opcional)
    
    Returns:
        model: Modelo cargado
        test_dataloader: DataLoader con los datos de test
        split_used: Split utilizado
    """
    
    # Cargar configuración desde config.json
    config_path = os.path.join(project_root, "train/deep_learning/1_modeling/config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Obtener configuración para este modelo específico
    model_type = os.path.basename(model_dir)
    if model_type not in config:
        raise ValueError(f"Tipo de modelo {model_type} no encontrado en config.json")
    
    model_config = config[model_type]
    # Importar dinámicamente la clase del modelo
    model_class_path = model_config["model"]
    module_path, class_name = model_class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    model_class = getattr(module, class_name)
    
    # Crear instancia del modelo
    model = model_class(**model_config["model_args"])
    
    # Determinar qué modelo cargar (mejor overall o específico de un split)
    if split_to_use is None:
        # Cargar el mejor modelo overall
        model_path = os.path.join(model_dir, "best_overall_model.pth")
        # Intentar determinar cuál fue el mejor split usado para este modelo
        base_dir = os.path.dirname(os.path.dirname(model_dir))  # Sube dos niveles (para saltar "models")
        results_dir = os.path.join(base_dir, "results", os.path.basename(model_dir))
        print(f"Buscando mejores resultados en: {results_dir}")
        split_files = glob.glob(os.path.join(results_dir, "split_*_results.csv"))
        best_auc = -np.inf
        best_split = None
        
        for split_file in split_files:
            df = pd.read_csv(split_file)
            max_auc = df['val_auc'].max()
            if max_auc > best_auc:
                best_auc = max_auc
                split_num = int(os.path.basename(split_file).split('_')[1])
                best_split = split_num
        
        split_to_use = best_split
        print(f"Usando el split {split_to_use} como test (mejor AUC: {best_auc:.4f})")
    else:
        # Cargar un modelo específico de un split
        model_path = os.path.join(model_dir, f"best_model_split_{split_to_use}.pth")
    
    # Cargar pesos del modelo
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()
    
    # Determinar modo de datos (full o gland)
    if "gland" in model_dir:
        loader_module = "data_loaders.data_loader_for_cv_roi"
    else:
        loader_module = "data_loaders.data_loader_for_cv_org"
    
    # Importar dinámicamente la clase de carga de datos
    MyDataLoader = dynamic_import(f"{loader_module}.MyDataLoader")
    
    # Preparar datos
    data_loader = MyDataLoader(
        csv_path=csv_path,
        input_shape=(128, 128, 32),  
        config={"batch_size": 1, "num_workers": 4},
    )
    
    # Obtener todos los datos
    all_data = data_loader.get_all_data()
    
    # Extraer etiquetas y IDs de paciente para validación cruzada estratificada
    all_labels = [int(torch.argmax(item["label"]).item()) for item in all_data]
    patient_ids = [item["patient_id"] for item in all_data]
    
    # Crear el objeto de validación cruzada estratificada por grupos (pacientes)
    n_splits = 5  # Ajustar según tu entrenamiento original
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    # Obtener los índices del split que queremos usar como test
    splits = list(splitter.split(all_data, all_labels, groups=patient_ids))
    _, test_idx = splits[split_to_use - 1]  # Restamos 1 porque los splits se numeran desde 1 en el script original
    
    # Obtener datos de test
    test_subset = [all_data[i] for i in test_idx]
    
    # Crear dataset y dataloader para test
    test_dataset = monai.data.Dataset(
        data=test_subset, 
        transform=data_loader.get_transforms(augment=False)
    )
    
    test_dataloader = monai.data.DataLoader(
        test_dataset,
        batch_size=1,  
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return model, test_dataloader, split_to_use

def calculate_occlusion_sensitivity(model, test_dataloader, maps_dir, occlusion_dir):
    """
    Calcula los mapas de sensibilidad de oclusión si no existen.
    
    Args:
        model: Modelo entrenado
        test_dataloader: DataLoader con los datos de test
        maps_dir: Directorio donde guardar los mapas individuales
        occlusion_dir: Directorio donde guardar los mapas agregados
    
    Returns:
        bool: True si se crearon nuevos mapas, False si ya existían
    """
    # Comprobar si ya existen los mapas agregados
    aggregated_maps_path = os.path.join(occlusion_dir, "aggregated_heatmaps.pth")
    
    # Comprobar si existen los mapas individuales
    individual_maps_exist = False
    if os.path.exists(maps_dir):
        individual_maps = glob.glob(os.path.join(maps_dir, "class*_*.pt")) + glob.glob(os.path.join(maps_dir, "class*_*.*"))
        individual_maps_exist = len(individual_maps) > 0
    
    # Si existen ambos, no es necesario recalcular
    if os.path.exists(aggregated_maps_path) and individual_maps_exist:
        print(f"✓ Los mapas agregados ya existen en {aggregated_maps_path}")
        print(f"✓ Encontrados {len(individual_maps)} mapas individuales en {maps_dir}")
        print("No es necesario recalcular los mapas de oclusión.")
        return False
    
    print("Calculando mapas de sensibilidad de oclusión...")
    
    # Crear directorios si no existen
    os.makedirs(maps_dir, exist_ok=True)
    os.makedirs(occlusion_dir, exist_ok=True)
    
    results = [torch.zeros((3, 128, 128, 32)), torch.zeros((3, 128, 128, 32))]
    counts = [0, 0]
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    occ_sens = monai.visualize.OcclusionSensitivity(nn_module=model, mask_size=8, n_batch=64, verbose=False)
    
    with torch.no_grad():    
        for data in tqdm(test_dataloader):
            img, label = data["image"].to(device), data["label"].to(device)
            pred_label = torch.nn.functional.softmax(model(img), dim=1).argmax().item()
            label = label.argmax().item()
            
            if label == pred_label:
                occ_result, _ = occ_sens(x=img)
                occ_result = occ_result[0, label][None]
                
                filename = os.path.basename(img.meta['filename_or_obj'][0])
                    
                torch.save(occ_result.cpu(), os.path.join(maps_dir, f"class{label}_{filename}"))
                results[label] += occ_result.cpu()
                counts[label] += 1
    
    print(f"Muestras por clase: Clase 0 = {counts[0]}, Clase 1 = {counts[1]}")
    
    # Promediar los mapas por clase
    no_csPCa = results[0] / max(counts[0], 1)  # Evitar división por cero
    csPCa = results[1] / max(counts[1], 1)     # Evitar división por cero
    
    # Guardar mapas agregados
    torch.save({"no_csPCa": no_csPCa, "csPCa": csPCa}, aggregated_maps_path)
    print(f"✓ Mapas agregados guardados en {aggregated_maps_path}")
    
    return True

def comprehensive_model_interpretation(dataloader, model, model_results_dir, criteria='correct_class0', 
                                   max_samples=3, max_attempts=100, csv_path=None,
                                   use_gradcam=True, use_occlusion=True, use_aggregated_maps=True,
                                   maps_dir=None, sensitivity_maps_dir=None):
    """
    Aplica múltiples técnicas de interpretabilidad a las mismas muestras seleccionadas.
    Para criteria='correct_class1', selecciona muestras ordenadas por ISUP (descendente).
    
    Args:
        dataloader: DataLoader con las muestras
        model: Modelo entrenado para hacer predicciones
        model_results_dir: Directorio de resultados específico para este modelo (ya creado)
        criteria: Criterio de selección ('correct_class0', 'correct_class1', 'incorrect', 'high_confidence', 'any')
        max_samples: Número máximo de muestras a procesar
        max_attempts: Número máximo de intentos para encontrar muestras aleatorias (no aplica para correct_class1)
        csv_path: Ruta al CSV con información ISUP (requerido para criteria='correct_class1')
        use_gradcam: Si se debe aplicar GradCAM y GuidedBackpropSmoothGrad
        use_occlusion: Si se debe buscar mapas de oclusión individuales
        use_aggregated_maps: Si se deben usar mapas de calor agregados por clase
        maps_dir: Directorio donde se encuentran los mapas agregados
        sensitivity_maps_dir: Directorio donde se encuentran los mapas de sensibilidad
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    
    # Verificar directorios necesarios
    if use_aggregated_maps and maps_dir is None:
        print("⚠ Se requiere el directorio de mapas agregados para use_aggregated_maps=True")
        use_aggregated_maps = False
    
    if use_occlusion and sensitivity_maps_dir is None:
        print("⚠ Se requiere el directorio de mapas de sensibilidad para use_occlusion=True")
        use_occlusion = False
    
    # Cargar mapas agregados si se requieren
    if use_aggregated_maps:
        try:
            maps = torch.load(maps_dir)
            print(f"✓ Mapas agregados cargados desde {maps_dir}")
        except Exception as e:
            print(f"⚠ Error al cargar mapas agregados: {e}")
            use_aggregated_maps = False
    
    # Inicializar CAM y GBP si se requieren
    if use_gradcam:
        try:
            if "EfficientNet" in str(type(model)):
                target_layer = "_conv_head"
                fc_layer = "_fc"
            elif "DenseNet" in str(type(model)):
                target_layer = "features.norm5"
                fc_layer = "class_layers.out"
            else:
                # Configuración predeterminada o para otros modelos
                print("Arquitectura de modelo no reconocida, intentando con configuración predeterminada")
                target_layer = "_conv_head"
                fc_layer = "_fc"
            
            cam = monai.visualize.class_activation_maps.CAM(
                nn_module=model, 
                target_layers=target_layer, 
                fc_layers=fc_layer
            )
            gbp = monai.visualize.gradient_based.GuidedBackpropSmoothGrad(
                model, 
                n_samples=50
            )
            print("✓ GradCAM y GuidedBackpropSmoothGrad inicializados")
        except Exception as e:
            print(f"⚠ Error al inicializar GradCAM: {e}")
            use_gradcam = False
    
    # Crear directorio para análisis completo dentro del directorio del modelo
    results_base_dir = os.path.join(model_results_dir)
    os.makedirs(results_base_dir, exist_ok=True)
    print(f"Los resultados se guardarán en: {results_base_dir}")
    
    # Variables para almacenar muestras seleccionadas
    selected_samples = []
    
    # Parte 1: Selección de muestras
    print("\n" + "="*80)
    print("FASE 1: SELECCIÓN DE MUESTRAS")
    print("="*80)
    
    # ========= CASO ESPECIAL: SELECCIÓN POR ISUP PARA 'correct_class1' =========
    if criteria == 'correct_class1' and csv_path is not None:
        print("Modo de selección: Casos csPCa ordenados por ISUP descendente")
        
        # Cargar CSV con información ISUP
        try:
            isup_data = pd.read_csv(csv_path)
            print(f"✓ CSV cargado con {len(isup_data)} registros")
        except Exception as e:
            print(f"⚠ Error al cargar CSV: {e}")
            return
        
        # Diccionarios para almacenar candidatos por nivel ISUP
        samples_found = 0
        
        # Buscar secuencialmente por ISUP decreciente
        for isup_target in [5, 4, 3, 2, 1]:
            if samples_found >= max_samples:
                break
                
            print(f"\nBuscando casos con ISUP {isup_target}...")
            
            # Recorrer dataset hasta encontrar suficientes casos o agotarlo
            for idx in range(len(dataloader.dataset)):
                if samples_found >= max_samples:
                    break
                    
                try:
                    # Obtener datos de la muestra
                    sample = dataloader.dataset[idx]
                    img = sample['image'].to(device)
                    label = sample['label']
                    true_class = label.argmax().item()
                    
                    # Solo procesamos casos csPCa (clase 1)
                    if true_class != 1:
                        continue
                    
                    # Verificar mapas de oclusión si es necesario
                    filename = None
                    if use_occlusion:
                        try:
                            filename = os.path.basename(img.meta['filename_or_obj'])
                            map_path = os.path.join(sensitivity_maps_dir, f"class{true_class}_{filename}")
                            
                            if not os.path.exists(map_path):
                                continue
                        except Exception:
                            continue
                    else:
                        try:
                            filename = os.path.basename(img.meta['filename_or_obj'])
                        except Exception:
                            filename = f"sample_{idx}"
                    
                    # Extraer patient_id y study_id del nombre del archivo
                    try:
                        parts = filename.split('_')
                        patient_id = int(parts[0])
                        study_id_part = parts[1].split('.')[0]
                        study_id = int(study_id_part)
                        
                        # Buscar información ISUP en el CSV
                        patient_data = isup_data[
                            (isup_data['patient_id'] == patient_id) & 
                            (isup_data['study_id'] == study_id)
                        ]
                        
                        if len(patient_data) == 0:
                            continue
                        
                        # Obtener valor ISUP
                        isup_value = patient_data.iloc[0]['case_ISUP']
                        
                        # Solo considerar casos con el ISUP objetivo
                        if isup_value != isup_target:
                            continue
                        
                        # Hacer predicción
                        with torch.no_grad():
                            output = model(img.unsqueeze(0))
                            pred = torch.nn.functional.softmax(output, dim=1)
                            pred_class = pred.argmax().item()
                        
                        # Solo considerar predicciones correctas
                        if pred_class != 1:
                            continue
                        
                        # ¡Caso válido encontrado!
                        sample_info = {
                            "idx": idx,
                            "img": img,
                            "label": label,
                            "true_class": true_class,
                            "pred": pred,
                            "pred_class": pred_class,
                            "filename": filename,
                            # Guardamos isup para referencia interna, no se mostrará en metadatos
                            "isup": isup_value
                        }
                        
                        selected_samples.append(sample_info)
                        samples_found += 1
                        
                        print(f"✓ Muestra {samples_found}/{max_samples} encontrada")
                        print(f"  ID: {idx}")
                        print(f"  ISUP: {isup_value}")
                        print(f"  Confianza: {pred[0, 1].item():.4f}")
                        
                        # Si ya tenemos suficientes muestras, terminamos
                        if samples_found >= max_samples:
                            break
                        
                    except Exception as e:
                        continue
                        
                except Exception:
                    continue
            
            # Mostrar resultados para este ISUP
            samples_this_isup = len([s for s in selected_samples if s["isup"] == isup_target])
            print(f"Encontrados {samples_this_isup} casos con ISUP {isup_target}")
        
        # Verificar si encontramos suficientes muestras
        if len(selected_samples) == 0:
            print("⚠ No se encontraron muestras que cumplan los criterios")
            return
        
        print(f"\n✓ Seleccionadas {len(selected_samples)} muestras para análisis")
    
    # ========= SELECCIÓN ESTÁNDAR PARA OTROS CRITERIOS =========
    else:
        # Selección aleatoria según criterios originales
        print(f"Modo de selección: Aleatorio según criterio '{criteria}'")
        
        samples_processed = 0
        for attempt in range(max_attempts):
            if samples_processed >= max_samples:
                break
                
            # Seleccionar muestra aleatoria
            idx = np.random.randint(0, len(dataloader.dataset))
            img = dataloader.dataset[idx]['image'].to(device)
            label = dataloader.dataset[idx]['label']
            true_class = label.argmax().item()
            
            # Comprobar si ya tenemos esta muestra
            if idx in [s["idx"] for s in selected_samples]:
                print(f"Intento {attempt+1}: Muestra {idx} ya seleccionada anteriormente")
                continue
                
            # Verificar mapas de oclusión si es necesario
            if use_occlusion:
                try:
                    filename = os.path.basename(img.meta['filename_or_obj'])
                    map_path = os.path.join(sensitivity_maps_dir, f"class{true_class}_{filename}")
                    
                    if not os.path.exists(map_path):
                        continue
                except (KeyError, AttributeError) as e:
                    print(f"Intento {attempt+1}: Error al verificar mapa de oclusión: {e}")
                    continue
            else:
                try:
                    filename = os.path.basename(img.meta['filename_or_obj'])
                except Exception:
                    filename = f"sample_{idx}"
            
            # Hacer predicción
            with torch.no_grad():
                output = model(img.unsqueeze(0))
                pred = torch.nn.functional.softmax(output, dim=1)
                pred_class = pred.argmax().item()
            
            # Verificar criterios
            if criteria == 'correct_class0' and (pred_class != 0 or true_class != 0):
                continue
                
            elif criteria == 'incorrect' and pred_class == true_class:
                continue
                
            elif criteria == 'high_confidence' and pred[0, pred_class].item() <= 0.9:
                continue
            
            # Si llegamos aquí, la muestra cumple todos los criterios
            print(f"✓ Muestra {samples_processed+1} encontrada (intento {attempt+1})")
            print(f"  ID: {idx}")
            print(f"  Forma: {img.shape}")
            print(f"  Etiqueta real: {true_class} ({'no_csPCa' if true_class==0 else 'csPCa'})")
            print(f"  Predicción: {pred_class} ({'no_csPCa' if pred_class==0 else 'csPCa'})")
            print(f"  Confianza: {pred[0, pred_class].item():.4f}")
            
            # Guardar información de la muestra
            sample_info = {
                "idx": idx,
                "img": img,
                "label": label,
                "true_class": true_class,
                "pred": pred,
                "pred_class": pred_class,
                "filename": filename
            }
            
            selected_samples.append(sample_info)
            samples_processed += 1
        
        # Verificar si encontramos suficientes muestras
        if len(selected_samples) == 0:
            print(f"⚠ No se encontraron muestras que cumplan con el criterio '{criteria}' después de {max_attempts} intentos")
            return
        
        print(f"\n✓ Seleccionadas {len(selected_samples)} muestras para análisis")
    
    # Parte 2: Aplicar técnicas de interpretabilidad a cada muestra
    for i, sample in enumerate(selected_samples):
        print("\n" + "="*80)
        print(f"ANÁLISIS DE MUESTRA {i+1}/{len(selected_samples)}")
        print("="*80)
        
        # Extraer información de la muestra
        idx = sample["idx"]
        img = sample["img"]
        true_class = sample["true_class"]
        pred_class = sample["pred_class"]
        pred = sample["pred"]
        filename = sample["filename"]
        
        # Crear nombre de directorio homogeneizado para todos los casos
        criteria_name = criteria.replace('_', '-')
        sample_dir_name = f"{criteria_name}_sample{i+1}_idx{idx}_class{true_class}"
        
        sample_dir = os.path.join(results_base_dir, sample_dir_name)
        os.makedirs(sample_dir, exist_ok=True)
        
        # Guardar metadatos homogeneizados
        with open(os.path.join(sample_dir, "metadata.txt"), 'w') as f:
            f.write(f"Índice: {idx}\n")
            f.write(f"Forma: {img.shape}\n")
            f.write(f"Nombre archivo: {filename}\n")
            f.write(f"Clase real: {true_class} ({'no_csPCa' if true_class==0 else 'csPCa'})\n")
            f.write(f"Predicción: {pred_class} ({'no_csPCa' if pred_class==0 else 'csPCa'})\n")
            f.write(f"Probabilidades: [no_csPCa: {pred[0, 0].item():.4f}, csPCa: {pred[0, 1].item():.4f}]\n")
        
        # 1. Guardar visualización de canales originales
        channel_names = ['T2W', 'ADC', 'DWI']
        plt.figure(figsize=(18, 6))
        for c_idx, c_name in enumerate(channel_names):
            plt.subplot(1, 3, c_idx+1)
            slice_idx = img.shape[3] // 2  # Slice central
            plt.imshow(img[c_idx, :, :, slice_idx].cpu().numpy(), cmap='gray')
            plt.title(f"Canal {c_name}")
        
        plt.suptitle(f"Imagen original - Clase: {true_class} ({'no_csPCa' if true_class==0 else 'csPCa'})")
        plt.tight_layout()
        plt.savefig(os.path.join(sample_dir, "original_channels.png"), dpi=300)
        plt.close()
        
        # 2. Aplicar GradCAM si está habilitado
        if use_gradcam:
            print("\nAplicando GradCAM y GuidedBackpropSmoothGrad...")
            gradcam_dir = os.path.join(sample_dir, "GradCAM")
            os.makedirs(gradcam_dir, exist_ok=True)
            
            try:
                # Generar mapas
                cam_result = cam(x=img.unsqueeze(0)).squeeze(0).cpu()
                gbp_result = gbp(x=img.unsqueeze(0)).squeeze(0).cpu()
                
                # Normalizar resultados
                gbp_result = 255*(gbp_result - gbp_result.min())/(gbp_result.max() - gbp_result.min())
                cam_result = 255*(cam_result - cam_result.min())/(cam_result.max() - cam_result.min())
                result = cam_result * gbp_result
                result = (result - result.min())/(result.max() - result.min())
                
                # Center crop para mejor visualización
                img_cropped = monai.transforms.CenterSpatialCrop(roi_size=(64, 64, -1))(img)
                cam_result_cropped = monai.transforms.CenterSpatialCrop(roi_size=(64, 64, -1))(cam_result)
                gbp_result_cropped = monai.transforms.CenterSpatialCrop(roi_size=(64, 64, -1))(gbp_result)
                result_cropped = monai.transforms.CenterSpatialCrop(roi_size=(64, 64, -1))(result)
                
                # Guardar visualizaciones 3D
                # GBP Result
                plt.figure(figsize=(12, 8))
                monai.visualize.matshow3d(volume=gbp_result_cropped[1:2], frame_dim=-1, channel_dim=0, every_n=2, margin=6, show=False)
                plt.suptitle(f"Guided Backpropagation (canal ADC)")
                plt.savefig(os.path.join(gradcam_dir, "GBP_3D.png"), dpi=300)
                plt.close()
                
                # CAM Result
                plt.figure(figsize=(12, 8))
                monai.visualize.matshow3d(volume=cam_result_cropped, frame_dim=-1, channel_dim=0, every_n=2, margin=6, show=False)
                plt.suptitle(f"Class Activation Map (todos los canales)")
                plt.savefig(os.path.join(gradcam_dir, "CAM_3D.png"), dpi=300)
                plt.close()
                
                # Combined Result
                plt.figure(figsize=(12, 8))
                monai.visualize.matshow3d(volume=result_cropped, frame_dim=-1, channel_dim=0, every_n=2, margin=6, fill_value=255, show=False, cmap='gray')
                plt.suptitle(f"Resultado combinado GradCAM * GBP")
                plt.savefig(os.path.join(gradcam_dir, "Combined_3D.png"), dpi=300)
                plt.close()
                
                # Visualizaciones combinadas para cada canal
                for c_idx, c_name in enumerate(channel_names):
                    blended = monai.visualize.utils.blend_images(
                        255*img_cropped.cpu()[c_idx:c_idx+1], 
                        255*result_cropped[c_idx:c_idx+1], 
                        alpha=0.2,
                        transparent_background=True
                    )
                    plt.figure(figsize=(12, 8))
                    monai.visualize.matshow3d(volume=blended, frame_dim=-1, channel_dim=0, every_n=2, margin=6, show=False)
                    plt.suptitle(f"Canal {c_name} con mapa combinado superpuesto")
                    plt.savefig(os.path.join(gradcam_dir, f"Blended_{c_name}_3D.png"), dpi=300)
                    plt.close()
                
                print(f"✓ Resultados GradCAM guardados en {gradcam_dir}")
                
            except Exception as e:
                print(f"⚠ Error al procesar GradCAM: {e}")
                import traceback
                traceback.print_exc()
        
        # 3. Aplicar visualización de mapas agregados si está habilitado
        if use_aggregated_maps:
            print("\nAplicando visualización de mapas agregados...")
            aggregated_dir = os.path.join(sample_dir, "AggregatedMaps")
            os.makedirs(aggregated_dir, exist_ok=True)
            
            try:
                # Normalizar los mapas
                no_csPCa = (maps["no_csPCa"] - maps["no_csPCa"].min()) / (maps["no_csPCa"].max() - maps["no_csPCa"].min())
                csPCa = (maps["csPCa"] - maps["csPCa"].min()) / (maps["csPCa"].max() - maps["csPCa"].min())
                
                # Umbral para filtrar valores
                threshold = 0.7
                
                # Visualizar mapas para cada canal
                for c_idx, c_name in enumerate(channel_names):
                    # Visualización de cada clase
                    for map_class, map_name, map_data in [
                        (0, "no_csPCa", no_csPCa), 
                        (1, "csPCa", csPCa)
                    ]:
                        # Aplicar umbral
                        thresholded_map = torch.where(
                            map_data.cpu()[c_idx:c_idx+1] > threshold,
                            map_data.cpu()[c_idx:c_idx+1],
                            torch.zeros_like(map_data.cpu()[c_idx:c_idx+1])
                        )
                        
                        # Mezclar con la imagen original
                        blended = monai.visualize.utils.blend_images(
                            255 * img.cpu()[c_idx:c_idx+1],
                            255 * thresholded_map,
                            alpha=0.3,
                            transparent_background=True
                        )
                        
                        # Visualizar
                        plt.figure(figsize=(15, 10))
                        monai.visualize.matshow3d(
                            volume=blended,
                            frame_dim=-1,
                            channel_dim=0,
                            every_n=4,
                            margin=10,
                            figsize=(15, 10),
                            show=False
                        )
                        
                        plt.suptitle(
                            f"Canal {c_name} con mapa agregado para clase {map_name}\n" +
                            f"Clase real: {true_class} ({'no_csPCa' if true_class==0 else 'csPCa'}), " +
                            f"Predicción: {pred_class} ({'no_csPCa' if pred_class==0 else 'csPCa'})",
                            fontsize=14
                        )
                        
                        plt.tight_layout()
                        plt.savefig(os.path.join(aggregated_dir, f"{c_name}_map_{map_name}.png"), dpi=300)
                        plt.close()
                
                print(f"✓ Resultados de mapas agregados guardados en {aggregated_dir}")
                
            except Exception as e:
                print(f"⚠ Error al procesar mapas agregados: {e}")
                import traceback
                traceback.print_exc()
        
        # 4. Aplicar visualización de mapas de oclusión individuales si está habilitado
        if use_occlusion:
            print("\nAplicando visualización de mapas de oclusión individuales...")
            occlusion_dir = os.path.join(sample_dir, "OcclusionSensitivity")
            os.makedirs(occlusion_dir, exist_ok=True)
            
            try:
                # Cargar mapa de oclusión
                map_path = os.path.join(sensitivity_maps_dir, f"class{true_class}_{filename}")
                heatmap = torch.load(map_path)
                
                # Normalizar el mapa
                heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
                
                # Umbral para filtrar valores
                threshold = 0.8
                
                # Visualizar para cada canal
                for c_idx, c_name in enumerate(channel_names):
                    # Aplicar umbral
                    thresholded_map = torch.where(
                        heatmap.cpu() > threshold,
                        heatmap.cpu(),
                        torch.zeros_like(heatmap.cpu())
                    )
                    
                    # Mezclar con la imagen original
                    blended = monai.visualize.utils.blend_images(
                        255 * img.cpu()[c_idx:c_idx+1],
                        255 * thresholded_map,
                        alpha=0.2,
                        transparent_background=True
                    )
                    
                    # Visualizar
                    plt.figure(figsize=(15, 10))
                    monai.visualize.matshow3d(
                        volume=blended,
                        frame_dim=-1,
                        channel_dim=0,
                        every_n=2,
                        margin=10,
                        show=False
                    )
                    
                    plt.suptitle(
                        f"Canal {c_name} con mapa de oclusión individual\n" +
                        f"Clase real: {true_class} ({'no_csPCa' if true_class==0 else 'csPCa'}), " +
                        f"Predicción: {pred_class} ({'no_csPCa' if pred_class==0 else 'csPCa'})",
                        fontsize=14
                    )
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(occlusion_dir, f"{c_name}_occlusion_map.png"), dpi=300)
                    plt.close()
                
                print(f"✓ Resultados de oclusión guardados en {occlusion_dir}")
                
            except Exception as e:
                print(f"⚠ Error al procesar mapas de oclusión: {e}")
                import traceback
                traceback.print_exc()
                
    print("\n" + "="*80)
    print(f"ANÁLISIS COMPLETO: {len(selected_samples)} muestras procesadas")
    print("="*80)
    print(f"Resultados guardados en: {results_base_dir}")

def main():
    """Función principal que ejecuta el análisis de interpretabilidad."""
    # Parsear argumentos
    args = parse_arguments()
    
    # Configuración visual y de dispositivo
    plt.style.use('dark_background')
    plt.rcParams['figure.figsize'] = (12, 8)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Utilizando dispositivo: {device}")
    
    # Determinar rutas base
    project_root = args.project_root
    
    # Comprobar y configurar rutas en sys.path
    sys.path.append(project_root)
    sys.path.append(os.path.join(project_root, "train/deep_learning/1_modeling"))
    
    # Directorios base
    model_base_dir = os.path.join(project_root, "artifacts/deep_learning/gland/models/")
    csv_path = os.path.join(project_root, "artifacts/data.csv")
    
    # Comprobar existencia de directorios y archivos clave
    if not os.path.exists(model_base_dir):
        print(f"⚠ Directorio de modelos no encontrado: {model_base_dir}")
        print("Asegúrate de que la ruta del proyecto sea correcta o proporciona --project-root")
        return
    
    if not os.path.exists(csv_path):
        print(f"⚠ Archivo CSV no encontrado: {csv_path}")
        return
    
    # Directorio del modelo específico
    model_dir = os.path.join(model_base_dir, args.model_type)
    if not os.path.exists(model_dir):
        print(f"⚠ Directorio de modelo específico no encontrado: {model_dir}")
        print(f"Modelos disponibles:")
        for dir in glob.glob(os.path.join(model_base_dir, "base-*")):
            print(f" - {os.path.basename(dir)}")
        return
    
    # Crear directorio de resultados
    model_results_dir = os.path.join(args.output_dir, args.model_type)
    os.makedirs(model_results_dir, exist_ok=True)
    print(f"Los resultados se guardarán en: {model_results_dir}")
    
    # Directorios para mapas de oclusión
    occlusion_dir = os.path.join(model_results_dir, "OcclusionSensitivity")
    maps_dir = os.path.join(occlusion_dir, "individual_maps")  # Subcarpeta para mapas individuales
    
    # Cargar modelo y datos de test
    print("Cargando modelo y datos de test...")
    model, test_dataloader, split_used = load_model_and_test_data(
        model_dir=model_dir, 
        csv_path=csv_path, 
        split_to_use=args.split,
        project_root=project_root
    )
    print(f"Modelo cargado. Usando split {split_used} como conjunto de test.")
    print(f"Conjunto de test: {len(test_dataloader)} muestras")
    
    # Calcular mapas de sensibilidad de oclusión si se requieren y no existen
    occlusion_maps_available = False
    if not args.skip_occlusion:
        # Verificar si ya existen los mapas necesarios
        if os.path.exists(os.path.join(occlusion_dir, "aggregated_heatmaps.pth")) and os.path.exists(maps_dir):
            individual_maps = glob.glob(os.path.join(maps_dir, "class*_*.*"))
            if len(individual_maps) > 0:
                occlusion_maps_available = True
                print(f"✓ Mapas de oclusión encontrados: {len(individual_maps)} mapas individuales")
                print(f"✓ Mapa agregado encontrado en {os.path.join(occlusion_dir, 'aggregated_heatmaps.pth')}")
        
        # Si no existen, calcularlos
        if not occlusion_maps_available:
            calculate_occlusion_sensitivity(
                model=model, 
                test_dataloader=test_dataloader, 
                maps_dir=maps_dir,
                occlusion_dir=occlusion_dir
            )
            occlusion_maps_available = True
    else:
        print("Omitiendo cálculo de mapas de oclusión (--skip-occlusion)")
        
    # Aplicar interpretabilidad comprehensiva
    comprehensive_model_interpretation(
        dataloader=test_dataloader,
        model=model,
        model_results_dir=model_results_dir,
        csv_path=csv_path,
        criteria=args.criteria,
        max_samples=args.max_samples,
        max_attempts=args.max_attempts,
        use_gradcam=not args.skip_gradcam,
        use_occlusion=not args.skip_occlusion,
        use_aggregated_maps=not args.skip_aggregated,
        maps_dir=os.path.join(occlusion_dir, "aggregated_heatmaps.pth") if not args.skip_aggregated else None,
        sensitivity_maps_dir=maps_dir if not args.skip_occlusion else None
    )
    
    print("\nAnálisis de interpretabilidad completado!")
    print(f"Todos los resultados están disponibles en: {model_results_dir}")

if __name__ == "__main__":
    main()
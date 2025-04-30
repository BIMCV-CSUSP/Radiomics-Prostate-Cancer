#!/usr/bin/env python
"""
Script para la extracción paralela de características radiómicas de imágenes de resonancia magnética.

Este script procesa imágenes de resonancia magnética multiparamétricas (T2W, ADC, DWI) y extrae características
radiómicas utilizando PyRadiomics, tanto para la glándula prostática completa como para la imagen completa.
Implementa procesamiento en paralelo para optimizar el tiempo de extracción.
"""

import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import SimpleITK as sitk
from radiomics import featureextractor, imageoperations
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Guardar referencia al método original de PyRadiomics que procesa máscaras
# Esta función se modifica temporalmente durante la ejecución según el modo de extracción
original_getMask = imageoperations.getMask

# Definición de rutas de entrada
pre_path = "../../../../" # Ruta relativa al directorio raíz del proyecto
input_csv = "../../../artifacts/data.csv" # CSV con información de pacientes e imágenes

# Configuración de rutas de salida
current_file = os.path.abspath(__file__)
project_root = os.path.abspath(
    os.path.join(current_file,
                 os.pardir,
                 os.pardir,
                 os.pardir,
                 os.pardir)
)
base_dir = os.path.join(project_root, "artifacts", "radiomics")
os.makedirs(base_dir, exist_ok=True)

## Definición de rutas para archivos CSV de salida
t2_features_gland_csv  = os.path.join(base_dir, "features_t2_gland.csv") # Características T2W de glándula
adc_features_gland_csv = os.path.join(base_dir, "features_adc_gland.csv") # Características ADC de glándula
dwi_features_gland_csv = os.path.join(base_dir, "features_dwi_gland.csv") # Características DWI de glándula

t2_features_full_csv   = os.path.join(base_dir, "features_t2_full.csv") # Características T2W de imagen completa
adc_features_full_csv  = os.path.join(base_dir, "features_adc_full.csv") # Características ADC de imagen completa
dwi_features_full_csv  = os.path.join(base_dir, "features_dwi_full.csv") # Características DWI de imagen completa

## Configuración de logging
logger = logging.getLogger("RadiomicsProcessing")
logger.setLevel(logging.INFO)
logger.propagate = False
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Configuración de salida a consola
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# Configuración de salida a archivo
log_path = os.path.join(base_dir, "radiomics_processing.log")
file_handler = logging.FileHandler(log_path, mode='w')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)


##############################################################################
#                         FUNCIONES DE UTILIDAD                              #
##############################################################################

def resample_to_reference(moving_image, reference_image, is_mask=False):
    """
    Remuestrea una imagen hacia el espacio de otra imagen de referencia.
    
    Args:
        moving_image (sitk.Image): Imagen a remuestrear
        reference_image (sitk.Image): Imagen de referencia
        is_mask (bool): Si es True, usa interpolación de vecino más cercano para preservar valores de etiquetas
                        Si es False, usa interpolación lineal para imágenes de intensidad
    
    Returns:
        sitk.Image: Imagen remuestreada al espacio de referencia
    """

    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(reference_image)
    if is_mask:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkLinear)
    return resample.Execute(moving_image)

def bias_field_correction(image_float32, 
                          shrink_factor=4, 
                          control_points=[4, 4, 4]):
    """
    Aplica corrección de campo de sesgo N4 a una imagen.
    
    Args:
        image_float32 (sitk.Image): Imagen en formato float32
        shrink_factor (int): Factor de reducción para acelerar el procesamiento 
        control_points (list): Puntos de control para el algoritmo N4
    
    Returns:
        sitk.Image: Imagen con corrección de campo de sesgo
    """
    # Reducir la imagen para acelerar el procesamiento
    shrinked_image = sitk.Shrink(image_float32, [shrink_factor] * image_float32.GetDimension())
    
    # Configurar y aplicar el filtro N4
    bias_field_filter = sitk.N4BiasFieldCorrectionImageFilter()
    bias_field_filter.SetNumberOfControlPoints(control_points)
    bias_field_filter.UseMaskLabelOff()
    
    bias_field_filter.Execute(shrinked_image)
    
    # Aplicar la corrección a la imagen original
    log_bias_field = bias_field_filter.GetLogBiasFieldAsImage(image_float32)
    bias_corrected_image = image_float32 / sitk.Exp(log_bias_field)
    
    return bias_corrected_image

def preprocess_image(image):
    """
    Aplica preprocesamiento a una imagen: conversión a float32, corrección de campo de sesgo y reducción de ruido.
    
    Args:
        image (sitk.Image): Imagen original
    
    Returns:
        sitk.Image: Imagen preprocesada
    """
    # Conversión a float32 para operaciones numéricas
    image_float32 = sitk.Cast(image, sitk.sitkFloat32)

    # Corrección de campo de sesgo para normalizar intensidades
    bias_corrected_image = bias_field_correction(image_float32)

    # Reducción de ruido mediante difusión anisotrópica
    denoised_image = sitk.CurvatureAnisotropicDiffusion(bias_corrected_image, timeStep=0.01875)

    return denoised_image

def extract_radiomic_features(extractor_local, image_sitk, mask, patient_id, study_id, label_value, mask_type="gland"):
    """
    Extrae características radiómicas de una imagen utilizando PyRadiomics.
    
    Args:
        extractor_local (RadiomicsFeatureExtractor): Objeto extractor configurado
        image_sitk (sitk.Image): Imagen preprocesada
        mask (sitk.Image o str): Máscara o ruta a la máscara
        patient_id (str): ID del paciente
        study_id (str): ID del estudio
        label_value (int): Valor de la etiqueta (0 o 1 para case_csPCa)
        mask_type (str): Tipo de máscara: "gland" para glándula prostática o "full" para imagen completa
    
    Returns:
        dict: Diccionario con características radiómicas y metadatos
    """
        
    if mask_type == "gland":
        # Para extraer características solo de la glándula prostática:
        # 1. Restauramos el comportamiento normal de PyRadiomics (filtra por región de interés)
        imageoperations.getMask = original_getMask

        # 2. Cargamos la máscara si es una ruta, o usamos la máscara ya cargada
        if isinstance(mask, str):
            mask_sitk = sitk.ReadImage(mask)
        else:
            mask_sitk = mask

        # 3. Aseguramos que la máscara tenga la misma geometría que la imagen
        mask_sitk = resample_to_reference(mask_sitk, image_sitk, is_mask=True)

    elif mask_type == "full":
        # Para extraer características de la imagen completa:
        # 1. Reemplazamos la función getMask de PyRadiomics para que no filtre regiones
        #    (devuelve la máscara sin modificar, ignorando el procesamiento normal)
        imageoperations.getMask = lambda mask, **kwargs: mask

        # 2. Creamos una máscara artificial de "unos" que cubre toda la imagen
        #    (esto hace que se analice la imagen completa)
        mask_array = np.ones(sitk.GetArrayFromImage(image_sitk).shape, dtype=np.uint8)
        mask_sitk = sitk.GetImageFromArray(mask_array)

        # 3. Copiamos los metadatos de la imagen para asegurar consistencia espacial
        mask_sitk.CopyInformation(image_sitk)
    else:
        raise ValueError("mask_type debe ser 'gland' o 'full'")

    # Ejecutar extracción de características
    features = extractor_local.execute(image_sitk, mask_sitk)
    
    # Añadir metadatos al diccionario de salida
    out_dict = {
        "patient_id": patient_id,
        "study_id": study_id,
        "label": label_value,
        "mask_type": mask_type
    }
    for k, v in features.items():
        out_dict[k] = v
        
    return out_dict

def process_modality(modality_key, image_rel_path, patient_id, study_id, label_val, gland_mask_image):
    """
    Procesa una modalidad de imagen (T2, ADC, DWI) y extrae características radiómicas.
    
    Args:
        modality_key (str): Modalidad de la imagen ("T2", "ADC", "DWI")
        image_rel_path (str): Ruta relativa a la imagen
        patient_id (str): ID del paciente
        study_id (str): ID del estudio
        label_val (int): Valor de la etiqueta (case_csPCa)
        gland_mask_image (sitk.Image): Máscara de la glándula prostática
    
    Returns:
        tuple: (features_gland, features_full) - Características radiómicas para glándula y para imagen completa
    """

    # Seleccionar el extractor adecuado según la modalidad
    if modality_key == "T2":
        extractor_local = featureextractor.RadiomicsFeatureExtractor("Params_T2w.yaml")
    elif modality_key == "ADC":
        extractor_local = featureextractor.RadiomicsFeatureExtractor("Params_ADC.yaml")
    elif modality_key == "DWI":
        extractor_local = featureextractor.RadiomicsFeatureExtractor("Params_DWI.yaml")
    else:
        raise ValueError("Modalidad desconocida")
    
    modality_img_path = os.path.join(pre_path, image_rel_path)
    if os.path.isfile(modality_img_path) and gland_mask_image is not None:
        try:
            # Cargar y preprocesar la imagen
            image = sitk.ReadImage(modality_img_path)
            preprocessed = preprocess_image(image)

            # Extraer características para la región de la glándula
            feats_gland = extract_radiomic_features(
                extractor_local,
                image_sitk=preprocessed,
                mask=gland_mask_image,
                patient_id=patient_id,
                study_id=study_id,
                label_value=label_val,
                mask_type="gland"
            )

            # Extraer características para la imagen completa
            feats_full = extract_radiomic_features(
                extractor_local,
                image_sitk=preprocessed,
                mask=gland_mask_image,  # Se ignora, pero se requiere un argumento
                patient_id=patient_id,
                study_id=study_id,
                label_value=label_val,
                mask_type="full"
            )
            return feats_gland, feats_full
        except Exception as e:
            logger.error(f"Error procesando {modality_key} para paciente {patient_id}: {e}", exc_info=True)
            return None, None
    else:
        logger.warning(f"Imagen {modality_key} no encontrada o máscara no disponible para paciente {patient_id}")
        return None, None

def process_row(row):
    """
    Procesa una fila del CSV y extrae las características radiómicas para todas las modalidades.
    
    Args:
        row (pd.Series): Fila del DataFrame con datos del paciente
    
    Returns:
        dict: Diccionario con las características extraídas para cada modalidad y tipo de máscara
    """

    results = {}
    patient_id = row["patient_id"]
    study_id = row["study_id"]
    label_val = row["case_csPCa"]
    gland_mask_path = os.path.join(pre_path, row["whole_gland_path"])

    # Inicializar resultados para todas las combinaciones
    for k in ["t2_gland", "adc_gland", "dwi_gland", 
              "t2_full", "adc_full", "dwi_full"]:
        results[k] = None

    # Cargar máscara de la glándula prostática
    if os.path.isfile(gland_mask_path):
        try:
            gland_mask_image = sitk.ReadImage(gland_mask_path)
        except Exception as e:
            logger.error(f"Error leyendo la máscara para paciente {patient_id}: {e}", exc_info=True)
            gland_mask_image = None
    else:
        logger.warning(f"Archivo de máscara no existe para paciente {patient_id}")
        gland_mask_image = None

    # Procesar cada modalidad
    # Procesar T2
    results["t2_gland"] , results["t2_full"] = process_modality("T2", row["t2w_path"],
                                                patient_id, study_id, label_val,
                                                gland_mask_image)
                                                               
    # Procesar ADC 
    results["adc_gland"], results["adc_full"] = process_modality("ADC", row["adc_path"],
                                                patient_id, study_id, label_val,
                                                gland_mask_image) 

    # Procesar DWI 
    results["dwi_gland"], results["dwi_full"] = process_modality("DWI", row["hbv_path"],
                                                patient_id, study_id, label_val,
                                                gland_mask_image)

    return results

def main():
    """
    Función principal que coordina la extracción de características radiómicas en paralelo.
    Lee los datos de entrada, procesa cada fila en paralelo y guarda los resultados en CSV.
    """
        
    logger.info("Iniciando procesamiento de características radiómicas")

    # Cargar el CSV con datos de pacientes
    df = pd.read_csv(input_csv)
    
    # Inicializar listas para almacenar resultados
    t2_features_gland  = []
    t2_features_full   = []
    adc_features_gland = []
    adc_features_full  = []
    dwi_features_gland = []
    dwi_features_full  = []
    
    # Configurar procesamiento en paralelo
    max_workers = min(4, multiprocessing.cpu_count())
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        # Crear tareas para cada fila
        futures = {executor.submit(process_row, row): idx for idx, row in df.iterrows()}
        
        # Procesar resultados a medida que se completan
        with tqdm(total=len(futures), desc="Procesando imágenes") as pbar:
            for future in as_completed(futures):
                row_result = future.result()
                
                # Recopilar resultados por modalidad y tipo de máscara
                if row_result.get("t2_gland") is not None:
                    t2_features_gland.append(row_result["t2_gland"])
                if row_result.get("t2_full") is not None:
                    t2_features_full.append(row_result["t2_full"])
                
                if row_result.get("adc_gland") is not None:
                    adc_features_gland.append(row_result["adc_gland"])
                if row_result.get("adc_full") is not None:
                    adc_features_full.append(row_result["adc_full"])
                
                if row_result.get("dwi_gland") is not None:
                    dwi_features_gland.append(row_result["dwi_gland"])
                if row_result.get("dwi_full") is not None:
                    dwi_features_full.append(row_result["dwi_full"])
                
                pbar.update(1)  # Actualizar barra de progreso
    
    # Guardar resultados como archivos CSV
    pd.DataFrame(t2_features_gland).to_csv(t2_features_gland_csv, index=False)
    pd.DataFrame(t2_features_full).to_csv(t2_features_full_csv, index=False)
    pd.DataFrame(adc_features_gland).to_csv(adc_features_gland_csv, index=False)
    pd.DataFrame(adc_features_full).to_csv(adc_features_full_csv, index=False)
    pd.DataFrame(dwi_features_gland).to_csv(dwi_features_gland_csv, index=False)
    pd.DataFrame(dwi_features_full).to_csv(dwi_features_full_csv, index=False)

    logger.info("Extracción completada. Archivos CSV generados.")

if __name__ == "__main__":
    main()
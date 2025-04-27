#!/usr/bin/env python
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import SimpleITK as sitk
from radiomics import featureextractor, imageoperations
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

original_getMask = imageoperations.getMask

# Input 
pre_path = "../../../../"
input_csv = "../../../artifacts/data.csv"

# Outputs

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

## CSVs

t2_features_gland_csv  = os.path.join(base_dir, "features_t2_gland.csv")
adc_features_gland_csv = os.path.join(base_dir, "features_adc_gland.csv")
dwi_features_gland_csv = os.path.join(base_dir, "features_dwi_gland.csv")

t2_features_full_csv   = os.path.join(base_dir, "features_t2_full.csv")
adc_features_full_csv  = os.path.join(base_dir, "features_adc_full.csv")
dwi_features_full_csv  = os.path.join(base_dir, "features_dwi_full.csv")

## Logs

logger = logging.getLogger("RadiomicsProcessing")
logger.setLevel(logging.INFO)
logger.propagate = False

formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

# Console handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# File handler en artifacts/radiomics
log_path = os.path.join(base_dir, "radiomics_processing.log")
file_handler = logging.FileHandler(log_path, mode='w')
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

def resample_to_reference(moving_image, reference_image, is_mask=False):
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

    shrinked_image = sitk.Shrink(image_float32, [shrink_factor] * image_float32.GetDimension())
    
    bias_field_filter = sitk.N4BiasFieldCorrectionImageFilter()
    bias_field_filter.SetNumberOfControlPoints(control_points)
    bias_field_filter.UseMaskLabelOff()
    
    bias_field_filter.Execute(shrinked_image)
    
    log_bias_field = bias_field_filter.GetLogBiasFieldAsImage(image_float32)
    bias_corrected_image = image_float32 / sitk.Exp(log_bias_field)
    
    return bias_corrected_image

def preprocess_image(image):
    image_float32 = sitk.Cast(image, sitk.sitkFloat32)
    bias_corrected_image = bias_field_correction(image_float32)
    denoised_image = sitk.CurvatureAnisotropicDiffusion(bias_corrected_image, timeStep=0.01875)
    return denoised_image

def extract_radiomic_features(extractor_local, image_sitk, mask, patient_id, study_id, label_value, mask_type="gland"):
    if mask_type == "gland":
        # Restaurar el comportamiento original para 'gland'
        imageoperations.getMask = original_getMask
        if isinstance(mask, str):
            mask_sitk = sitk.ReadImage(mask)
        else:
            mask_sitk = mask
        mask_sitk = resample_to_reference(mask_sitk, image_sitk, is_mask=True)
    elif mask_type == "full":
        # Para 'full' forzamos que getMask devuelva la máscara sin modificaciones
        imageoperations.getMask = lambda mask, **kwargs: mask
        mask_array = np.ones(sitk.GetArrayFromImage(image_sitk).shape, dtype=np.uint8)
        mask_sitk = sitk.GetImageFromArray(mask_array)
        mask_sitk.CopyInformation(image_sitk)
    else:
        raise ValueError("mask_type debe ser 'gland' o 'full'")

    features = extractor_local.execute(image_sitk, mask_sitk)
    
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
            image = sitk.ReadImage(modality_img_path)
            preprocessed = preprocess_image(image)
            feats_gland = extract_radiomic_features(
                extractor_local,
                image_sitk=preprocessed,
                mask=gland_mask_image,
                patient_id=patient_id,
                study_id=study_id,
                label_value=label_val,
                mask_type="gland"
            )
            feats_full = extract_radiomic_features(
                extractor_local,
                image_sitk=preprocessed,
                mask=gland_mask_image,  
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
    Procesa una fila del CSV y extrae las características radiómicas para cada modalidad:
      - T2
      - ADC
      - DWI
    """
    results = {}
    patient_id = row["patient_id"]
    study_id = row["study_id"]
    label_val = row["case_csPCa"]
    gland_mask_path = os.path.join(pre_path, row["whole_gland_path"])

    for k in ["t2_gland", "adc_gland", "dwi_gland", 
              "t2_full", "adc_full", "dwi_full"]:
        results[k] = None

    if os.path.isfile(gland_mask_path):
        try:
            gland_mask_image = sitk.ReadImage(gland_mask_path)
        except Exception as e:
            logger.error(f"Error leyendo la máscara para paciente {patient_id}: {e}", exc_info=True)
            gland_mask_image = None
    else:
        logger.warning(f"Archivo de máscara no existe para paciente {patient_id}")
        gland_mask_image = None

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
    logger.info("Iniciando procesamiento de características radiómicas")
    # df = pd.read_csv(input_csv)
    df = pd.read_csv(input_csv).head(5)
    
    t2_features_gland  = []
    t2_features_full   = []
    adc_features_gland = []
    adc_features_full  = []
    dwi_features_gland = []
    dwi_features_full  = []
    
    max_workers = min(4, multiprocessing.cpu_count())
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(process_row, row): idx for idx, row in df.iterrows()}
        
        with tqdm(total=len(futures), desc="Procesando imágenes") as pbar:
            for future in as_completed(futures):
                row_result = future.result()
                
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
                
                pbar.update(1)
    
    pd.DataFrame(t2_features_gland).to_csv(t2_features_gland_csv, index=False)
    pd.DataFrame(t2_features_full).to_csv(t2_features_full_csv, index=False)
    pd.DataFrame(adc_features_gland).to_csv(adc_features_gland_csv, index=False)
    pd.DataFrame(adc_features_full).to_csv(adc_features_full_csv, index=False)
    pd.DataFrame(dwi_features_gland).to_csv(dwi_features_gland_csv, index=False)
    pd.DataFrame(dwi_features_full).to_csv(dwi_features_full_csv, index=False)

    logger.info("Extracción completada. Archivos CSV generados.")

if __name__ == "__main__":
    main()
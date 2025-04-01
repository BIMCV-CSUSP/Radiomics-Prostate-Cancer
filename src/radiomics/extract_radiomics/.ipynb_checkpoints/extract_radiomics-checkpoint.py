#!/usr/bin/env python
import os
import pandas as pd
import numpy as np
from tqdm import tqdm
import SimpleITK as sitk
from radiomics import featureextractor
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed

logging.getLogger('radiomics').setLevel(logging.CRITICAL)

input_csv = "../../../data/data.csv"
pre_path = "../../../"

t2_features_gland_csv  = "features_t2_gland.csv"
adc_features_gland_csv = "features_adc_gland.csv"
dwi_features_gland_csv = "features_dwi_gland.csv"

t2_features_full_csv        = "features_t2_full.csv"
adc_features_full_csv       = "features_adc_full.csv"
dwi_features_full_csv       = "features_dwi_full.csv"

extractor = featureextractor.RadiomicsFeatureExtractor()
extractor.settings['geometryTolerance'] = 1e-4
extractor.settings['binWidth'] = 25

def resample_to_reference(moving_image, reference_image, is_mask=False):
    """
    Reamuestra 'moving_image' (sitk.Image) a la geometría de 'reference_image'.
    Si is_mask es True, usa interpolación NearestNeighbor (para máscaras);
    de lo contrario, usa interpolación Linear.
    """
    resample = sitk.ResampleImageFilter()
    resample.SetReferenceImage(reference_image)
    
    if is_mask:
        resample.SetInterpolator(sitk.sitkNearestNeighbor)
    else:
        resample.SetInterpolator(sitk.sitkLinear)
    
    resampled = resample.Execute(moving_image)
    return resampled

def preprocess_image(image):
    """
    Aplica preprocesamiento a la imagen:
      - Denoising mediante filtro de flujo de curvatura.
      - Corrección de no uniformidad (bias) usando N4BiasFieldCorrection.
    """
    # 1. Denoising: aplicamos un filtro de flujo de curvatura.
    curvatureFilter = sitk.CurvatureFlowImageFilter()
    curvatureFilter.SetTimeStep(0.125)
    curvatureFilter.SetNumberOfIterations(5)
    denoised = curvatureFilter.Execute(image)
    
    # 2. Corrección de no uniformidad:
    # Se genera una máscara inicial con un umbral de Otsu y se aplica N4.
    mask = sitk.OtsuThreshold(denoised, 0, 1, 200)
    corrected = sitk.N4BiasFieldCorrection(denoised, mask)
    
    return corrected

def extract_radiomic_features_from_sitk(image_sitk, mask_path, patient_id, study_id, label_value, mask_type="gland"):
    """
    Extrae características radiómicas a partir de una imagen (sitk.Image) ya preprocesada.
    Se puede extraer sobre:
      - La glándula (mask_type="gland"): se utiliza la máscara proporcionada.
      - La imagen completa (mask_type="full"): se crea una máscara de 1’s.
    Las partes correspondientes a "full" se dejan comentadas en otros lugares.
    """
    if mask_type == "gland":
        mask_sitk = sitk.ReadImage(mask_path)
        mask_sitk = resample_to_reference(mask_sitk, image_sitk, is_mask=True)
    elif mask_type == "full":
        mask_sitk = sitk.Image(image_sitk.GetSize(), sitk.sitkUInt8)
        mask_sitk.CopyInformation(image_sitk)
        mask_sitk = sitk.Add(mask_sitk, 1)
    else:
        raise ValueError("mask_type debe ser 'gland' o 'full'")
    
    features = extractor.execute(image_sitk, mask_sitk)
    
    out_dict = {
        "patient_id": patient_id,
        "study_id": study_id,
        "label": label_value,
        "mask_type": mask_type
    }
    
    for k, v in features.items():
        if k.startswith("original_"):
            out_dict[k] = v
    
    return out_dict

def extract_radiomic_features(image_path, mask_path, patient_id, study_id, label_value, mask_type="gland"):
    """
    Carga la imagen desde 'image_path', le aplica preprocesamiento y extrae las características.
    """
    image_sitk = sitk.ReadImage(image_path)
    image_sitk = preprocess_image(image_sitk)
    
    return extract_radiomic_features_from_sitk(image_sitk, mask_path, patient_id, study_id, label_value, mask_type)

def process_row(row):
    """
    Procesa una fila del CSV y extrae las características radiómicas para cada modalidad:
      - T2
      - ADC
      - DWI
    Devuelve un diccionario con las claves:
       "t2_gland", "adc_gland", "dwi_gland"
    """
    results = {}
    patient_id = row["patient_id"]
    study_id = row["study_id"]
    label_val = row["case_csPCa"]
    gland_mask_path = os.path.join(pre_path, row["whole_gland_path"])
    
    results["t2_gland"] = None
    results["adc_gland"] = None
    results["dwi_gland"] = None
    results["t2_full"] = None
    results["adc_full"] = None
    results["dwi_full"] = None
    
    # Procesamiento de T2:
    t2_img_path = os.path.join(pre_path, row["t2w_path"])
    if os.path.isfile(t2_img_path) and os.path.isfile(gland_mask_path):
        try:
            feats_t2_gland = extract_radiomic_features(
                image_path=t2_img_path,
                mask_path=gland_mask_path,
                patient_id=patient_id,
                study_id=study_id,
                label_value=label_val,
                mask_type="gland"
            )
            results["t2_gland"] = feats_t2_gland
            
            feats_t2_full = extract_radiomic_features(
                image_path=t2_img_path,
                mask_path=gland_mask_path,
                patient_id=patient_id,
                study_id=study_id,
                label_value=label_val,
                mask_type="full"
            )
            results["t2_full"] = feats_t2_full
        except Exception as e:
            print(f"Error procesando T2 para paciente {patient_id}: {e}")
    
    # Procesamiento de ADC:
    adc_img_path = os.path.join(pre_path, row["adc_path"])
    if os.path.isfile(adc_img_path) and os.path.isfile(gland_mask_path):
        try:
            feats_adc_gland = extract_radiomic_features(
                image_path=adc_img_path,
                mask_path=gland_mask_path,
                patient_id=patient_id,
                study_id=study_id,
                label_value=label_val,
                mask_type="gland"
            )
            results["adc_gland"] = feats_adc_gland
            
            feats_adc_full = extract_radiomic_features(
                image_path=adc_img_path,
                mask_path=gland_mask_path,
                patient_id=patient_id,
                study_id=study_id,
                label_value=label_val,
                mask_type="full"
            )
            results["adc_full"] = feats_adc_full
        except Exception as e:
            print(f"Error procesando ADC para paciente {patient_id}: {e}")
    
    # Procesamiento de DWI:
    dwi_img_path = os.path.join(pre_path, row["hbv_path"])
    if os.path.isfile(dwi_img_path) and os.path.isfile(gland_mask_path):
        try:
            feats_dwi_gland = extract_radiomic_features(
                image_path=dwi_img_path,
                mask_path=gland_mask_path,
                patient_id=patient_id,
                study_id=study_id,
                label_value=label_val,
                mask_type="gland"
            )
            results["dwi_gland"] = feats_dwi_gland
            
            feats_dwi_full = extract_radiomic_features(
                image_path=dwi_img_path,
                mask_path=gland_mask_path,
                patient_id=patient_id,
                study_id=study_id,
                label_value=label_val,
                mask_type="full"
            )
            results["dwi_full"] = feats_dwi_full
        except Exception as e:
            print(f"Error procesando DWI para paciente {patient_id}: {e}")
    
    return results

def main():
    df = pd.read_csv(input_csv)
    df = df.sample(n=25, random_state=42)
    
    t2_features_gland  = []
    t2_features_full   = []
    adc_features_gland = []
    adc_features_full  = []
    dwi_features_gland = []
    dwi_features_full  = []
    
    with ProcessPoolExecutor() as executor:
        futures = {executor.submit(process_row, row): idx for idx, row in df.iterrows()}
        with tqdm(total=len(futures), desc="Procesando imágenes") as pbar:
            for future in as_completed(futures):
                try:
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
                except Exception as e:
                    print(f"Error procesando una fila: {e}")
                pbar.update(1)
    
    # Se guardan los resultados en archivos CSV.
    pd.DataFrame(t2_features_gland).to_csv(t2_features_gland_csv, index=False)
    pd.DataFrame(t2_features_full).to_csv(t2_features_full_csv, index=False)
    pd.DataFrame(adc_features_gland).to_csv(adc_features_gland_csv, index=False)
    pd.DataFrame(adc_features_full).to_csv(adc_features_full_csv, index=False)
    pd.DataFrame(dwi_features_gland).to_csv(dwi_features_gland_csv, index=False)
    pd.DataFrame(dwi_features_full).to_csv(dwi_features_full_csv, index=False)

if __name__ == "__main__":
    main()
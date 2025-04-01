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

extractor = featureextractor.RadiomicsFeatureExtractor("Params.yaml")
extractor.settings['geometryTolerance'] = 1e-4

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

def bias_field_correction(image_float32, shrink_factor=4, control_points=[4, 4, 4]):
    shrinked_image = sitk.Shrink(image_float32, [shrink_factor] * image_float32.GetDimension())
    
    bias_field_filter = sitk.N4BiasFieldCorrectionImageFilter()
    bias_field_filter.SetNumberOfControlPoints(control_points)
    bias_field_filter.UseMaskLabelOff()
    
    _ = bias_field_filter.Execute(shrinked_image)
    
    log_bias_field = bias_field_filter.GetLogBiasFieldAsImage(image_float32)
    
    bias_corrected_image = image_float32 / sitk.Exp(log_bias_field)
    
    return bias_corrected_image

def preprocess_image(image):
    """
    Aplica preprocesamiento a la imagen
    """
    image_float32 = sitk.Cast(image, sitk.sitkFloat32)
    bias_corrected_image = bias_field_correction(image_float32)
    denoised_image = sitk.CurvatureAnisotropicDiffusion(bias_corrected_image, timeStep=0.03125)
    return denoised_image

def extract_radiomic_features_from_sitk(image_sitk, mask, patient_id, study_id, label_value, mask_type="gland"):
    """
    Extrae características radiómicas a partir de una imagen ya preprocesada (sitk.Image).
    Se puede extraer sobre:
      - La glándula (mask_type="gland"): se utiliza la máscara proporcionada.
      - La imagen completa (mask_type="full"): se crea una máscara de 1’s.
    
    El parámetro 'mask' puede ser una ruta (str) o un objeto sitk.Image.
    """
    if mask_type == "gland":
        if isinstance(mask, str):
            mask_sitk = sitk.ReadImage(mask)
        else:
            mask_sitk = mask
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
        out_dict[k] = v
    
    return out_dict

def process_row(row):
    """
    Procesa una fila del CSV y extrae las características radiómicas para cada modalidad:
      - T2
      - ADC
      - DWI

    Para cada modalidad, se carga la imagen, se preprocesa una sola vez y se extraen las
    características usando la máscara de glándula (gland) y la máscara completa (full).

    Devuelve un diccionario con las claves:
       "t2_gland", "adc_gland", "dwi_gland", "t2_full", "adc_full", "dwi_full"
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

    if os.path.isfile(gland_mask_path):
        try:
            gland_mask_image = sitk.ReadImage(gland_mask_path)
        except Exception as e:
            print(f"Error leyendo la máscara para paciente {patient_id}: {e}")
            gland_mask_image = None
    else:
        gland_mask_image = None

    def process_modality(modality_key, image_rel_path):
        modality_img_path = os.path.join(pre_path, image_rel_path)
        if os.path.isfile(modality_img_path) and gland_mask_image is not None:
            try:
                image = sitk.ReadImage(modality_img_path)
                preprocessed = preprocess_image(image)
                feats_gland = extract_radiomic_features_from_sitk(
                    image_sitk=preprocessed,
                    mask=gland_mask_image,
                    patient_id=patient_id,
                    study_id=study_id,
                    label_value=label_val,
                    mask_type="gland"
                )
                feats_full = extract_radiomic_features_from_sitk(
                    image_sitk=preprocessed,
                    mask=gland_mask_image,  # No se usa para "full"
                    patient_id=patient_id,
                    study_id=study_id,
                    label_value=label_val,
                    mask_type="full"
                )
                return feats_gland, feats_full
            except Exception as e:
                print(f"Error procesando {modality_key} para paciente {patient_id}: {e}")
                return None, None
        else:
            return None, None

    # T2
    feats_t2_gland, feats_t2_full = process_modality("T2", row["t2w_path"])
    results["t2_gland"] = feats_t2_gland
    results["t2_full"] = feats_t2_full

    # ADC
    feats_adc_gland, feats_adc_full = process_modality("ADC", row["adc_path"])
    results["adc_gland"] = feats_adc_gland
    results["adc_full"] = feats_adc_full

    # DWI
    feats_dwi_gland, feats_dwi_full = process_modality("DWI", row["hbv_path"])
    results["dwi_gland"] = feats_dwi_gland
    results["dwi_full"] = feats_dwi_full

    return results

def main():
    df = pd.read_csv(input_csv)
    df = df.sample(n=1, random_state=42)
    
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
    
    pd.DataFrame(t2_features_gland).to_csv(t2_features_gland_csv, index=False)
    pd.DataFrame(t2_features_full).to_csv(t2_features_full_csv, index=False)
    pd.DataFrame(adc_features_gland).to_csv(adc_features_gland_csv, index=False)
    pd.DataFrame(adc_features_full).to_csv(adc_features_full_csv, index=False)
    pd.DataFrame(dwi_features_gland).to_csv(dwi_features_gland_csv, index=False)
    pd.DataFrame(dwi_features_full).to_csv(dwi_features_full_csv, index=False)

if __name__ == "__main__":
    main()
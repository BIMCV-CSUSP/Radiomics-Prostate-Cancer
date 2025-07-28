#!/usr/bin/env python
"""
Script para la extracción paralela de características radiómicas de imágenes de resonancia magnética.

Este script procesa imágenes de resonancia magnética multiparamétricas (T2W, ADC, DWI) y extrae características
radiómicas utilizando PyRadiomics, tanto para la glándula prostática completa como para la imagen completa.
Implementa procesamiento en paralelo para optimizar el tiempo de extracción, y guarda resultados intermedios
para poder reanudar en caso de interrupción.
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

# ── 1) Per-worker globals ───────────────────────────────────────────────────────
EXTRACTOR_T2  = None
EXTRACTOR_ADC = None
EXTRACTOR_DWI = None

def init_worker():
    """Inicializa cada worker con sus extractores (solo se carga una vez)."""
    global EXTRACTOR_T2, EXTRACTOR_ADC, EXTRACTOR_DWI
    EXTRACTOR_T2  = featureextractor.RadiomicsFeatureExtractor(
        "/home/jaalzate/Radiomics-Prostate-Cancer/train/radiomics/1_extract_radiomics/Params_T2w.yaml"
    )
    EXTRACTOR_ADC = featureextractor.RadiomicsFeatureExtractor(
        "/home/jaalzate/Radiomics-Prostate-Cancer/train/radiomics/1_extract_radiomics/Params_T2w.yaml"
    )
    EXTRACTOR_DWI = featureextractor.RadiomicsFeatureExtractor(
        "/home/jaalzate/Radiomics-Prostate-Cancer/train/radiomics/1_extract_radiomics/Params_DWI.yaml"
    )

# Referencia al método original de getMask
original_getMask = imageoperations.getMask

# ── 2) Rutas y checkpoint ──────────────────────────────────────────────────────
pre_path      = "../../../../"
input_csv     = "/home/jaalzate/Radiomics-Prostate-Cancer/artifacts/bimcv_data/BIMCV_with_predictions_filtered_volume copy.csv"

# directorio de salida
base_dir      = os.path.abspath(
    os.path.join(os.path.dirname(__file__), *([".."]*3), "artifacts", "radiomics")
)
os.makedirs(base_dir, exist_ok=True)

# CSVs de salida
t2_gland_csv  = os.path.join(base_dir, "features_t2_gland_BIMCV.csv")
print(t2_gland_csv)
adc_gland_csv = os.path.join(base_dir, "features_adc_gland_BIMCV.csv")
dwi_gland_csv = os.path.join(base_dir, "features_dwi_gland_BIMCV.csv")
t2_full_csv   = os.path.join(base_dir, "features_t2_full_BIMCV.csv")
adc_full_csv  = os.path.join(base_dir, "features_adc_full_BIMCV.csv")
dwi_full_csv  = os.path.join(base_dir, "features_dwi_full_BIMCV.csv")

# archivo de checkpoint: lista de patient_study ya procesados
checkpoint_file = os.path.join(base_dir, "processed_rows_BIMCV.txt")

# ── 3) Logging ─────────────────────────────────────────────────────────────────
logger = logging.getLogger("RadiomicsProcessing")
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch  = logging.StreamHandler(); ch.setFormatter(fmt); logger.addHandler(ch)
fh  = logging.FileHandler(os.path.join(base_dir, "radiomics_processing.log"), mode='a')
fh.setFormatter(fmt); logger.addHandler(fh)


# ── 4) Utilidades ──────────────────────────────────────────────────────────────

def resample_to_reference(moving, reference, is_mask=False):
    rf = sitk.ResampleImageFilter()
    rf.SetReferenceImage(reference)
    rf.SetInterpolator(sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear)
    return rf.Execute(moving)

def bias_field_correction(image_float32, shrink_factor=4, control_points=[4,4,4]):
    shrinked = sitk.Shrink(image_float32, [shrink_factor]*image_float32.GetDimension())
    n4 = sitk.N4BiasFieldCorrectionImageFilter()
    n4.SetNumberOfControlPoints(control_points)
    n4.UseMaskLabelOff()
    n4.Execute(shrinked)
    log_bias = n4.GetLogBiasFieldAsImage(image_float32)
    return image_float32 / sitk.Exp(log_bias)

# ── 5) Preprocesamiento (sin cambios) ─────────────────────────────────────────

def preprocess_image(image):
    if image.GetPixelID() != sitk.sitkFloat32:
        image = sitk.Cast(image, sitk.sitkFloat32)
    bias_corr = bias_field_correction(image)
    return sitk.CurvatureAnisotropicDiffusion(bias_corr, timeStep=0.01875)

def preprocess_scalar_image(img):
    return preprocess_image(img)

def preprocess_image_wrapper(image):
    """Divide DWI 4D en canales 3D, aplica preprocess_image a cada uno, recompone."""
    if image.GetDimension() == 4 and image.GetNumberOfComponentsPerPixel() == 1:
        size = list(image.GetSize())   # [X, Y, Z, T]
        size[3] = 0

        # Índice de inicio: extraer t=0
        index = [0, 0, 0, 0]

        # Extraemos el primer volumen 3D
        first_vol = sitk.Extract(image, size, index)

        # Lo procesamos como imagen escalar 3D
        return preprocess_scalar_image(first_vol)
    else:
        return preprocess_scalar_image(image)

# ── 6) Extracción ─────────────────────────────────────────────────────────────

def extract_radiomic_features(extractor, img, mask, pid, sid, label, mask_type="gland"):
    if mask_type == "gland":
        imageoperations.getMask = original_getMask
        if isinstance(mask, str):
            m = sitk.ReadImage(mask)
            m = resample_to_reference(m, img, is_mask=True)
        else:
            m = mask
    else:  # full
        imageoperations.getMask = lambda mk, **kw: mk
        arr = np.ones(sitk.GetArrayFromImage(img).shape, dtype=np.uint8)
        m   = sitk.GetImageFromArray(arr)
        m.CopyInformation(img)

    feats = extractor.execute(img, m)
    out = {
        "patient_id": pid,
        "study_id":   sid,
        "label":      label,
        "mask_type":  mask_type,
        **feats
    }
    return out

def process_modality(modality, rel_path, pid, sid, label, gland_mask):
    if modality == "T2":
        ext = EXTRACTOR_T2
    elif modality == "ADC":
        ext = EXTRACTOR_ADC
    elif modality == "DWI":
        ext = EXTRACTOR_DWI
    else:
        raise ValueError("Modalidad desconocida")

    img_path = os.path.join(pre_path, rel_path)
    if not os.path.isfile(img_path) or gland_mask is None:
        logger.warning(f"{modality} faltante para {pid}")
        return None, None

    try:
        # img = sitk.ReadImage(img_path)
        img = sitk.ReadImage(img_path, sitk.sitkFloat32)
        pre = preprocess_image_wrapper(img)
        # remuestro UNA sola vez
        gland_res = resample_to_reference(gland_mask, pre, is_mask=True)

        f_g = extract_radiomic_features(ext, pre, gland_res, pid, sid, label, mask_type="gland")
        # f_f = extract_radiomic_features(ext, pre, gland_res, pid, sid, label, mask_type="full")
        return f_g

    except Exception as e:
        logger.error(f"Error en {modality} {pid}: {e}", exc_info=True)
        return None, None

def process_row(row):
    pid, sid, lbl = row["patient_id"], row["study_id"], row["case_csPCa"]
    mask_path = os.path.join(pre_path, row["whole_gland_path"])
    gland_mask = None
    if os.path.isfile(mask_path):
        try:
            gland_mask = sitk.ReadImage(mask_path)
        except Exception as e:
            logger.error(f"Error leyendo máscara {pid}: {e}", exc_info=True)

    out = {}
    out["t2_gland"] = process_modality("T2",  row["t2w_path"], pid, sid, lbl, gland_mask)
    out["adc_gland"] = process_modality("ADC", row["adc_path"], pid, sid, lbl, gland_mask)
    out["dwi_gland"] = process_modality("DWI", row["hbv_path"], pid, sid, lbl, gland_mask)
    return out

# ── 7) Main con checkpoint y append ──────────────────────────────────────────

def main():
    logger.info("Iniciando extracción radiómica con checkpoint")

    df = pd.read_csv(input_csv, dtype=str)
    # crear clave única por fila
    df["row_key"] = df["patient_id"] + "_" + df["study_id"]

    # cargar ya procesados
    processed = set()
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file) as f:
            processed = set(line.strip() for line in f if line.strip())

    to_do = df[~df["row_key"].isin(processed)]
    logger.info(f"{len(to_do)} filas pendientes de {len(df)} totales")

    # ejecutor paralelo
    max_workers = min(4, multiprocessing.cpu_count())
    with ProcessPoolExecutor(max_workers=max_workers, initializer=init_worker) as exe, \
         open(checkpoint_file, "a") as ckpt:

        futures = {
            exe.submit(process_row, row): row["row_key"]
            for _, row in to_do.iterrows()
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Procesando filas"):
            key = futures[future]
            res = future.result()

            # para cada modalidad, si no es None → append al CSV correspondiente
            if res["t2_gland"]:
                pd.DataFrame([res["t2_gland"]]).to_csv(
                    t2_gland_csv, mode="a", header=not os.path.exists(t2_gland_csv), index=False
                )
                # pd.DataFrame([res["t2_full"]]).to_csv(
                #     t2_full_csv, mode="a", header=not os.path.exists(t2_full_csv), index=False
                # )
                pd.DataFrame([res["adc_gland"]]).to_csv(
                    adc_gland_csv, mode="a", header=not os.path.exists(adc_gland_csv), index=False
                )
                # pd.DataFrame([res["adc_full"]]).to_csv(
                #     adc_full_csv, mode="a", header=not os.path.exists(adc_full_csv), index=False
                # )
                pd.DataFrame([res["dwi_gland"]]).to_csv(
                    dwi_gland_csv, mode="a", header=not os.path.exists(dwi_gland_csv), index=False
                )
                # pd.DataFrame([res["dwi_full"]]).to_csv(
                #     dwi_full_csv, mode="a", header=not os.path.exists(dwi_full_csv), index=False
                # )

            # marco esta fila como completada
            ckpt.write(key + "\n")
            ckpt.flush()

    logger.info("Proceso terminado. Puede reiniciar sin reprocesar lo ya completado.")

if __name__ == "__main__":
    main()

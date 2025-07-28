#!/usr/bin/env python
"""
Script optimizado para la extracción paralela de características radiómicas de imágenes de resonancia magnética,
con cálculo correcto de tiempos por modalidad, evitando errores de serialización.
"""

import os
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
import SimpleITK as sitk
from radiomics import featureextractor, imageoperations
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# ── 1) Per-worker globals ───────────────────────────────────────────────────────
EXTRACTOR_T2 = None
EXTRACTOR_ADC = None
EXTRACTOR_DWI = None
original_getMask = imageoperations.getMask

# ── 2) Rutas y checkpoint ──────────────────────────────────────────────────────
pre_path = "../../../../"
input_csv = "/home/jaalzate/Radiomics-Prostate-Cancer/artifacts/bimcv_data/BIMCV_with_predictions_filtered_volume copy.csv"

base_dir = os.path.abspath(
    os.path.join(os.path.dirname(__file__), *(['..']*3), 'artifacts', 'radiomics')
)
os.makedirs(base_dir, exist_ok=True)

# CSVs de salida (gland features)
t2_gland_csv = os.path.join(base_dir, "features_t2_gland_BIMCV.csv")
adc_gland_csv = os.path.join(base_dir, "features_adc_gland_BIMCV.csv")
dwi_gland_csv = os.path.join(base_dir, "features_dwi_gland_BIMCV.csv")

checkpoint_file = os.path.join(base_dir, "processed_rows_BIMCV.txt")

# ── 3) Logging ─────────────────────────────────────────────────────────────────
logger = logging.getLogger("RadiomicsProcessing")
logger.setLevel(logging.INFO)
fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch = logging.StreamHandler(); ch.setFormatter(fmt); logger.addHandler(ch)
fh = logging.FileHandler(os.path.join(base_dir, "radiomics_processing.log"), mode='a')
fh.setFormatter(fmt); logger.addHandler(fh)

# ── 4) Utilities ──────────────────────────────────────────────────────────────
def resample_to_reference(moving, reference, is_mask=False):
    rf = sitk.ResampleImageFilter()
    rf.SetReferenceImage(reference)
    rf.SetInterpolator(sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear)
    return rf.Execute(moving)


def bias_field_correction(image, shrink_factor=4, control_points=[4,4,4]):
    shrinked = sitk.Shrink(image, [shrink_factor]*image.GetDimension())
    n4 = sitk.N4BiasFieldCorrectionImageFilter()
    n4.SetNumberOfControlPoints(control_points)
    n4.UseMaskLabelOff()
    n4.Execute(shrinked)
    log_bias = n4.GetLogBiasFieldAsImage(image)
    return image / sitk.Exp(log_bias)


def preprocess_image(img):
    if img.GetPixelID() != sitk.sitkFloat32:
        img = sitk.Cast(img, sitk.sitkFloat32)
    corr = bias_field_correction(img)
    return sitk.CurvatureAnisotropicDiffusion(corr, timeStep=0.01875)


def preprocess_image_wrapper(image):
    # Extrae primer volumen de DWI 4D si aplica
    if image.GetDimension() == 4 and image.GetNumberOfComponentsPerPixel() == 1:
        size = list(image.GetSize()); size[3] = 0
        idx = [0,0,0,0]
        vol0 = sitk.Extract(image, size, idx)
        return preprocess_image(vol0)
    return preprocess_image(image)

# ── 5) Worker initializer ──────────────────────────────────────────────────────
def init_worker():
    global EXTRACTOR_T2, EXTRACTOR_ADC, EXTRACTOR_DWI
    EXTRACTOR_T2 = featureextractor.RadiomicsFeatureExtractor(
        "/home/jaalzate/Radiomics-Prostate-Cancer/train/radiomics/1_extract_radiomics/Params_T2w.yaml"
    )
    EXTRACTOR_ADC = featureextractor.RadiomicsFeatureExtractor(
        "/home/jaalzate/Radiomics-Prostate-Cancer/train/radiomics/1_extract_radiomics/Params_T2w.yaml"
    )
    EXTRACTOR_DWI = featureextractor.RadiomicsFeatureExtractor(
        "/home/jaalzate/Radiomics-Prostate-Cancer/train/radiomics/1_extract_radiomics/Params_DWI.yaml"
    )

# ── 6) Extraction logic ─────────────────────────────────────────────────────────
def extract_radiomic_features(extractor, img, mask, pid, sid, label):
    imageoperations.getMask = original_getMask
    feats = extractor.execute(img, mask)
    return {"patient_id": pid, "study_id": sid, "label": label, **feats}


def process_modality(modality, img_path, gland_mask, pid, sid, label):
    start = time.perf_counter()
    try:
        img = sitk.ReadImage(img_path, sitk.sitkFloat32)
        pre = preprocess_image_wrapper(img)
        resampled_mask = resample_to_reference(gland_mask, pre, is_mask=True)
        extractor = {"T2": EXTRACTOR_T2, "ADC": EXTRACTOR_ADC, "DWI": EXTRACTOR_DWI}[modality]
        feats = extract_radiomic_features(extractor, pre, resampled_mask, pid, sid, label)
    except Exception as e:
        logger.error(f"Error en {modality} {pid}: {e}", exc_info=True)
        feats = None
    duration = time.perf_counter() - start
    return feats, duration


def process_row(record):
    # record es un dict simple, serializable
    pid, sid, lbl = record['patient_id'], record['study_id'], record['case_csPCa']
    mask_f = os.path.join(pre_path, record['whole_gland_path'])
    gland_mask = sitk.ReadImage(mask_f) if os.path.isfile(mask_f) else None

    results = {}
    times = {"T2": 0.0, "ADC": 0.0, "DWI": 0.0}
    for mod, path_key in [("T2","t2w_path"),("ADC","adc_path"),("DWI","hbv_path")]:
        img_p = os.path.join(pre_path, record[path_key])
        if gland_mask is None or not os.path.isfile(img_p):
            logger.warning(f"{mod} faltante para {pid}")
            continue
        feats, dt = process_modality(mod, img_p, gland_mask, pid, sid, lbl)
        if feats:
            results[f"{mod.lower()}_gland"] = feats
        times[mod] = dt

    return record['patient_id'] + '_' + record['study_id'], results, times

# ── 7) Main ───────────────────────────────────────────────────────────────────
def main():
    logger.info("Iniciando extracción radiómica optimizada")
    df = pd.read_csv(input_csv, dtype=str)
    df['row_key'] = df['patient_id'] + '_' + df['study_id']

    processed = set()
    if os.path.exists(checkpoint_file):
        with open(checkpoint_file) as f:
            processed = set(f.read().splitlines())
    to_do_df = df[~df['row_key'].isin(processed)]
    records = to_do_df.to_dict(orient='records')
    logger.info(f"{len(records)} filas pendientes de {len(df)} totales")

    totals = {"T2": 0.0, "ADC": 0.0, "DWI": 0.0}
    max_workers = min(multiprocessing.cpu_count(), len(records), 4)

    with ProcessPoolExecutor(max_workers=max_workers, initializer=init_worker) as exe, \
         open(checkpoint_file, 'a') as ckpt:

        futures = {exe.submit(process_row, rec): rec['row_key'] for rec in records}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Procesando"):  # progress bar
            key, res, times = future.result()
            # write features
            for mod, csv_path in [("T2", t2_gland_csv), ("ADC", adc_gland_csv), ("DWI", dwi_gland_csv)]:
                feat = res.get(f"{mod.lower()}_gland")
                if feat:
                    pd.DataFrame([feat]).to_csv(
                        csv_path, mode='a', header=not os.path.exists(csv_path), index=False
                    )
            # accumulate times and checkpoint
            for m in totals:
                totals[m] += times.get(m, 0.0)
            ckpt.write(key + '\n'); ckpt.flush()

    # reporte de tiempos
    for mod, t in totals.items():
        avg = t / len(records) if records else 0
        logger.info(f"Tiempo total {mod}: {t:.2f}s ({avg:.3f}s/imagen)")
    logger.info("Proceso terminado.")

if __name__ == '__main__':
    main()

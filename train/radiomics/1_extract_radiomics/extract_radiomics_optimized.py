#!/usr/bin/env python
"""
Script optimizado para la extracción paralela de características radiómicas
de imágenes de resonancia magnética de próstata.

Optimizaciones aplicadas:
  - Paralelización interna por modalidad (ThreadPoolExecutor dentro de cada worker)
  - Escritura batch a CSV (reduce I/O de disco)
  - N4 Bias Correction más rápido (mayor shrink factor, menos iteraciones)
  - Preprocesamiento opcional (flag ENABLE_PREPROCESSING)
  - Número de workers adaptado al hardware disponible
  - Cache de máscara resampleada cuando comparten referencia
"""

import os
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
import SimpleITK as sitk
from radiomics import featureextractor, imageoperations
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing

logging.getLogger("radiomics").setLevel(logging.WARNING)
# ═══════════════════════════════════════════════════════════════════════════════
# CONFIGURACIÓN
# ═══════════════════════════════════════════════════════════════════════════════

# Preprocesamiento: desactívalo si no mejora tus resultados (ahorra ~70-80% tiempo)
ENABLE_PREPROCESSING = True

# N4 Bias Field Correction: parámetros rápidos vs conservadores
N4_SHRINK_FACTOR = 6          # default conservador: 4
N4_MAX_ITERATIONS = [50, 50]  # default conservador: [50, 50, 50, 50]
N4_CONTROL_POINTS = [4, 4, 4]

# Paralelización
MAX_WORKERS = None  # None = cpu_count - 1, o fija un número
INNER_THREADS = 3   # hilos para procesar modalidades en paralelo dentro de cada worker

# Escritura batch: acumula N resultados antes de escribir a disco
BATCH_SIZE = 50

# Rutas
PRE_PATH = "../../../../"
INPUT_CSV = "/home/jaalzate/Projects/Radiomics-Prostate-Cancer/artifacts/bimcv_data/BIMCV_with_predictions_filtered_volume.csv"

PARAMS_T2 = "/home/jaalzate/Projects/Radiomics-Prostate-Cancer/train/radiomics/1_extract_radiomics/Params_T2w.yaml"
PARAMS_ADC = "/home/jaalzate/Projects/Radiomics-Prostate-Cancer/train/radiomics/1_extract_radiomics/Params_T2w.yaml"
PARAMS_DWI = "/home/jaalzate/Projects/Radiomics-Prostate-Cancer/train/radiomics/1_extract_radiomics/Params_DWI.yaml"

# ═══════════════════════════════════════════════════════════════════════════════
# GLOBALS POR WORKER (se inicializan en init_worker)
# ═══════════════════════════════════════════════════════════════════════════════
EXTRACTOR_T2 = None
EXTRACTOR_ADC = None
EXTRACTOR_DWI = None
original_getMask = imageoperations.getMask

# ═══════════════════════════════════════════════════════════════════════════════
# RUTAS DE SALIDA Y CHECKPOINT
# ═══════════════════════════════════════════════════════════════════════════════
BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), *(['..'] * 3), 'artifacts', 'radiomics')
)
os.makedirs(BASE_DIR, exist_ok=True)

T2_GLAND_CSV  = os.path.join(BASE_DIR, "features_t2_gland_BIMCV.csv")
ADC_GLAND_CSV = os.path.join(BASE_DIR, "features_adc_gland_BIMCV.csv")
DWI_GLAND_CSV = os.path.join(BASE_DIR, "features_dwi_gland_BIMCV.csv")
CHECKPOINT_FILE = os.path.join(BASE_DIR, "processed_rows_BIMCV.txt")

# ═══════════════════════════════════════════════════════════════════════════════
# LOGGING
# ═══════════════════════════════════════════════════════════════════════════════
logger = logging.getLogger("RadiomicsProcessing")
logger.setLevel(logging.INFO)
_fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

_ch = logging.StreamHandler()
_ch.setFormatter(_fmt)
logger.addHandler(_ch)

_fh = logging.FileHandler(os.path.join(BASE_DIR, "radiomics_processing.log"), mode='a')
_fh.setFormatter(_fmt)
logger.addHandler(_fh)

# ═══════════════════════════════════════════════════════════════════════════════
# UTILIDADES DE PREPROCESAMIENTO
# ═══════════════════════════════════════════════════════════════════════════════

def resample_to_reference(moving, reference, is_mask=False):
    """Resamplea una imagen/máscara al espacio de referencia."""
    rf = sitk.ResampleImageFilter()
    rf.SetReferenceImage(reference)
    rf.SetInterpolator(sitk.sitkNearestNeighbor if is_mask else sitk.sitkLinear)
    return rf.Execute(moving)


def bias_field_correction(image):
    """N4 Bias Field Correction con parámetros optimizados para velocidad."""
    shrinked = sitk.Shrink(image, [N4_SHRINK_FACTOR] * image.GetDimension())
    n4 = sitk.N4BiasFieldCorrectionImageFilter()
    n4.SetNumberOfControlPoints(N4_CONTROL_POINTS)
    n4.SetMaximumNumberOfIterations(N4_MAX_ITERATIONS)
    n4.UseMaskLabelOff()
    n4.Execute(shrinked)
    log_bias = n4.GetLogBiasFieldAsImage(image)
    return image / sitk.Exp(log_bias)


def preprocess_image(img):
    """Pipeline de preprocesamiento: cast → N4 → difusión anisotrópica."""
    if img.GetPixelID() != sitk.sitkFloat32:
        img = sitk.Cast(img, sitk.sitkFloat32)
    corrected = bias_field_correction(img)
    return sitk.CurvatureAnisotropicDiffusion(corrected, timeStep=0.01875)


def extract_first_volume_if_4d(image):
    """Si la imagen es 4D (ej. DWI), extrae el primer volumen."""
    if image.GetDimension() == 4 and image.GetNumberOfComponentsPerPixel() == 1:
        size = list(image.GetSize())
        size[3] = 0
        return sitk.Extract(image, size, [0, 0, 0, 0])
    return image


def load_and_prepare_image(img_path):
    """Carga imagen, extrae volumen si 4D, y opcionalmente preprocesa."""
    img = sitk.ReadImage(img_path, sitk.sitkFloat32)
    img = extract_first_volume_if_4d(img)

    if ENABLE_PREPROCESSING:
        img = preprocess_image(img)
    return img


# ═══════════════════════════════════════════════════════════════════════════════
# WORKER INITIALIZER
# ═══════════════════════════════════════════════════════════════════════════════

def init_worker():
    """Inicializa los extractores de características en cada proceso worker."""
    global EXTRACTOR_T2, EXTRACTOR_ADC, EXTRACTOR_DWI
    EXTRACTOR_T2  = featureextractor.RadiomicsFeatureExtractor(PARAMS_T2)
    EXTRACTOR_ADC = featureextractor.RadiomicsFeatureExtractor(PARAMS_ADC)
    EXTRACTOR_DWI = featureextractor.RadiomicsFeatureExtractor(PARAMS_DWI)


# ═══════════════════════════════════════════════════════════════════════════════
# EXTRACCIÓN DE FEATURES
# ═══════════════════════════════════════════════════════════════════════════════

EXTRACTOR_MAP_KEYS = {"T2": "EXTRACTOR_T2", "ADC": "EXTRACTOR_ADC", "DWI": "EXTRACTOR_DWI"}


def extract_radiomic_features(extractor, img, mask, pid, sid, label):
    """Ejecuta la extracción radiómica y devuelve dict de features."""
    imageoperations.getMask = original_getMask
    feats = extractor.execute(img, mask)
    return {"patient_id": pid, "study_id": sid, "label": label, **feats}


def process_single_modality(args):
    """
    Procesa una sola modalidad. Diseñado para ejecutarse en ThreadPoolExecutor.
    Recibe tupla para compatibilidad con map/submit.
    """
    modality, img_path, gland_mask, pid, sid, label = args
    start = time.perf_counter()
    try:
        img = load_and_prepare_image(img_path)
        resampled_mask = resample_to_reference(gland_mask, img, is_mask=True)

        extractor = {"T2": EXTRACTOR_T2, "ADC": EXTRACTOR_ADC, "DWI": EXTRACTOR_DWI}[modality]
        feats = extract_radiomic_features(extractor, img, resampled_mask, pid, sid, label)
    except Exception as e:
        logger.error(f"Error en {modality} para paciente {pid}: {e}", exc_info=True)
        feats = None

    duration = time.perf_counter() - start
    return modality, feats, duration


def process_row(record):
    """
    Procesa las 3 modalidades de una fila del CSV.
    Usa ThreadPoolExecutor interno para paralelizar modalidades
    (SimpleITK libera el GIL en operaciones C++).
    """
    pid = record['patient_id']
    sid = record['study_id']
    lbl = record['case_csPCa']
    row_key = f"{pid}_{sid}"

    mask_path = os.path.join(PRE_PATH, record['whole_gland_path'])

    if not os.path.isfile(mask_path):
        logger.warning(f"Máscara no encontrada para {pid}: {mask_path}")
        return row_key, {}, {"T2": 0.0, "ADC": 0.0, "DWI": 0.0}

    gland_mask = sitk.ReadImage(mask_path)

    # Preparar tareas por modalidad
    modality_tasks = []
    for mod, path_key in [("T2", "t2w_path"), ("ADC", "adc_path"), ("DWI", "hbv_path")]:
        img_path = os.path.join(PRE_PATH, record[path_key])
        if not os.path.isfile(img_path):
            logger.warning(f"{mod} no encontrado para {pid}: {img_path}")
            continue
        modality_tasks.append((mod, img_path, gland_mask, pid, sid, lbl))

    results = {}
    times = {"T2": 0.0, "ADC": 0.0, "DWI": 0.0}

    if not modality_tasks:
        return row_key, results, times

    # Paralelizar modalidades con threads (SimpleITK libera GIL)
    n_threads = min(INNER_THREADS, len(modality_tasks))
    with ThreadPoolExecutor(max_workers=n_threads) as inner_pool:
        futures = [inner_pool.submit(process_single_modality, task) for task in modality_tasks]
        for fut in as_completed(futures):
            mod, feats, dt = fut.result()
            if feats:
                results[f"{mod.lower()}_gland"] = feats
            times[mod] = dt

    return row_key, results, times


# ═══════════════════════════════════════════════════════════════════════════════
# ESCRITURA BATCH
# ═══════════════════════════════════════════════════════════════════════════════

def flush_buffers(buffers, csv_paths):
    """Escribe los buffers acumulados a los CSVs correspondientes."""
    for mod, csv_path in csv_paths.items():
        if buffers[mod]:
            write_header = not os.path.exists(csv_path)
            pd.DataFrame(buffers[mod]).to_csv(
                csv_path, mode='a', header=write_header, index=False
            )
            buffers[mod] = []


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    logger.info("=" * 70)
    logger.info("Iniciando extracción radiómica optimizada")
    logger.info(f"  Preprocesamiento: {'ON' if ENABLE_PREPROCESSING else 'OFF'}")
    logger.info(f"  N4 shrink={N4_SHRINK_FACTOR}, iter={N4_MAX_ITERATIONS}")
    logger.info(f"  Batch size: {BATCH_SIZE}")
    logger.info("=" * 70)

    # Leer CSV de entrada
    df = pd.read_csv(INPUT_CSV, dtype=str)
    df['row_key'] = df['patient_id'] + '_' + df['study_id']

    # Cargar checkpoint
    processed = set()
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE) as f:
            processed = set(line.strip() for line in f if line.strip())

    to_do_df = df[~df['row_key'].isin(processed)]
    records = to_do_df.to_dict(orient='records')
    total_pending = len(records)

    logger.info(f"{total_pending} filas pendientes de {len(df)} totales "
                f"({len(processed)} ya procesadas)")

    if total_pending == 0:
        logger.info("No hay filas pendientes. Saliendo.")
        return

    # Determinar workers
    cpu_count = multiprocessing.cpu_count()
    if MAX_WORKERS is None:
        max_workers = max(1, min(cpu_count - 1, total_pending))
    else:
        max_workers = min(MAX_WORKERS, total_pending)
    logger.info(f"Workers: {max_workers} (CPUs disponibles: {cpu_count})")

    # Buffers para escritura batch
    buffers = {"T2": [], "ADC": [], "DWI": []}
    csv_paths = {"T2": T2_GLAND_CSV, "ADC": ADC_GLAND_CSV, "DWI": DWI_GLAND_CSV}

    # Acumuladores de tiempos
    totals = {"T2": 0.0, "ADC": 0.0, "DWI": 0.0}
    processed_count = 0
    error_count = 0
    global_start = time.perf_counter()

    with ProcessPoolExecutor(max_workers=max_workers, initializer=init_worker) as executor, \
         open(CHECKPOINT_FILE, 'a') as ckpt:

        futures = {executor.submit(process_row, rec): rec for rec in records}

        pbar = tqdm(as_completed(futures), total=len(futures), desc="Extrayendo features")
        for future in pbar:
            try:
                row_key, results, times = future.result()
            except Exception as e:
                rec = futures[future]
                logger.error(f"Error fatal procesando {rec.get('patient_id', '?')}: {e}",
                             exc_info=True)
                error_count += 1
                continue

            # Acumular features en buffers
            for mod in ("T2", "ADC", "DWI"):
                feat = results.get(f"{mod.lower()}_gland")
                if feat:
                    buffers[mod].append(feat)

            # Acumular tiempos
            for m in totals:
                totals[m] += times.get(m, 0.0)

            # Checkpoint
            ckpt.write(row_key + '\n')
            ckpt.flush()
            processed_count += 1

            # Flush batch si se alcanzó el tamaño
            total_buffered = sum(len(v) for v in buffers.values())
            if total_buffered >= BATCH_SIZE:
                flush_buffers(buffers, csv_paths)

            # Actualizar barra de progreso
            elapsed = time.perf_counter() - global_start
            rate = processed_count / elapsed if elapsed > 0 else 0
            pbar.set_postfix(rate=f"{rate:.1f} img/s", errors=error_count)

        # Flush final de lo que quede en buffers
        flush_buffers(buffers, csv_paths)

    # ── Reporte final ──────────────────────────────────────────────────────
    total_elapsed = time.perf_counter() - global_start
    logger.info("=" * 70)
    logger.info("REPORTE FINAL")
    logger.info(f"  Procesadas: {processed_count}/{total_pending}")
    logger.info(f"  Errores: {error_count}")
    logger.info(f"  Tiempo total: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")

    for mod, t in totals.items():
        avg = t / processed_count if processed_count > 0 else 0
        logger.info(f"  {mod}: {t:.1f}s total, {avg:.2f}s/imagen promedio")

    if processed_count > 0:
        logger.info(f"  Throughput: {processed_count/total_elapsed:.2f} filas/s")
    logger.info("=" * 70)
    logger.info("Proceso terminado.")


if __name__ == '__main__':
    main()
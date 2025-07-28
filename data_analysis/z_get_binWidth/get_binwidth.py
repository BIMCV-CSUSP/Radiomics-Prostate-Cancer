import os
import numpy as np
import pandas as pd
import SimpleITK as sitk
import concurrent.futures
from radiomics import featureextractor

# Path to your CSV file containing image paths
CSV_PATH = '/home/jaalzate/Radiomics-Prostate-Cancer/artifacts/bimcv_data/BIMCV_with_predictions_filtered_volume.csv'

# Configuración de extractores por modalidad	
extractors = {
    'image_t2': featureextractor.RadiomicsFeatureExtractor(
        '/home/jaalzate/Radiomics-Prostate-Cancer/data_analysis/z_get_binWidth/get_binWidth_Params/Params_T2w.yaml'
    ),
    'image_dwi': featureextractor.RadiomicsFeatureExtractor(
        '/home/jaalzate/Radiomics-Prostate-Cancer/data_analysis/z_get_binWidth/get_binWidth_Params/Params_DWI.yaml'
    ),
    'image_adc': featureextractor.RadiomicsFeatureExtractor(
        '/home/jaalzate/Radiomics-Prostate-Cancer/data_analysis/z_get_binWidth/get_binWidth_Params/Params_ADC.yaml'
    )
}

# Sólo calcular firstorder:Range
for modality, extractor in extractors.items():
    extractor.disableAllFeatures()
    extractor.enableFeaturesByName(firstorder=['Range'])

def process_file(image_path, modality, extractor):
    """
    Procesa una imagen y devuelve:
      - modality
      - firstorder:Range
      - fd_binWidth (Freedman–Diaconis) para ese ROI
    """
    try:
        image = sitk.ReadImage(image_path)

        # Si es 4D scalar, extraigo sólo la primera volumen
        if image.GetDimension() == 4 and image.GetNumberOfComponentsPerPixel() == 1:
            size = list(image.GetSize()); size[3] = 0
            image = sitk.Extract(image, size, (0,0,0,0))

        # Array + máscara
        img_arr = sitk.GetArrayFromImage(image)
        mask_arr = np.ones_like(img_arr, dtype=np.uint8)
        mask_arr.flat[0] = 0

        # Vector vs scalar mask
        if image.GetNumberOfComponentsPerPixel() > 1:
            mask = sitk.GetImageFromArray(mask_arr, isVector=True)
        else:
            mask = sitk.GetImageFromArray(mask_arr)
        mask.CopyInformation(image)

        # Extraer Range
        result = extractor.execute(image, mask)
        rng = result.get('original_firstorder_Range', None)

        # Freedman–Diaconis bin width sobre los intensidades reales del ROI
        vox = img_arr[mask_arr == 1].astype(float)
        if vox.size > 0:
            q75, q25 = np.percentile(vox, [75, 25])
            iqr = q75 - q25
            n = vox.size
            fd_bw = 2 * iqr / (n ** (1/3))
        else:
            fd_bw = None

        return modality, rng, fd_bw

    except Exception as e:
        print(f"Error procesando {modality} en {image_path}: {e}")
        return modality, None, None

# Leer CSV y montar tareas
df = pd.read_csv(CSV_PATH)
tasks = []
for mod, ext in extractors.items():
    if mod not in df.columns:
        raise ValueError(f"La columna '{mod}' no existe en el CSV")
    for p in df[mod].dropna():
        tasks.append((p, mod, ext))

# Diccionarios para Range y FD-binWidth
modality_ranges = {m: [] for m in extractors}
modality_fd      = {m: [] for m in extractors}

# Procesamiento concurrente
with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as exe:
    futures = [exe.submit(process_file, p, m, e) for p,m,e in tasks]
    for f in concurrent.futures.as_completed(futures):
        m, rng, fd = f.result()
        modality_ranges[m].append(rng)
        modality_fd[m].append(fd)

# Mostrar estadísticas y sugerencias
for mod in extractors:
    print(f"\n=== Modalidad: {mod.upper()} ===")
    ranges = np.array([r for r in modality_ranges[mod] if r is not None])
    fdbws  = np.array([b for b in modality_fd[mod]      if b is not None])

    if ranges.size:
        print("Firstorder:Range:")
        print(f"  Media:   {ranges.mean():.2f}")
        print(f"  Mínimo:  {ranges.min():.2f}")
        print(f"  Máximo:  {ranges.max():.2f}")

        # Sugerir binWidth fijo según número de bins objetivo
        print("\nSugerencia de binWidth fija (fixed bin width):")
        for tb in [16, 32, 64, 128]:
            bw = ranges.mean() / tb
            print(f"  → {tb} bins ⇒ binWidth ≃ {bw:.2f}")

    else:
        print("No se extrajo ningún Range.")

    if fdbws.size:
        print(f"\nFreedman–Diaconis (FD) binWidth media: {fdbws.mean():.2f}")
        print("  (Regla FD: binWidth = 2·IQR / n^(1/3), donde IQR y n son del ROI)")
    else:
        print("No se pudo calcular FD binWidth para ningún ROI.")
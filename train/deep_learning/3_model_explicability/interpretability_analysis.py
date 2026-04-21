from __future__ import annotations

import argparse
import glob
import importlib
import json
import os
import random
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
COMMON_DIR = PROJECT_ROOT / "train" / "common"
if str(COMMON_DIR) not in sys.path:
    sys.path.append(str(COMMON_DIR))

import matplotlib.pyplot as plt
import monai
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from runtime_utils import load_splits, resolve_project_path
from z_data_loader_for_explicability_roi import MyDataLoader

plt.style.use("dark_background")
plt.rcParams["figure.figsize"] = (12, 8)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Interpretability analysis for deep learning models.")
    
    parser.add_argument("--model-type", type=str, default="config1", help="Model configuration to analyse.")
    
    parser.add_argument(
        "--preselected-indices",
        type=str,
        default=None,
        help="Comma-separated sample indices to analyse directly (for example: 5,23,42).",
    )
    
    parser.add_argument("--criteria", type=str, default="correct_class1",
                        choices=["correct_class0", "correct_class1", "incorrect", "high_confidence", "any"],
                        help="Rule used to select validation samples for analysis.")
    
    parser.add_argument("--max-samples", type=int, default=3, help="Maximum number of samples to analyse.")
    
    parser.add_argument("--skip-gradcam", action="store_true", help="Skip Grad-CAM generation.")
    
    parser.add_argument("--skip-occlusion", action="store_true", help="Skip occlusion sensitivity maps.")
    
    parser.add_argument("--skip-aggregated", action="store_true", help="Skip aggregated interpretability maps.")
    
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=100,
        help="Maximum attempts when searching for random candidate samples.",
    )
    
    parser.add_argument(
        "--split",
        type=int,
        default=None,
        help="Validation split to analyse. If omitted, the split with the best validation AUC is used.",
    )
    
    parser.add_argument(
        "--project-root",
        type=str,
        default=str(PROJECT_ROOT),
        help="Project root directory.",
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/deep_learning/interpretability",
        help="Directory where interpretability artefacts will be saved.",
    )
    
    args = parser.parse_args()

    if args.preselected_indices is not None:
        args.preselected_indices = [int(x) for x in args.preselected_indices.split(",") if x.strip()]
    else:
        args.preselected_indices = None

    return args


def load_model_and_test_data(model_dir, csv_path, split_to_use=None, project_root=None):
    """
    Load a trained model and retrieve the validation subset for a specific persisted split.
    
    Args:
        model_dir: Directorio donde se encuentra el modelo
        csv_path: Ruta al CSV de datos
        split_to_use: Validation split index to analyse. If None, the best validation split is used.
    
    Returns:
        model: Modelo cargado
        validation_dataloader: DataLoader with the persisted validation subset
    """
    config_path = resolve_project_path(project_root, "train/deep_learning/1_modeling/config.json")
    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)
    
    # Obtener configuración para este modelo específico
    model_type = os.path.basename(model_dir)
    if model_type not in config:
        raise ValueError(f"Model type '{model_type}' was not found in config.json.")
    
    model_config = config[model_type]

    model_class_path = model_config["model"]
    module_path, class_name = model_class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    model_class = getattr(module, class_name)
    
    # Crear instancia del modelo
    model = model_class(**model_config["model_args"])
    
    if split_to_use is None:
        model_path = os.path.join(model_dir, "best_overall_model.pth")
        base_dir = os.path.dirname(os.path.dirname(model_dir))
        results_dir = os.path.join(base_dir, "results", os.path.basename(model_dir))
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
        print(f"Using validation split {split_to_use} (best validation AUC: {best_auc:.4f})")
    else:
        model_path = os.path.join(model_dir, f"best_model_split_{split_to_use}.pth")
    
    model.load_state_dict(torch.load(model_path, map_location=torch.device("cuda" if torch.cuda.is_available() else "cpu")))
    model.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval()

    data_loader = MyDataLoader(
        csv_path=csv_path,
        input_shape=(128, 128, 32),  
        config={"batch_size": 1, "num_workers": 4},
    )

    all_data = data_loader.get_all_data()

    split_file = resolve_project_path(
        project_root,
        os.path.join(
            "artifacts",
            "deep_learning",
            "gland",
            "splits",
            f"{Path(csv_path).stem}_gland_5fold_seed42.json",
        ),
    )
    split_bundle = load_splits(split_file)
    validation_indices = split_bundle["splits"][split_to_use - 1]["validation_indices"]
    validation_subset = [all_data[i] for i in validation_indices]

    test_dataset = monai.data.Dataset(
        data=validation_subset,
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

def seleccionar_indices_muestras(
    dataloader,
    model,
    model_results_dir,
    csv_path,
    criteria,
    preselected_indices = None,   
    max_samples = 3,
    max_attempts = 100,
    verbose = True,
):
    """
    Devuelve una lista de dict:
        { idx, true_class, dir }

    Para cada muestra aceptada se crea:
        <dir>/metadata.txt   con:
            Índice …
            Forma …
            Nombre archivo …
            Clase real …
            Predicción …
            Probabilidades …
    """
    os.makedirs(model_results_dir, exist_ok=True)
    selected_info: list[dict] = []
    sample_counter = 0
    already_selected: set[int] = set()

    # helper -------------------------------------------------------------
    def _save_metadata(idx, img, filename, true_c, pred_c, probs, out_dir):
        names = ("no_csPCa", "csPCa")
        with open(os.path.join(out_dir, "metadata.txt"), "w") as f:
            f.write(f"Índice: {idx}\n")
            f.write(f"Forma: {tuple(img.shape)}\n")
            f.write(f"Nombre archivo: {filename}\n")
            f.write(f"Clase real: {true_c} ({names[true_c]})\n")
            f.write(f"Predicción: {pred_c} ({names[pred_c]})\n")
            f.write(
                f"Probabilidades: [no_csPCa: {probs[0]:.4f}, "
                f"csPCa: {probs[1]:.4f}]\n"
            )

    device = next(model.parameters()).device

    # =====================================================================
    # (0) ÍNDICES PRESELECCIONADOS
    # =====================================================================
    if preselected_indices:
        if verbose:
            print(f"[Selección] índices dados: {preselected_indices}")

        for idx in preselected_indices[:max_samples]:
            try:
                sample = dataloader.dataset[idx]
                img    = sample["image"].to(device)
                label  = sample["label"]
                true_c = int(label.argmax())

                with torch.no_grad():
                    probs = torch.softmax(model(img.unsqueeze(0)), dim=1)[0].cpu()
                pred_c = int(probs.argmax())

                sample_counter += 1
                already_selected.add(idx)

                dir_name = f"manual_sample{sample_counter}_idx{idx}_class{true_c}"
                dir_path = os.path.join(model_results_dir, dir_name)
                os.makedirs(dir_path, exist_ok=True)

                # nombre de archivo o genérico
                try:
                    filename = os.path.basename(img.meta["filename_or_obj"])
                except Exception:
                    filename = f"sample_{idx}"

                _save_metadata(idx, img, filename, true_c, pred_c, probs, dir_path)

                selected_info.append({"idx": idx, "true_class": true_c, "dir": dir_path})

                if verbose:
                    print(f"✓ ({sample_counter}/{max_samples}) idx={idx} class={true_c}")

            except Exception as e:
                if verbose:
                    print(f"⚠ No se pudo procesar idx={idx}: {e}")

    if sample_counter >= max_samples:
        return selected_info

    # =====================================================================
    # (1) CRITERIO «correct_class1» con ISUP
    # =====================================================================
    if criteria == "correct_class1" and csv_path:
        if verbose:
            print("[Selección] csPCa correctos por ISUP (desc)")

        try:
            isup_data = pd.read_csv(csv_path)
            if verbose:
                print(f"✓ CSV '{csv_path}' cargado ({len(isup_data)} filas)")
        except Exception as e:
            print(f"⚠ No se pudo leer el CSV: {e}")
            isup_data = pd.DataFrame()

        for isup_target in [5, 4, 3, 2, 1]:
            if sample_counter >= max_samples:
                break
            if verbose:
                print(f"\nBuscando ISUP = {isup_target} …")

            indices = list(range(len(dataloader.dataset)))
            random.shuffle(indices)
            for idx in indices:
                if sample_counter >= max_samples:
                    break
                if idx in already_selected:
                    continue
                try:
                    sample = dataloader.dataset[idx]
                    img    = sample["image"].to(device)
                    label  = sample["label"]
                    true_c = int(label.argmax())
                    if true_c != 1:
                        continue

                    try:
                        filename = os.path.basename(img.meta["filename_or_obj"])
                    except Exception:
                        filename = f"sample_{idx}.nii.gz"
                    parts = filename.split("_")
                    patient_id = int(parts[0])
                    study_id   = int(parts[1].split(".")[0])

                    row = isup_data[
                        (isup_data.patient_id == patient_id) &
                        (isup_data.study_id   == study_id)
                    ]
                    if row.empty or int(row.iloc[0]["case_ISUP"]) != isup_target:
                        continue

                    with torch.no_grad():
                        probs = torch.softmax(model(img.unsqueeze(0)), dim=1)[0].cpu()
                    pred_c = int(probs.argmax())
                    if pred_c != 1:
                        continue

                    sample_counter += 1
                    already_selected.add(idx)

                    dir_name = f"{criteria}_sample{sample_counter}_idx{idx}_class{true_c}"
                    dir_path = os.path.join(model_results_dir, dir_name)
                    os.makedirs(dir_path, exist_ok=True)

                    _save_metadata(idx, img, filename, true_c, pred_c, probs, dir_path)

                    selected_info.append({"idx": idx, "true_class": true_c, "dir": dir_path})

                    if verbose:
                        print(f"✓ ({sample_counter}/{max_samples}) idx={idx} "
                              f"ISUP={isup_target} conf={probs[1]:.4f}")

                except Exception:
                    continue

        if sample_counter >= max_samples:
            return selected_info

    # =====================================================================
    # (2) MODOS ALEATORIOS
    # =====================================================================
    if verbose:
        print(f"[Selección] modo aleatorio · criterio = '{criteria}'")

    tried = 0
    while sample_counter < max_samples and tried < max_attempts:
        idx = random.randrange(len(dataloader.dataset))
        if idx in already_selected:
            continue
        tried += 1

        sample = dataloader.dataset[idx]
        img    = sample["image"].to(device)
        label  = sample["label"]
        true_c = int(label.argmax())

        with torch.no_grad():
            probs = torch.softmax(model(img.unsqueeze(0)), dim=1)[0].cpu()
        pred_c = int(probs.argmax())
        conf   = float(probs[pred_c])

        if criteria == "correct_class0" and (pred_c != 0 or true_c != 0):
            continue
        if criteria == "incorrect" and pred_c == true_c:
            continue
        if criteria == "high_confidence" and conf <= 0.9:
            continue

        sample_counter += 1
        already_selected.add(idx)

        dir_name = f"{criteria}_sample{sample_counter}_idx{idx}_class{true_c}"
        dir_path = os.path.join(model_results_dir, dir_name)
        os.makedirs(dir_path, exist_ok=True)

        try:
            filename = os.path.basename(img.meta["filename_or_obj"])
        except Exception:
            filename = f"sample_{idx}"

        _save_metadata(idx, img, filename, true_c, pred_c, probs, dir_path)

        selected_info.append({"idx": idx, "true_class": true_c, "dir": dir_path})

        if verbose:
            print(f"✓ ({sample_counter}/{max_samples}) idx={idx} "
                  f"real={true_c} pred={pred_c} conf={conf:.4f}")

    if sample_counter < max_samples and verbose:
        print(f"⚠ Sólo se seleccionaron {sample_counter}/{max_samples} muestras.")

    return selected_info

def get_bounding_box(mask: torch.Tensor, margin: int = 0):
    """
    Devuelve (slice_z, slice_y, slice_x) que engloban los voxels > 0 de `mask`,
    extendidas `margin` voxels por cada lado.

    mask  : (C, Z, Y, X)  o  (Z, Y, X)  |   se asume que ya está en el dispositivo correcto
    margin: voxels extra de cada lado (int ≥ 0)
    """
    # Quitar canal si viene como (1, Z, Y, X)
    mask_ = mask[0] if (mask.ndim == 4 and mask.shape[0] == 1) else mask

    nz = torch.nonzero(mask_, as_tuple=False)

    if nz.numel() == 0:                      # máscara vacía → caja = volumen completo
        z0 = y0 = x0 = 0
        z1, y1, x1 = mask_.shape
    else:
        min_idx = nz.min(0).values - margin
        max_idx = nz.max(0).values + 1 + margin

        # tensor con las dimensiones, PERO en el mismo dispositivo
        dims = torch.tensor(mask_.shape, device=mask_.device)

        min_idx = min_idx.clamp(min=0)
        max_idx = max_idx.clamp_max(dims)

        z0, y0, x0 = min_idx.tolist()
        z1, y1, x1 = max_idx.tolist()

    return slice(z0, z1), slice(y0, y1), slice(x0, x1)

def guardar_imagenes_full_y_roi(
    dataloader,
    selected_info,
    margin: int = 20
):
    """
    Para cada elemento de selected_info guarda:
      <dir>/images/roi/T2w.png  … ADC.png  … DWI.png
      <dir>/images/original/T2w.png  … ADC.png  … DWI.png
    usando un crop basado en la bounding box de la glándula, más un margen.
    """
    for info in selected_info:
        idx      = info["idx"]
        dir_base = info["dir"]

        try:
            sample    = dataloader.dataset[idx]
            img_roi   = sample['image'].to(device)        # ROI (próstata)
            img_full  = sample['image_full'].to(device)   # Imagen completa
            mask      = sample['mask'].to(device)         # Máscara ya redimensionada

            # Asegurar canal: (1, Z, Y, X)
            if mask.ndim == 3:
                mask = mask.unsqueeze(0)

            bbox_slices = get_bounding_box(mask, margin=margin)

            # Crop ROI, full y máscara según bbox
            img_roi_crop  = img_roi[:, bbox_slices[0], bbox_slices[1], bbox_slices[2]]
            img_full_crop = img_full[:, bbox_slices[0], bbox_slices[1], bbox_slices[2]]

            # Directorios de salida
            dir_roi  = os.path.join(dir_base, "original_images", "gland")
            dir_full = os.path.join(dir_base, "original_images", "full")
            os.makedirs(dir_roi,  exist_ok=True)
            os.makedirs(dir_full, exist_ok=True)

            # --- 1. GUARDAR ROI ---
            for ch, name in enumerate(["T2w", "ADC", "DWI"]):
                plt.figure()
                monai.visualize.matshow3d(
                    volume=img_roi_crop.cpu()[ch:ch+1],
                    frame_dim=-1, channel_dim=0, every_n=2,
                    margin=6, show=False, cmap='gray'
                )
                plt.savefig(os.path.join(dir_roi, f"{name}.png"), dpi=500)
                plt.close()

            # --- 2. GUARDAR ORIGINAL ---
            for ch, name in enumerate(["T2w", "ADC", "DWI"]):
                plt.figure()
                monai.visualize.matshow3d(
                    volume=img_full_crop.cpu()[ch:ch+1],
                    frame_dim=-1, channel_dim=0, every_n=2,
                    margin=6, show=False, cmap='gray'
                )
                plt.savefig(os.path.join(dir_full, f"{name}.png"), dpi=500)
                plt.close()

        except Exception as e:
            print(f"Error al procesar índice {idx}: {e}")
            import traceback; traceback.print_exc()

def generar_gradcam_gbp(
    model,
    dataloader,
    selected_info,
    margin = 20,
    target_layer = "features.norm5",
    fc_layer = "class_layers.out",
):
    """
    Genera y guarda mapas Grad-CAM, Guided Backprop y su combinación
    (CAM × GBP) + mezcla con la imagen ADC, recortados a la ROI de la próstata
    con un margen adicional.

    ▸ Para cada elemento de `selected_info` (dict con "idx" y "dir") crea:
          <dir>/images/gradcam/gbp.png
          <dir>/images/gradcam/cam.png
          <dir>/images/gradcam/result.png
          <dir>/images/gradcam/blended.png
    ▸ Requiere que exista la función `get_bounding_box(mask, margin)`.
    """
    # inicializadores MONAI
    cam = monai.visualize.class_activation_maps.CAM(
        nn_module=model, target_layers=target_layer, fc_layers=fc_layer
    )
    gbp = monai.visualize.gradient_based.GuidedBackpropSmoothGrad(model, n_samples=50)

    for info in selected_info:
        idx      = info["idx"]
        dir_base = info["dir"]

        try:
            # ----------------------- datos y predicción -----------------------
            sample   = dataloader.dataset[idx]
            img      = sample["image"].to(device)          # (C, Z, Y, X) – ROI original
            mask     = sample["mask"].to(device)           # (1, Z, Y, X) o (Z, Y, X)
            label    = sample["label"].to(device)

            pred = torch.nn.functional.softmax(model(img.unsqueeze(0)), dim=1)
            pred_label  = pred.argmax().item()
            true_label  = label.argmax().item() if label.numel() > 1 else int(label)

            if pred_label != true_label:
                continue                                   # omitimos los fallos

            # ------------------------- Grad-CAM & GBP -------------------------
            cam_result = cam(x=img.unsqueeze(0)).squeeze(0).cpu()        # (C, Z, Y, X)
            gbp_result = gbp(x=img.unsqueeze(0)).squeeze(0).cpu()

            # normalización sencilla [0–255] y combinación
            gbp_result = 255 * (gbp_result - gbp_result.min()) / (gbp_result.max() - gbp_result.min())
            cam_result = 255 * (cam_result - cam_result.min()) / (cam_result.max() - cam_result.min())
            result     = (cam_result * gbp_result)
            result     = (result - result.min()) / (result.max() - result.min())

            # mezcla (ADC + result)
            blended = monai.visualize.utils.blend_images(
                255 * img.cpu()[1:2], result[1:2], alpha=0.2, transparent_background=True
            )

            # -------------------- bounding-box de la próstata -----------------
            if mask.ndim == 3:
                mask = mask.unsqueeze(0)
            bbox_slices = get_bounding_box(mask, margin=margin)

            # recortes a la ROI
            cam_crop     = cam_result[:,  bbox_slices[0], bbox_slices[1], bbox_slices[2]]
            gbp_crop     = gbp_result[:,  bbox_slices[0], bbox_slices[1], bbox_slices[2]]
            result_crop  = result[:,      bbox_slices[0], bbox_slices[1], bbox_slices[2]]
            blended_crop = blended[:,     bbox_slices[0], bbox_slices[1], bbox_slices[2]]

            # -------------------------- guardado ------------------------------
            gradcam_dir = os.path.join(dir_base, "GradCAM")
            os.makedirs(gradcam_dir, exist_ok=True)

            # GBP (ADC)
            plt.figure()
            monai.visualize.matshow3d(
                volume=gbp_crop[1:2], frame_dim=-1, channel_dim=0,
                every_n=2, margin=6, show=False
            )
            plt.savefig(os.path.join(gradcam_dir, "gbp.png"), dpi=500)
            plt.close()

            # CAM
            plt.figure()
            monai.visualize.matshow3d(
                volume=cam_crop, frame_dim=-1, channel_dim=0,
                every_n=2, margin=6, show=False
            )
            plt.savefig(os.path.join(gradcam_dir, "cam.png"), dpi=500)
            plt.close()

            # RESULT = CAM × GBP
            plt.figure()
            monai.visualize.matshow3d(
                volume=result_crop, frame_dim=-1, channel_dim=0,
                every_n=2, margin=6, fill_value=255, show=False, cmap="gray"
            )
            plt.savefig(os.path.join(gradcam_dir, "result.png"), dpi=500)
            plt.close()

            # BLENDED (ADC + mapa fusionado)
            plt.figure()
            monai.visualize.matshow3d(
                volume=blended_crop, frame_dim=-1, channel_dim=0,
                every_n=2, margin=6, show=False
            )
            plt.savefig(os.path.join(gradcam_dir, "blended.png"), dpi=500)
            plt.close()

        except Exception as e:
            print(f"[GradCAM] Error en índice {idx}: {e}")
            import traceback; traceback.print_exc()

def generar_aggregated_maps(
    dataloader,
    selected_info,
    maps,
    margin: int = 20,          
    threshold: float = 0.8,
    channel_names=("T2w", "ADC", "DWI"),
):
    """
    Para cada caso genera dos carpetas:

        <dir>/AggregatedMaps/gland/
            T2w_map_<clase>.png   (ROI)
            ADC_map_<clase>.png
            DWI_map_<clase>.png

        <dir>/AggregatedMaps/full/
            T2w_map_<clase>.png   (imagen completa + overlay)
            ADC_map_<clase>.png
            DWI_map_<clase>.png

    – El overlay es el mapa agregado de la **clase real**,
      umbralizado (> threshold) y normalizado 0-255.
    – El blend se hace canal-a-canal con `monai.visualize.utils.blend_images`.
    """

    # Normalizar una vez los mapas globales
    maps_norm = {k: (v - v.min()) / (v.max() - v.min()) for k, v in maps.items()}

    for info in selected_info:
        idx, dir_base = info["idx"], info["dir"]

        try:
            # ──────────── cargar muestra ────────────
            sample     = dataloader.dataset[idx]
            img_roi    = sample["image"].to(device)         # (C, Z, Y, X)
            img_full   = sample["image_full"].to(device)    # (C, Z, Y, X)
            mask       = sample["mask"].to(device)          # (1, Z, Y, X) o (Z, Y, X)
            label      = sample["label"].to(device)

            true_class = label.argmax().item() if label.numel() > 1 else int(label)
            class_key  = "no_csPCa" if true_class == 0 else "csPCa"
            if class_key not in maps_norm:
                print(f"[AggMaps] mapa '{class_key}' no encontrado (idx {idx})")
                continue

            map_global = maps_norm[class_key]               # (C, Z, Y, X)

            # ──────────── bounding-box ROI ────────────
            if mask.ndim == 3:
                mask = mask.unsqueeze(0)
            z_s, y_s, x_s = get_bounding_box(mask, margin=margin)

            img_crop = img_roi.cpu()[:, z_s, y_s, x_s]      # ROI (C, Z, Y, X)
            map_crop = map_global[:, z_s, y_s, x_s]

            # umbralizar
            map_thr = torch.where(map_crop > threshold, map_crop, torch.zeros_like(map_crop))

            # ──────────── carpetas de salida ────────────
            dir_gland = os.path.join(dir_base, "AggregatedMaps", "gland")
            dir_full  = os.path.join(dir_base, "AggregatedMaps", "full")
            os.makedirs(dir_gland, exist_ok=True)
            os.makedirs(dir_full,  exist_ok=True)

            # ──────────── helper para guardar ────────────
            def _save(vol, path, cmap=None):
                plt.figure()
                monai.visualize.matshow3d(
                    volume=vol, frame_dim=-1, channel_dim=0,
                    every_n=2, margin=6, show=False,
                    cmap=cmap
                )
                plt.savefig(path, dpi=500)
                plt.close()

            # ──────────── guardar ROI (gland) ────────────
            for c_idx, c_name in enumerate(channel_names):
                blended_roi = monai.visualize.utils.blend_images(
                    255 * img_crop[c_idx:c_idx+1],
                    255 * map_thr[c_idx:c_idx+1],
                    alpha=0.3, transparent_background=True
                )
                _save(blended_roi,
                      os.path.join(dir_gland, f"{c_name}_map_{class_key}.png"))

            # ──────────── preparar mapas FULL ────────────
            adc_full = 255 * img_full.cpu()                 # base completa (C, Z, Y, X)

            # tensor vacío para cada canal y pegamos el overlay en la ROI
            overlay_full = torch.zeros_like(adc_full)       #  ←  antes era torch.zeros_like(map_thr)
            overlay_full[:, z_s, y_s, x_s] = 255 * map_thr  # coloca la ROI en su sitio

            # ──────────── guardar FULL ────────────
            for c_idx, c_name in enumerate(channel_names):
                blended_full = monai.visualize.utils.blend_images(
                    adc_full[c_idx:c_idx+1],
                    overlay_full[c_idx:c_idx+1],
                    alpha=0.3, transparent_background=True
                )
                _save(blended_full,
                      os.path.join(dir_full, f"{c_name}_map_{class_key}.png"))

        except Exception as e:
            print(f"[AggMaps] Error en índice {idx}: {e}")
            import traceback; traceback.print_exc()

def generar_occlusion_maps(
    dataloader,
    selected_info,
    sensitivity_maps_dir,               # carpeta donde vive cada *.pth
    margin: int = 20,
    threshold: float = 0.8,
    channel_names=("T2w", "ADC", "DWI"),
):
    """
    Para cada caso ->  dos carpetas y 3 PNG en cada una:

        <dir>/OcclusionSensitivity/gland/
            T2w_occlusion.png   ADC_occlusion.png   DWI_occlusion.png   (ROI)

        <dir>/OcclusionSensitivity/full/
            T2w_occlusion.png   ADC_occlusion.png   DWI_occlusion.png   (imagen completa)

    El mapa de oclusión se normaliza, se umbraliza (> threshold) y se mezcla
    con el canal correspondiente de la imagen (alpha = 0.2).
    """

    # ---------- helper para guardar con matshow3d ----------
    def _save(vol, path):
        plt.figure()
        monai.visualize.matshow3d(
            volume=vol, frame_dim=-1, channel_dim=0,
            every_n=2, margin=6, show=False
        )
        plt.savefig(path, dpi=500)
        plt.close()

    for info in selected_info:
        idx, dir_base = info["idx"], info["dir"]

        try:
            # ---------------- datos de la muestra ----------------
            sample   = dataloader.dataset[idx]
            img_roi  = sample["image"].to(device)        # (C, Z, Y, X)
            img_full = sample["image_full"].to(device)   # (C, Z, Y, X)
            mask     = sample["mask"].to(device)
            label    = sample["label"].to(device)

            true_class = label.argmax().item() if label.numel() > 1 else int(label)
            class_str  = "no_csPCa" if true_class == 0 else "csPCa"

            # nombre de archivo original que se usó al crear los mapas
            filename = os.path.basename(img_roi.meta["filename_or_obj"])
            map_path = os.path.join(sensitivity_maps_dir, f"class{true_class}_{filename}")
            if not os.path.exists(map_path):
                print(f"[OccMaps] Mapa no encontrado → {map_path}")
                continue

            heatmap = torch.load(map_path).cpu()                        # (1, Z, Y, X) o (Z, Y, X)
            heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())

            # --------------- ROI bbox (+ margen) ------------------
            if mask.ndim == 3:
                mask = mask.unsqueeze(0)
            z_s, y_s, x_s = get_bounding_box(mask, margin=margin)

            img_crop = img_roi.cpu()[:, z_s, y_s, x_s]                  # (C, Z, Y, X)
            hm_crop  = heatmap[:, z_s, y_s, x_s] if heatmap.ndim == 4 else heatmap[z_s, y_s, x_s]

            # umbral
            hm_thr   = torch.where(hm_crop > threshold, hm_crop, torch.zeros_like(hm_crop))

            # --------------- carpetas de salida -------------------
            dir_gland = os.path.join(dir_base, "OcclusionSensitivity", "gland")
            dir_full  = os.path.join(dir_base, "OcclusionSensitivity", "full")
            os.makedirs(dir_gland, exist_ok=True)
            os.makedirs(dir_full,  exist_ok=True)

            # --------------- guardar ROI --------------------------
            for c_idx, c_name in enumerate(channel_names):
                blended_roi = monai.visualize.utils.blend_images(
                    255 * img_crop[c_idx:c_idx+1],
                    255 * hm_thr if heatmap.ndim == 3 else 255 * hm_thr,  # hm_thr ya (Z,Y,X)
                    alpha=0.2, transparent_background=True
                )
                _save(blended_roi, os.path.join(dir_gland, f"{c_name}_occlusion.png"))

            # --------------- preparar FULL ------------------------
            adc_full     = 255 * img_full.cpu()               # (C, Z, Y, X)
            overlay_full = torch.zeros_like(adc_full)         # (C, Z, Y, X)
            
            # quitar eje-extra si existe → hm_src queda (Zc, Yc, Xc)
            hm_src = hm_thr.squeeze(0) if hm_thr.ndim == 4 else hm_thr
            
            # colocar ROI en su sitio; broadcast al primer eje (C)
            overlay_full[:, z_s, y_s, x_s] = 255 * hm_src    
            
            # ---------- guardar FULL ----------
            for c_idx, c_name in enumerate(channel_names):
                blended_full = monai.visualize.utils.blend_images(
                    adc_full[c_idx:c_idx+1],
                    overlay_full[c_idx:c_idx+1],
                    alpha=0.2,
                    transparent_background=True
                )
                _save(blended_full, os.path.join(dir_full, f"{c_name}_occlusion.png"))
                
            print(f"✓ Occlusion maps guardados en {os.path.join(dir_base,'OcclusionSensitivity')}")

        except Exception as e:
            print(f"[OccMaps] Error en índice {idx}: {e}")
            import traceback; traceback.print_exc()

def ejecutar_todos_los_analisis(
    model,
    dataloader,
    selected_info,
    maps=None,                     # para aggregated
    sensitivity_maps_dir=None,     # para oclusión
    margin=20,
    threshold=0.8,
    channel_names=("T2w", "ADC", "DWI"),
    skip_gradcam=False,
    skip_occlusion=False,
    skip_aggregated=False,
):
    # 1. GradCAM + GBP
    if not skip_gradcam:
        print("\n--- Generando GradCAM y GBP ---")
        generar_gradcam_gbp(
            model=model,
            dataloader=dataloader,
            selected_info=selected_info,
            margin=margin
        )
    else:
        print("Saltando GradCAM y GBP...")

    # 2. Occlusion Sensitivity (mapas individuales y agregados)
    if not skip_occlusion:
        print("\n--- Generando mapas de oclusión ---")
        assert sensitivity_maps_dir is not None, "Se requiere sensitivity_maps_dir para oclusión"
        generar_occlusion_maps(
            dataloader=dataloader,
            selected_info=selected_info,
            sensitivity_maps_dir=sensitivity_maps_dir,
            margin=margin,
            threshold=threshold,
            channel_names=channel_names
        )
    else:
        print("Saltando mapas de oclusión...")

    # 3. Aggregated maps
    if not skip_aggregated:
        print("\n--- Generando mapas agregados ---")
        assert maps is not None, "Se requiere el objeto 'maps' para mapas agregados"
        generar_aggregated_maps(
            dataloader=dataloader,
            selected_info=selected_info,
            maps=maps,
            margin=margin,
            threshold=threshold,
            channel_names=channel_names
        )
    else:
        print("Saltando mapas agregados...")

    print("\n✓ Análisis completo para todas las muestras seleccionadas.")


if __name__ == "__main__":
    args = parse_arguments()

    # 1. Resolve the project-specific paths.
    model_base_dir = os.path.join(args.project_root, "artifacts/deep_learning/gland/models/", args.model_type)
    csv_path = os.path.join(args.project_root, "artifacts", "data.csv")
    model_results_dir = os.path.join(args.output_dir, args.model_type)
    occlusion_dir = os.path.join(model_results_dir, "OcclusionSensitivity")
    maps_dir = os.path.join(model_results_dir, "OcclusionSensitivity", "individual_maps")
    sensitivity_maps_dir = maps_dir

    # 2. Load the model and the persisted validation DataLoader.
    model, test_dataloader, split_to_use = load_model_and_test_data(
        model_base_dir,
        csv_path,
        split_to_use=args.split,
        project_root=args.project_root
    )

    # 3. Compute occlusion maps if needed and load aggregated maps.
    calculate_occlusion_sensitivity(model, test_dataloader, maps_dir, occlusion_dir)
    agg_maps_path = os.path.join(occlusion_dir, "aggregated_heatmaps.pth")
    maps = torch.load(agg_maps_path) if os.path.exists(agg_maps_path) else None

    # 4. Select validation samples for the analysis.
    selected_info = seleccionar_indices_muestras(
        dataloader=test_dataloader,
        model=model,
        model_results_dir=model_results_dir,
        csv_path=csv_path,
        criteria=args.criteria,
        preselected_indices=args.preselected_indices,
        max_samples=args.max_samples,
        max_attempts=args.max_attempts,
        verbose=True
    )
    if not selected_info:
        print("No validation samples were selected for analysis.")
        sys.exit(1)

    # 5. Save the baseline full and ROI images.
    guardar_imagenes_full_y_roi(
        dataloader=test_dataloader,
        selected_info=selected_info,
        margin=20
    )

    # 6. Run the main interpretability analyses.
    ejecutar_todos_los_analisis(
        model=model,
        dataloader=test_dataloader,
        selected_info=selected_info,
        maps=maps,
        sensitivity_maps_dir=sensitivity_maps_dir,
        margin=20,
        threshold=0.8,
        channel_names=("T2w", "ADC", "DWI"),
        skip_gradcam=args.skip_gradcam,
        skip_occlusion=args.skip_occlusion,
        skip_aggregated=args.skip_aggregated
    )

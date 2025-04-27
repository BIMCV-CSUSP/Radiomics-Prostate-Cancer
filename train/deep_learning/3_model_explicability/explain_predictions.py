#!/usr/bin/env python
import argparse
import importlib
import json
import logging
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")  # backend no interactivo
import matplotlib.pyplot as plt
import numpy as np
import shap
import torch
from monai.data import DataLoader, Dataset

import torch.multiprocessing as mp
mp.set_sharing_strategy("file_system")

from data_loader_for_cv_for_predict import MyDataLoader

# -----------------------------------------------------------------------------
# Utilidades
# -----------------------------------------------------------------------------

def dynamic_import(class_path: str):
    """Importa dinámicamente una clase dado su path completo."""
    module_name, class_name = class_path.rsplit('.', 1)
    module = importlib.import_module(module_name)
    return getattr(module, class_name)


def setup_logger(log_file: Path):
    logger = logging.getLogger("shap_logger")
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    if logger.hasHandlers():
        logger.handlers.clear()
    fh = logging.FileHandler(log_file, mode='w')
    fh.setFormatter(fmt)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(fmt)
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)
    return logger


# -------------------- helpers para 3‑D -> 2‑D --------------------------------

def _middle_slice(t: torch.Tensor) -> torch.Tensor:
    """Recorta la rebanada central de un tensor (N,C,D,H,W) → (N,C,H,W)."""
    if t.ndim == 5:
        mid = t.shape[2] // 2
        return t[:, :, mid, :, :]
    return t  # ya es 4‑D


def _to_numpy_rgb(batch_tensor: torch.Tensor) -> np.ndarray:
    """Convierte (N,C,[D],H,W) → (N,H,W,C) float32 para shap.image_plot."""
    batch_tensor = _middle_slice(batch_tensor)          # (N,C,H,W)
    if batch_tensor.ndim == 3:                          # (C,H,W) single sample
        batch_tensor = batch_tensor.unsqueeze(0)
    return batch_tensor.permute(0, 2, 3, 1).detach().cpu().float().numpy()


def _to_numpy_shap(shap_tensor: torch.Tensor) -> np.ndarray:
    """Idem para tensores SHAP (N,C,[D],H,W) → (N,H,W,C)."""
    shap_tensor = _middle_slice(shap_tensor)
    return shap_tensor.permute(0, 2, 3, 1).detach().cpu().numpy()

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Genera mapas SHAP para un modelo CNN")

    # paths ------------------------------------------------------------------
    parser.add_argument("--model_dir", required=True, type=str,
                        help="Carpeta que contiene best_overall_model.pth")
    parser.add_argument("--config_file", default="../../../../../data/results/deep_learning/config.json", type=str,
                        help="JSON con la configuración de modelos (clave = nombre de carpeta)")
    parser.add_argument("--csv_path", default="../../../../../data/data.csv", type=str,
                        help="CSV con rutas de imágenes, etiquetas y patient_id")
    parser.add_argument("--input_shape", nargs=3, type=int, default=[128, 128, 32],
                        help="Dimensiones de la imagen (C,H,W)")

    # SHAP -------------------------------------------------------------------
    parser.add_argument("--num_background", type=int, default=64,
                        help="Número de imágenes de background")
    parser.add_argument("--num_explain", type=int, default=16,
                        help="Número de imágenes a explicar")

    # runtime ----------------------------------------------------------------
    parser.add_argument("--batch_size", type=int, default=2)
    parser.add_argument("--output_dir", type=str, default="shap_outputs")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")

    args = parser.parse_args()

    # -------------------- rutas y logger -----------------------------------
    model_dir = Path(args.model_dir)
    model_path = model_dir / "best_overall_model.pth"
    if not model_path.is_file():
        raise FileNotFoundError(f"No se encontró {model_path}")

    model_name = model_dir.name
    out_dir = Path(args.output_dir) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(out_dir / "generate_shap.log")
    logger.info("======= Generación SHAP (modelo único) =======")
    logger.info(f"Modelo: {model_name}")

    # -------------------- leer configuración ------------------------------
    with open(args.config_file, "r") as f:
        configs = json.load(f)
    if model_name not in configs:
        logger.error(f"El modelo '{model_name}' no aparece en {args.config_file}")
        raise KeyError(f"Sin configuración para {model_name}")

    config = configs[model_name]
    ModelClass = dynamic_import(config["model"])
    model_args = config.get("model_args", {})
    num_classes = model_args.get("num_classes", 2)

    # -------------------- datos -------------------------------------------
    data_loader = MyDataLoader(
        csv_path=args.csv_path,
        input_shape=tuple(args.input_shape),
        config={"batch_size": args.batch_size, "num_workers": 4},
        transformations=[],
        num_classes=num_classes,
    )
    all_data = data_loader.get_all_data()
    if not all_data:
        raise RuntimeError("No se cargaron datos del CSV")

    rng = np.random.default_rng(42)
    bg_indices = rng.choice(len(all_data), size=min(args.num_background, len(all_data)), replace=False)
    explain_indices = rng.choice(len(all_data), size=min(args.num_explain, len(all_data)), replace=False)

    bg_dataset = Dataset([all_data[i] for i in bg_indices], transform=data_loader.get_transforms(augment=False))
    explain_dataset = Dataset([all_data[i] for i in explain_indices], transform=data_loader.get_transforms(augment=False))

    bg_loader = DataLoader(bg_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    explain_loader = DataLoader(explain_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    device = torch.device("cuda" if (args.device == "auto" and torch.cuda.is_available()) or args.device == "cuda" else "cpu")
    logger.info(f"Dispositivo: {device}")

    # -------------------- cargar modelo -----------------------------------
    model = ModelClass(**model_args)
    model.load_state_dict(torch.load(model_path, map_location="cpu"))
    model.to(device)
    model.eval()
    logger.info(f"Modelo cargado desde: {model_path}")

    # -------------------- background & explain tensors --------------------
    bg_tensors: List[torch.Tensor] = []
    for batch in bg_loader:
        bg_tensors.append(batch["image"])
    background = torch.cat(bg_tensors, dim=0)[: args.num_background].to(device)

    explain_tensors: List[torch.Tensor] = []
    explain_meta: List[dict] = []
    for batch in explain_loader:
        explain_tensors.append(batch["image"])
        explain_meta.extend([
            {"patient_id": pid, "label": torch.argmax(lbl).item()} for pid, lbl in zip(batch["patient_id"], batch["label"])
        ])
    explain_tensor = torch.cat(explain_tensors, dim=0)[: args.num_explain].to(device)

    # Imágenes 2‑D para overlay
    explain_np = _to_numpy_rgb(explain_tensor)

    # -------------------- SHAP -------------------------------------------
    logger.info("Calculando valores SHAP ...")
    explainer = shap.GradientExplainer(model, background)
    with torch.no_grad():
        shap_vals = explainer.shap_values(explain_tensor)   # lista por clase
    logger.info("SHAP calculado")

    # Convertimos cada tensor SHAP a (N,H,W,C)
    shap_vals_2d = [_to_numpy_shap(s) for s in shap_vals]

    # -------------------- visualización ----------------------------------
    shap.image_plot(shap_vals_2d, explain_np, show=False)
    plt.tight_layout()
    plt.savefig(out_dir / "shap_overview.png", dpi=200)
    plt.close()
    logger.info("Guardado shap_overview.png")

    for i in range(args.num_explain):
        plt.figure()
        shap.image_plot([s[i:i+1] for s in shap_vals_2d], explain_np[i:i+1], show=False)
        fname = f"explanation_{i:03d}_pid_{explain_meta[i]['patient_id']}.png"
        plt.savefig(out_dir / fname, dpi=200)
        plt.close()
    logger.info("✔️  Proceso SHAP completado")


if __name__ == "__main__":
    main()
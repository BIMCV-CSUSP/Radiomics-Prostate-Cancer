import os
import pandas as pd

import nibabel as nib
import torch
from torch import as_tensor
from torch.nn.functional import one_hot

from sklearn.model_selection import train_test_split

from monai.data import Dataset, DataLoader
from monai import transforms
from monai.transforms import (
    LoadImaged,
    ResampleToMatchd,
    Resized,
    ScaleIntensityd,
    ConcatItemsd,
    SelectItemsd,
)

class MyDataLoader:
    def __init__(
        self,
        csv_path: str,
        input_shape: tuple = (128, 128, 128),
        train_ratio: float = 0.8,
        seed: int = 42,
        config: dict = None,
        transformations: list = None,
        num_classes: int = 2,
    ):
        """
        csv_path: ruta al CSV con columnas, p.ej.:
            - t2w_path (ruta a la imagen T2)
            - adc_path (ruta a la imagen ADC)
            - hbv_path (ruta a la imagen DWI/HBV)
            - case_csPCa (etiqueta binaria o multi-clase, ajusta según tu caso)
            - (otros campos que estimes necesarios)

        transformations: lista de transformaciones MONAI que se aplicarán
                         sobre la clave "image" (ya concatenada).
                         Si es None o lista vacía, no se aplican.

        num_classes: número de clases si deseas hacer one-hot.
        """
        if config is None:
            config = {}
        self.config_args = config

        # ---------------------------------------------------------------------
        # 1) Leer CSV y preparar datos
        # ---------------------------------------------------------------------
        df = pd.read_csv(csv_path)

        all_data = []
        labels_for_stratify = []

        for _, row in df.iterrows():
            t2_path = "../../../" + row["t2w_path"]
            adc_path = "../../../" + row["adc_path"]
            dwi_path = "../../../" + row["hbv_path"]

            label_value = int(row["case_csPCa"])
            labels_for_stratify.append(label_value)

            label_tensor = one_hot(
                as_tensor([label_value], dtype=torch.long),
                num_classes=num_classes
            ).float().squeeze(0)

            data_dict = {
                "t2": t2_path,
                "adc": adc_path,
                "dwi": dwi_path,
                "label": label_tensor,
            }
            all_data.append(data_dict)

        # ---------------------------------------------------------------------
        # 2) Split estratificado: train y test
        # ---------------------------------------------------------------------
        train_data, test_data = train_test_split(
            all_data,
            train_size=train_ratio,       
            random_state=seed,
            stratify=labels_for_stratify
        )

        self.train_data = train_data
        self.test_data = test_data

        # ---------------------------------------------------------------------
        # 3) Calcular pesos de clase (usando los datos de entrenamiento)
        # ---------------------------------------------------------------------
        class_counts = torch.zeros(num_classes, dtype=torch.long)
        for d in self.train_data:
            idx = torch.argmax(d["label"]).item()
            class_counts[idx] += 1

        train_len = len(self.train_data)
        weights = []
        for c in range(num_classes):
            count_c = class_counts[c].item() if class_counts[c] > 0 else 1e-6
            weights.append(train_len / count_c)

        weights = torch.tensor(weights, dtype=torch.float32)
        self._class_weights = weights / weights.sum()

        # ---------------------------------------------------------------------
        # 4) Definir transformaciones base
        # ---------------------------------------------------------------------
        self.input_shape = input_shape
        self.base_transforms = [
            LoadImaged(keys=["t2", "adc", "dwi"], image_only=True, ensure_channel_first=True),
            ResampleToMatchd(keys=["adc", "dwi"], key_dst="t2", mode=("bilinear", "bilinear")),
            Resized(keys=["t2", "adc", "dwi"], spatial_size=self.input_shape, mode=("trilinear",)*3),
            ScaleIntensityd(keys=["t2", "adc", "dwi"], minv=0.0, maxv=1.0),
            ConcatItemsd(keys=["t2", "adc", "dwi"], name="image", dim=0),
        ]

        self.augment_transforms = []
        if transformations:
            self.augment_transforms.extend(transformations)

        self.select_items = [SelectItemsd(keys=["image", "label"])]

    def __call__(self, partition: str):
        """
        partition: 'train' o 'test'
        """
        if partition not in ["train", "test"]:
            print("Partición no válida. Usa 'train' o 'test'.")
            return None

        if partition == "train":
            data_subset = self.train_data
            transforms_list = self.base_transforms + self.augment_transforms + self.select_items
            self.config_args["shuffle"] = True
        else:
            data_subset = self.test_data
            transforms_list = self.base_transforms + self.select_items
            self.config_args["shuffle"] = False

        dataset = Dataset(data=data_subset, transform=transforms.Compose(transforms_list))
        loader = DataLoader(dataset, **self.config_args)
        return loader

    @property
    def class_weights(self):
        return self._class_weights
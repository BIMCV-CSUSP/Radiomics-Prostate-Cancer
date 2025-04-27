import os
import pandas as pd
import torch
from torch import as_tensor
from torch.nn.functional import one_hot

from monai.data import Dataset
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
        config: dict = None,
        transformations: list = None,
        num_classes: int = 2,
    ):
        if config is None:
            config = {}
        self.config_args = config

        df = pd.read_csv(csv_path)
        all_data = []
        for _, row in df.iterrows():
            t2_path = "../../../../" + row["t2w_path"]
            adc_path = "../../../../" + row["adc_path"]
            dwi_path = "../../../../" + row["hbv_path"]

            label_value = int(row["case_csPCa"])
            label_tensor = one_hot(as_tensor([label_value]), num_classes=num_classes).float().squeeze(0)

            patient_id = row["patient_id"]

            data_dict = {
                "t2": t2_path,
                "adc": adc_path,
                "dwi": dwi_path,
                "label": label_tensor,
                "patient_id": patient_id,
            }
            all_data.append(data_dict)

        self.all_data = all_data

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

    def get_transforms(self, augment: bool = False):
        """
        Devuelve la lista de transformaciones base + (opcionalmente) augmentación.
        """
        if augment:
            return transforms.Compose(self.base_transforms + self.augment_transforms + self.select_items)
        else:
            return transforms.Compose(self.base_transforms + self.select_items)

    def get_all_data(self):
        """
        Devuelve la lista con todos los diccionarios (sin hacer split).
        """
        return self.all_data
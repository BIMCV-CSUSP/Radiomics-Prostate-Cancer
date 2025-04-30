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
    MaskIntensityd,
    EnsureTyped
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
            t2_path = "../../../../../" + row["t2w_path"]
            adc_path = "../../../../../" + row["adc_path"]
            dwi_path = "../../../../../" + row["hbv_path"]
            mask_path = "../../../../../" + row["whole_gland_path"]

            label_value = int(row["case_csPCa"])
            label_tensor = one_hot(as_tensor([label_value]), num_classes=num_classes).float().squeeze(0)

            patient_id = row["patient_id"]

            data_dict = {
                "t2": t2_path,
                "adc": adc_path,
                "dwi": dwi_path,
                "mask": mask_path,  
                "label": label_tensor,
                "patient_id": patient_id,
            }
            all_data.append(data_dict)

        self.all_data = all_data
        self.input_shape = input_shape

        # TRANSFORMACIONES BASE
        self.base_transforms = [
            # 1) Cargamos la imagen y la máscara
            LoadImaged(
                keys=["t2", "adc", "dwi", "mask"],
                image_only=True,
                ensure_channel_first=True,
            ),
            # 2) Resampleamos ADC y DWI y la máscara para que "coincidan" con t2
            ResampleToMatchd(
                keys=["adc", "dwi", "mask"],
                key_dst="t2",
                mode=("bilinear", "bilinear", "nearest")
            ),
            # 3) Redimensionamos (incluida la máscara)
            Resized(
                keys=["t2", "adc", "dwi", "mask"],
                spatial_size=self.input_shape,
                mode=("trilinear", "trilinear", "trilinear", "nearest")
            ),
            # 4) Aplicamos la máscara a t2, adc y dwi
            MaskIntensityd(
                keys=["t2", "adc", "dwi"],
                mask_key="mask",
                select_fn=lambda x: x > 0.5
            ),
            # 5) Normalizamos intensidades (ya solo en la zona enmascarada)
            ScaleIntensityd(keys=["t2", "adc", "dwi"], minv=0.0, maxv=1.0),
            # 6) Concatenamos en un solo tensor 'image'
            ConcatItemsd(keys=["t2", "adc", "dwi"], name="image", dim=0),
            EnsureTyped(keys=["image"], track_meta=False),
        ]

        self.augment_transforms = []
        if transformations:
            self.augment_transforms.extend(transformations)

        self.select_items = [SelectItemsd(keys=["image", "label", "patient_id"])]

    def get_transforms(self, augment: bool = False):
        """
        Devuelve la lista de transformaciones base + (opcionalmente) augmentación.
        """
        if augment:
            return transforms.Compose(
                self.base_transforms + self.augment_transforms + self.select_items
            )
        else:
            return transforms.Compose(self.base_transforms + self.select_items)

    def get_all_data(self):
        """
        Devuelve la lista con todos los diccionarios (sin hacer split).
        """
        return self.all_data
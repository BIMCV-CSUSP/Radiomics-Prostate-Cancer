from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[4]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
COMMON_DIR = PROJECT_ROOT / "train" / "common"
if str(COMMON_DIR) not in sys.path:
    sys.path.append(str(COMMON_DIR))

import pandas as pd
from torch import as_tensor
from torch.nn.functional import one_hot

from monai import transforms
from monai.transforms import (
    ConcatItemsd,
    LoadImaged,
    MaskIntensityd,
    ResampleToMatchd,
    Resized,
    ScaleIntensityd,
    SelectItemsd,
)

from runtime_utils import infer_project_path_from_csv


class MyDataLoader:
    """Load ROI volumes for interpretability while keeping both masked and full images."""

    def __init__(
        self,
        csv_path: str,
        input_shape: tuple = (128, 128, 128),
        config: dict | None = None,
        transformations: list | None = None,
        num_classes: int = 2,
    ):
        if config is None:
            config = {}

        self.config_args = config
        self.csv_path = Path(csv_path).resolve()
        self.input_shape = input_shape

        df = pd.read_csv(self.csv_path)
        all_data = []
        for _, row in df.iterrows():
            label_value = int(row["case_csPCa"])
            label_tensor = one_hot(as_tensor([label_value]), num_classes=num_classes).float().squeeze(0)

            all_data.append(
                {
                    "t2": str(infer_project_path_from_csv(self.csv_path, row["t2w_path"])),
                    "adc": str(infer_project_path_from_csv(self.csv_path, row["adc_path"])),
                    "dwi": str(infer_project_path_from_csv(self.csv_path, row["hbv_path"])),
                    "mask": str(infer_project_path_from_csv(self.csv_path, row["whole_gland_path"])),
                    "label": label_tensor,
                    "patient_id": row["patient_id"],
                    "study_id": row["study_id"],
                }
            )

        self.all_data = all_data
        self.base_transforms = [
            LoadImaged(keys=["t2", "adc", "dwi", "mask"], image_only=True, ensure_channel_first=True),
            ResampleToMatchd(
                keys=["adc", "dwi", "mask"],
                key_dst="t2",
                mode=("bilinear", "bilinear", "nearest"),
            ),
            Resized(
                keys=["t2", "adc", "dwi", "mask"],
                spatial_size=self.input_shape,
                mode=("trilinear", "trilinear", "trilinear", "nearest"),
            ),
            ConcatItemsd(keys=["t2", "adc", "dwi"], name="image_full", dim=0),
            MaskIntensityd(keys=["t2", "adc", "dwi"], mask_key="mask", select_fn=lambda x: x > 0.5),
            ScaleIntensityd(keys=["t2", "adc", "dwi"], minv=0.0, maxv=1.0),
            ConcatItemsd(keys=["t2", "adc", "dwi"], name="image", dim=0),
        ]

        self.augment_transforms = list(transformations) if transformations else []
        self.select_items = [
            SelectItemsd(keys=["image", "image_full", "mask", "label", "patient_id", "study_id"])
        ]

    def get_transforms(self, augment: bool = False):
        """Return the MONAI transform pipeline, optionally including augmentation."""

        if augment:
            return transforms.Compose(self.base_transforms + self.augment_transforms + self.select_items)
        return transforms.Compose(self.base_transforms + self.select_items)

    def get_all_data(self):
        """Return the dataset entries before any split is applied."""

        return self.all_data

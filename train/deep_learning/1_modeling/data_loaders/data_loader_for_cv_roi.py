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
    MaskIntensityd, # ← Transformación adicional respecto a la versión "org"
)

class MyDataLoader:
    """
    Clase para cargar y preprocesar imágenes de resonancia magnética multiparamétricas.
    Esta versión aplica una máscara de la glándula prostática para procesar solo la ROI.
    
    Esta clase:
    1. Carga imágenes T2, ADC, DWI y la máscara de segmentación de la glándula
    2. Aplica preprocesamiento (carga, remuestreo, redimensionado)
    3. Aplica la máscara para restringir el análisis a la región prostática
    4. Normaliza intensidades y concatena las tres secuencias
    """
        
    def __init__(
        self,
        csv_path: str,
        input_shape: tuple = (128, 128, 128),
        config: dict = None,
        transformations: list = None,
        num_classes: int = 2,
    ):
        """
        Inicializa el cargador de datos.
        
        Args:
            csv_path: Ruta al archivo CSV con información de las imágenes
            input_shape: Dimensiones espaciales deseadas para las imágenes
            config: Configuración adicional para el DataLoader
            transformations: Transformaciones adicionales a aplicar (aumentación)
            num_classes: Número de clases para codificación one-hot de etiquetas
        """

        if config is None:
            config = {}
        self.config_args = config

        # Cargar el CSV con las rutas a las imágenes
        df = pd.read_csv(csv_path)
        all_data = []

        # Procesar cada fila del CSV
        for _, row in df.iterrows():
            # Construir rutas completas a imágenes
            t2_path = "../../../../" + row["t2w_path"]
            adc_path = "../../../../" + row["adc_path"]
            dwi_path = "../../../../" + row["hbv_path"]

            # Ruta a la máscara de la glándula prostática (diferencia clave con data_loader_org)
            mask_path = "../../../../" + row["whole_gland_path"]
            
            # Convertir etiqueta a tensor one-hot
            label_value = int(row["case_csPCa"])
            label_tensor = one_hot(as_tensor([label_value]), num_classes=num_classes).float().squeeze(0)

            patient_id = row["patient_id"]

            # Crear diccionario con rutas e información del paciente
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
            # 1) Cargamos todas las imágenes y la máscara
            LoadImaged(
                keys=["t2", "adc", "dwi", "mask"],  # Incluye la máscara
                image_only=True,
                ensure_channel_first=True,
            ),
            
            # 2) Remuestreamos ADC, DWI y la máscara para que coincidan con el espacio de t2
            # Nota: La máscara usa interpolación "nearest" para preservar los valores binarios
            ResampleToMatchd(
                keys=["adc", "dwi", "mask"],
                key_dst="t2",
                mode=("bilinear", "bilinear", "nearest")  
            ),
            
            # 3) Redimensionamos todas las imágenes y la máscara al tamaño deseado
            # Nota: La máscara usa interpolación "nearest" para mantener bordes nítidos
            Resized(
                keys=["t2", "adc", "dwi", "mask"],
                spatial_size=self.input_shape,
                mode=("trilinear", "trilinear", "trilinear", "nearest")
            ),
            
            # 4) Aplicamos la máscara a cada modalidad (T2, ADC, DWI)
            # Esta es la principal diferencia: restringimos el análisis a la región prostática
            MaskIntensityd(
                keys=["t2", "adc", "dwi"],
                mask_key="mask",
                select_fn=lambda x: x > 0.5  # Umbral para binarizar la máscara
            ),
            
            # 5) Normalizamos intensidades solo en la región enmascarada
            ScaleIntensityd(keys=["t2", "adc", "dwi"], minv=0.0, maxv=1.0),
            
            # 6) Concatenamos las tres modalidades en un único tensor
            ConcatItemsd(keys=["t2", "adc", "dwi"], name="image", dim=0),
        ]

        # Transformaciones adicionales (aumentación de datos)
        self.augment_transforms = []
        if transformations:
            self.augment_transforms.extend(transformations)

        self.select_items = [SelectItemsd(keys=["image", "label"])]

    def get_transforms(self, augment: bool = False):
        """
        Devuelve la lista de transformaciones base + (opcionalmente) augmentación.
        
        Args:
            augment: Si True, incluye transformaciones de aumentación
            
        Returns:
            Composición de transformaciones MONAI
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
        
        Returns:
            Lista de diccionarios con datos de cada paciente
        """
        return self.all_data
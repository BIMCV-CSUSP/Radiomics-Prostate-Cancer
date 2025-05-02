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
    """
    Clase para cargar y preprocesar imágenes de resonancia magnética multiparamétricas.
    Esta versión procesa la imagen completa sin restringirse a una región de interés.
    
    Esta clase:
    1. Carga imágenes T2, ADC y DWI desde rutas especificadas en un CSV
    2. Aplica preprocesamiento básico (carga, remuestreo, redimensionado, normalización)
    3. Concatena las tres secuencias en un único tensor multicanal
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
            input_shape: Dimensiones espaciales deseadas para las imágenes (ancho, alto, profundidad)
            config: Configuración adicional para el DataLoader
            transformations: Transformaciones adicionales a aplicar (aumentación de datos)
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

            # Convertir etiqueta a tensor one-hot (ej: 0 -> [1,0], 1 -> [0,1] para binario)
            label_value = int(row["case_csPCa"])
            label_tensor = one_hot(as_tensor([label_value]), num_classes=num_classes).float().squeeze(0)

            patient_id = row["patient_id"]

            # Crear diccionario con rutas e información del paciente
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

        # Transformaciones base que siempre se aplican, en orden:
        self.base_transforms = [
            # 1. Cargar imágenes multimodales, asegurando que tengan canal como primera dimensión
            LoadImaged(keys=["t2", "adc", "dwi"], image_only=True, ensure_channel_first=True),
            
            # 2. Remuestrear ADC y DWI para que coincidan con T2 (referencia espacial)
            ResampleToMatchd(keys=["adc", "dwi"], key_dst="t2", mode=("bilinear", "bilinear")),
            
            # 3. Redimensionar todas las imágenes al tamaño especificado usando interpolación trilineal
            Resized(keys=["t2", "adc", "dwi"], spatial_size=self.input_shape, mode=("trilinear",)*3),
            
            # 4. Normalizar intensidades en rango [0,1]
            ScaleIntensityd(keys=["t2", "adc", "dwi"], minv=0.0, maxv=1.0),
            
            # 5. Concatenar las tres modalidades en un único tensor "image" (canal 0=T2, 1=ADC, 2=DWI)
            ConcatItemsd(keys=["t2", "adc", "dwi"], name="image", dim=0),
        ]

        # Transformaciones adicionales (aumentación de datos)
        self.augment_transforms = []
        if transformations:
            self.augment_transforms.extend(transformations)

        # Transformación final para seleccionar solo imagen y etiqueta
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
            return transforms.Compose(self.base_transforms + self.augment_transforms + self.select_items)
        else:
            return transforms.Compose(self.base_transforms + self.select_items)

    def get_all_data(self):
        """
        Devuelve la lista con todos los diccionarios (sin hacer split).
        
        Returns:
            Lista de diccionarios con datos de cada paciente
        """
        return self.all_data
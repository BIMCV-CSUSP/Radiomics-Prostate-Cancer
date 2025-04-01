import nibabel as nib
from bimcv_aikit.monai.transforms import DeleteBlackSlices
from monai import transforms
from monai.data import CacheDataset, DataLoader
from numpy import array, float32, unique
from pandas import read_csv
from sklearn.utils.class_weight import compute_class_weight
from torch import as_tensor
from torch.nn.functional import one_hot

# Diccionario de configuración por defecto (vacío en este caso)
config_default = {}

class Dataloader:
    def __init__(
        self,
        path: str,                       # Ruta al archivo CSV con la información de las imágenes y etiquetas
        sep: str = ",",                  # Separador del archivo CSV
        classes: list = ["noCsPCa", "CsPCa"],  # Lista de clases (opcional)
        img_columns=["t2", "adc", "dwi"],      # Nombre de las columnas con rutas a las imágenes
        test_run: bool = False,          # Si True, se usa un subconjunto reducido de datos
        input_shape: str = "(128, 128, 128)",  # Tamaño al que se redimensionarán las imágenes
        rand_prob: int = 0.5,            # Probabilidad utilizada en algunas transformaciones (no se usa actualmente)
        partition_column: str = "partition",   # Columna del CSV que indica la partición (train, val, test)
        config: dict = config_default,   # Configuraciones adicionales para el DataLoader
    ):
        # Lee el CSV en un dataframe de pandas
        df = read_csv(path, sep=sep)

        # Calcula el número de clases a partir de las etiquetas
        n_classes = len(unique(df["label"].values))

        # Agrupa el dataframe por la columna de partición
        self.groupby = df.groupby(partition_column)

        # Calcula los pesos de clase para lidiar con desbalances
        # Se asume que existe una partición 'train' en el CSV
        self._class_weights = compute_class_weight(
            class_weight="balanced",
            classes=unique(self.groupby.get_group("train")["label"].values),
            y=self.groupby.get_group("train")["label"].values,
        )

        # Transforms para el conjunto de entrenamiento
        self.train_transforms = transforms.Compose([
            transforms.LoadImaged(keys=img_columns, image_only=True, ensure_channel_first=True),
            # Ajusta "adc" y "dwi" para que tengan la misma resolución y dimensiones que "t2"
            transforms.ResampleToMatchd(
                keys=["adc", "dwi"],
                key_dst="t2",
                mode=("bilinear", "bilinear"),
            ),
            # Parte la dimensión de "dwi" (asumiendo que tiene múltiples canales en la última dimensión)
            transforms.SplitDimd(
                keys=["dwi"],
                keepdim=True,
            ),
            # Redimensiona "t2", "dwi_0" (el resultado de SplitDim), y "adc" a la forma especificada
            transforms.Resized(
                keys=['t2', 'dwi_0', 'adc'],
                spatial_size=eval(input_shape),
                mode=("trilinear", "trilinear", "trilinear"),
            ),
            # Escala la intensidad de las imágenes al rango [0, 1]
            transforms.ScaleIntensityd(keys=['t2','dwi_0','adc'], minv=0.0, maxv=1.0, allow_missing_keys=True),
            # Concatena las tres imágenes (t2, dwi_0, adc) en un solo tensor con canales
            transforms.ConcatItemsd(keys=['t2', 'dwi_0', 'adc'], name="image", dim=0),
            # Selecciona solamente "image" y "label" para la salida final
            transforms.SelectItemsd(keys=["image", "label"])
        ])

        # Transforms para el conjunto de validación (similares a los de entrenamiento, sin augmentations)
        self.val_transforms = transforms.Compose([
            transforms.LoadImaged(keys=img_columns, image_only=True, ensure_channel_first=True),
            transforms.ResampleToMatchd(
                keys=["adc", "dwi"],
                key_dst="t2",
                mode=("bilinear", "bilinear"),
            ),
            transforms.SplitDimd(
                keys=["dwi"],
                keepdim=True,
            ),
            transforms.Resized(
                keys=['t2','dwi_0','adc'],
                spatial_size=eval(input_shape),
                mode=("trilinear", "trilinear", "trilinear"),
            ),
            transforms.ScaleIntensityd(keys=['t2','dwi_0','adc'], minv=0.0, maxv=1.0, allow_missing_keys=True),
            transforms.ConcatItemsd(keys=['t2','dwi_0','adc'], name="image", dim=0),
            transforms.SelectItemsd(keys=["image", "label"])
        ])

        self.test_run = test_run
        self.config_args = config

    def __call__(self, partition: str):
        # Si se pasa 'None' como partición, retorna None
        if partition == "None":
            print("Partition not found")
            return None

        # Crea una lista de diccionarios con las rutas a las imágenes y la etiqueta one-hot
        data = [
            {"t2": t2, "adc": adc, "dwi": dwi, "label": label}
            for t2, adc, dwi, label in zip(
                self.groupby.get_group(partition)["filepath_t2w_cropped"].values,
                self.groupby.get_group(partition)["filepath_adc_cropped"].values,
                self.groupby.get_group(partition)["filepath_hbv_cropped"].values,
                # Convierte las etiquetas a tensores one-hot
                one_hot(as_tensor(self.groupby.get_group(partition)["label"].values, dtype=int)).float(),
            )
        ]

        # Si es un test_run, limita la cantidad de datos a 16
        if self.test_run:
            data = data[:16]

        # Si es entrenamiento, usa los train_transforms, de lo contrario usa val_transforms
        if partition == "train":
            dataset = CacheDataset(data=data, transform=self.train_transforms, num_workers=7)
        else:
            dataset = CacheDataset(data=data, transform=self.val_transforms, num_workers=7)
            # Forzamos a no barajar los datos en validación
            self.config_args["shuffle"] = False

        # Retorna un DataLoader a partir del dataset y las configuraciones proporcionadas
        return DataLoader(dataset, **self.config_args)

    @property
    def class_weights(self):
        # Retorna los pesos de clase calculados en el constructor
        return self._class_weights
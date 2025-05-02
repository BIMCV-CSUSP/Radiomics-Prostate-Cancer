# Estructura de archivos y metodología en Deep Learning

## Estructura de archivos

```
1_modeling/                                  
├── config.json                              # Parámetros de entrenamiento
├── train.py                                 # Script principal de entrenamiento
└── data_loaders/                            # Scripts de carga de datos para entrenamiento
    ├── data_loader_for_cv_org.py
    └── data_loader_for_cv_roi.py

2_analyse_results/                           
├── predict_&_analyse_probs/
│   ├── 1_predict.py                         # Generación de predicciones
│   ├── 2_analyse_predictions.py             # Análisis de la distribución de las predicciones
│   └── z_data_loader_for_cv_for_predict.py  # Carga de datos para evaluación
├── simple_statistical_analysis/
│   └── compare_models.py                    # Comparación estadística entre métricas de los modelos

3_model_explicability/
└── explain_predictions.py                  # Análisis de explicabilidad de predicciones
```

## Metodología

El proceso de análisis mediante Deep Learning se organiza en tres fases principales:

### 1. Entrenamiento y validación ([`train.py`](./1_modeling/train.py))

Este script entrena modelos convolucionales mediante validación cruzada estratificada con grupos.

1. Cargadores de datos

    Se pueden usar dos estrategias de carga de datos, según el valor del parámetro `--mode`:

    - [`data_loader_for_cv_org.py`](./1_modeling/data_loaders/data_loader_for_cv_org.py): utiliza la imagen completa como entrada.  
    - [`data_loader_for_cv_roi.py`](./1_modeling/data_loaders/data_loader_for_cv_roi.py): usa únicamente la región correspondiente a la glándula prostática.

2. Modelos y configuraciones

    - Los modelos utilizados se definen en `config.json`, donde cada clave corresponde a una configuración concreta.  
    - Es posible definir modelos con o sin transformaciones adicionales, especificadas en el campo `extra_transforms`.

3. Entrenamiento y evaluación

    - Se entrena el modelo seleccionado mediante validación cruzada, con cálculo de métricas clásicas: AUC, F1 (macro y binario), accuracy, sensibilidad, especificidad, MCC, etc.  
    - Se aplica *early stopping* basado en el AUC de validación.  
    - Por cada *split* se guarda el mejor modelo, y también se identifica y almacena el mejor modelo global.

#### Archivos generados

Los resultados de esta fase se almacenan en [`artifacts/deep_learning`](../../artifacts/deep_learning/), organizados por modalidad (`full` o `gland`) y configuración:

- Checkpoints de modelos por split y mejor modelo global (`.pth`)  
- Archivos `.csv` con métricas de entrenamiento y validación  
- Logs de entrenamiento en formato `.log`


### 2. Comparación de modelos

Se utilizan dos estrategias complementarias para comparar el rendimiento de los distintos modelos entrenados:

1. Comparación simple

    - El script [`compare_models.py`](./2_analyse_results/simple_statistical_analysis/compare_models.py) analiza los resultados directamente a partir de los archivos de entrenamiento.  
    - Se identifican los modelos más prometedores mediante la visualización de métricas como AUC, F1, accuracy, sensibilidad, etc.  
    - Se aplican análisis estadísticos como el test de Friedman y comparaciones *post-hoc* (Wilcoxon con corrección de Holm).

2. Comparación mediante predicciones

    - El script [`1_predict.py`](./2_analyse_results/predict_&_analyse_probs/1_predict.py) genera las predicciones de los modelos entrenados sobre sus conjuntos de validación, incluyendo las probabilidades asignadas a cada clase para cada paciente.  
    - Estas probabilidades se comparan entre modelos utilizando [`2_analyse_predictions.py`](./2_analyse_results/predict_&_analyse_probs/2_analyse_predictions.py), que realiza un análisis estadístico acompañado de visualizaciones (boxplots, heatmaps de *p-valores*).

#### Archivos generados

Los resultados se almacenan en la carpeta [`results/deep_learning/model_comparison`](../../results/deep_learning/model_comparison/), incluyendo:

- Visualizaciones por métrica (gráficos de radar, barras, boxplots).  
- Archivos `.csv` con métricas combinadas y estadísticas resumen.  
- Informes `.txt` con resultados de los análisis estadísticos.

### 3. Explicabilidad del modelo

<!-- TO-DO -->
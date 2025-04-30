# Estructura de archivos y metodología en radiómica

## Estructura de archivos

```
├───1_extract_radiomics
│   ├── extract_radiomics.py                    # Script para extracción de características radiómicas
│   ├── Params_ADC.yaml                         # Parámetros de extracción para la secuencia ADC
│   ├── Params_DWI.yaml                         # Parámetros de extracción para la secuencia DWI
│   └── Params_T2w.yaml                         # Parámetros de extracción para la secuencia T2W
└───2_modeling
    ├── 1_train_and_evaluate.py                 # Entrenamiento y validación inicial de modelos
    ├── 2_model_differences.py                  # Comparación entre diferentes modelos entrenados
    └── 3_retrain_best_model_and_evaluate.py    # Retrain del mejor modelo y evaluación final
```

## Metodología

El proceso de extracción y análisis radiómico se divide en tres fases principales:

### 1. Extracción de características ([`1_extract_radiomics.py`](./1_extract_radiomics/extract_radiomics.py))

Este script realiza la extracción paralela de características radiómicas:

1. **Preprocesamiento de imágenes**

    - Corrección de campo de sesgo N4 para normalizar intensidades no uniformes  
    - Reducción de ruido mediante filtro de difusión anisotrópica

2. **Extracción de características**. Se procesan 3 secuencias (T2W, ADC, DWI) con 2 enfoques espaciales:

    * **Enfoque glandular**: análisis limitado a la próstata  
    * **Enfoque completo**: análisis de toda la imagen

#### Archivos generados

El script produce un total de 6 archivos `.csv`, que se almacenan en [`artifacts/radiomics/`](../../artifacts/radiomics/), cada uno con las características extraídas por secuencia y enfoque:

- `features_t2_gland.csv`  
- `features_adc_gland.csv`  
- `features_dwi_gland.csv`  
- `features_t2_full.csv`  
- `features_adc_full.csv`  
- `features_dwi_full.csv`

### 2. Fusión de características ([`concatenate_data.ipynb`](../../artifacts/radiomics/concatenate_data.ipynb))

Este notebook combina los seis archivos CSV generados en la fase anterior en dos conjuntos unificados:

1. **Proceso de fusión**:
   * Unificación de características por `paciente_estudio`
   * Adición de prefijos según modalidad (adc_, dwi_, t2_)
   * Preservación de columnas de identificación y etiquetas

2. **Archivos generados** en `concatenate_data/`:
   * `features_all_gland.csv`: Dataset combinado de características de glándula prostática
   * `features_all_full.csv`: Dataset combinado de características de imagen completa


### 3. Construcción, validación y optimización de los modelos

Se entrenan y evalúan distintos clasificadores clásicos utilizando las características extraídas, incorporando un análisis de explicabilidad sobre el modelo con mejor rendimiento.

1. **Entrenamiento y validación ([`1_train_and_evaluate.py`](./2_modeling/1_train_and_evaluate.py))**. Se entrenan seis clasificadores (SVM, LR, RF, NB, KNN, GB) usando validación cruzada estratificada con grupos, repetida múltiples veces.

2. **Comparación estadística ([`2_model_differences.py`](./2_modeling/2_model_differences.py))**. Se evalúan las diferencias de rendimiento entre modelos mediante el test de Friedman y comparaciones post-hoc (Wilcoxon con corrección de Holm).

3. **Optimización y explicabilidad ([`3_retrain_best_model_and_evaluate.py`](./2_modeling/3_retrain_best_model_and_evaluate.py))**. El mejor modelo se refina con búsqueda bayesiana, calibración y análisis de explicabilidad con SHAP y LIME.

Los resultados generados en esta fase se almacenan en la carpeta [`results/radiomics`](../../results/radiomics).
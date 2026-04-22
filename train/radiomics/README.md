# Flujo de trabajo de radiómica

Esta carpeta contiene la rama de radiómica del proyecto. El objetivo es clasificar cáncer de próstata clínicamente significativo (`csPCa`) a partir de RM multiparamétrica (`T2W`, `ADC`, `DWI/HBV`) usando características radiómicas y modelos clásicos de aprendizaje automático.

La etiqueta binaria objetivo es:

- `0`: no clínicamente significativo
- `1`: clínicamente significativo (`ISUP >= 2`)

## Estructura

```text
├── 1_extract_radiomics
│   ├── extract_radiomics.py
│   ├── Params_T2w.yaml
│   ├── Params_ADC.yaml
│   └── Params_DWI.yaml
└── 2_modeling
    ├── 0_build_concatenated_feature_table.py
    ├── 1_train_and_evaluate.py
    ├── 2_model_differences.py
    ├── 2a_gland_vs_full_differences.py
    └── 3_retrain_best_model_and_evaluate.py
```

## Paso a paso del pipeline

### 1. Extraer características radiómicas por modalidad

Script: [`1_extract_radiomics/extract_radiomics.py`](./1_extract_radiomics/extract_radiomics.py)

El script lee `artifacts/data.csv` y, para cada estudio:

1. Carga las tres secuencias: `T2W`, `ADC` y `DWI/HBV`.
2. Carga la máscara de glándula prostática.
3. Preprocesa cada imagen:
   - conversión a `float32`
   - corrección de inhomogeneidad con `N4`
   - reducción de ruido con difusión anisotrópica
4. Extrae características con dos enfoques:
   - `gland`: solo dentro de la glándula prostática
   - `full`: en toda la imagen usando una máscara de unos
5. Guarda un CSV por modalidad y por enfoque.

Archivos generados en `artifacts/radiomics/`:

- `features_t2_gland.csv`
- `features_adc_gland.csv`
- `features_dwi_gland.csv`
- `features_t2_full.csv`
- `features_adc_full.csv`
- `features_dwi_full.csv`

### 2. Concatenar modalidades en una sola tabla de modelado

Script: [`2_modeling/0_build_concatenated_feature_table.py`](./2_modeling/0_build_concatenated_feature_table.py)

Este script une los CSV anteriores para construir una tabla final por enfoque espacial:

- `features_all_gland.csv`
- `features_all_full.csv`

Qué hace exactamente:

1. Carga los CSV de `T2`, `ADC` y `DWI`.
2. Elimina columnas diagnósticas de PyRadiomics (`diagnostics_*`).
3. Mantiene `patient_id`, `study_id` y `label`.
4. Añade prefijos por modalidad para evitar colisiones:
   - `t2_...`
   - `adc_...`
   - `dwi_...`
5. Conserva las variables de forma solo de una modalidad de referencia para no duplicarlas tres veces.
6. Genera `sample_id = patient_id + "_" + study_id`.

## 3. Entrenamiento principal con validación cruzada repetida

Script: [`2_modeling/1_train_and_evaluate.py`](./2_modeling/1_train_and_evaluate.py)

Aquí se comparan seis clasificadores:

- `SVM`
- `Logistic Regression`
- `Random Forest`
- `Naive Bayes`
- `KNN`
- `Gradient Boosting`

El protocolo por defecto es:

- `5 folds`
- `10 repeticiones`
- agrupamiento por `patient_id`

Eso significa que cada clasificador se evalúa en `50` folds de validación y que nunca deberían quedar estudios del mismo paciente repartidos entre train y validación.

Además, el script genera el plan de folds una sola vez y usa exactamente los mismos folds para todos los modelos. Eso hace que la comparación entre clasificadores sea más justa.

## 4. Cómo se hace la selección de características

Esta es la parte más importante del pipeline y también la que suele generar más confusión.

### Idea principal

La selección de características no se hace una sola vez sobre todo el dataset. Se hace dentro de cada fold usando solo el conjunto de entrenamiento de ese fold.

Eso evita leakage. En otras palabras:

- la validación no participa en la selección
- la validación solo se usa al final para medir rendimiento

### Secuencia exacta dentro de cada fold

Si se usa `--feature_strategy most_discriminant`, el código hace esto:

1. Toma solo las columnas numéricas radiómicas.
2. Elimina metadatos y columnas no válidas para modelado:
   - `patient_id`
   - `study_id`
   - `label`
   - `sample_id`
   - `mask_type`
   - columnas `diagnostics_*`
3. Se queda únicamente con el conjunto de entrenamiento del fold actual.
4. Puntúa cada característica de forma univariada:
   - descarta variables sin variación o con demasiados valores inválidos
   - intenta comprobar normalidad
   - si la distribución parece aproximadamente normal, usa `t-test`
   - si no, usa `Mann-Whitney U`
   - además calcula `AUC` univariado para ordenar las variables
   - calcula también un umbral óptimo univariado con índice de Youden
5. Corrige los `p-values` por comparaciones múltiples con Benjamini-Hochberg.
   - si hay variables con `q <= fdr_alpha`, esas forman el pool preferido
   - si no sobrevive ninguna, el script usa como respaldo las variables válidas ordenadas por relevancia univariada
6. Calcula cuántas variables puede permitirse conservar en ese fold.
   - no es un número fijo rígido
   - depende del tamaño del entrenamiento
   - depende también del número de casos de la clase minoritaria
   - esto busca no seleccionar demasiadas variables para la cantidad de datos disponible
7. Elimina redundancia por correlación.
   - ordena las variables candidatas por relevancia
   - recorre la lista de forma greedily
   - si una variable está muy correlacionada con otra ya aceptada, se descarta
   - por defecto usa correlación absoluta de Pearson con umbral `0.90`
8. Se queda con las mejores variables restantes hasta alcanzar el límite permitido.
9. Entrena el clasificador con ese subconjunto.
10. Evalúa en la partición de validación de ese fold.

### Qué implica esto en la práctica

- El conjunto de variables puede cambiar de un fold a otro.
- Eso no es un error; es lo esperable cuando la selección se hace correctamente dentro de cada train fold.
- Como el plan de folds y la selección se calculan una vez y luego se reutilizan en todos los modelos, todos los clasificadores compiten bajo las mismas condiciones.

### Qué archivos deja la selección de características

En `results/radiomics/<feature_strategy>/<mode>/feature_selection/` se guardan salidas como:

- `selected_features_by_fold.csv`: detalle completo por fold
- `feature_selection_frequency.csv`: frecuencia con que aparece cada variable
- `top_selected_features.txt`: variables más estables por clasificador
- `recommended_features_by_classifier.txt`: lista resumida por clasificador

Esto sirve para responder dos preguntas distintas:

- qué variables usó realmente el modelo en cada fold
- qué variables fueron más estables a través de los folds

## 5. Qué pasa después de entrenar los `5 folds x 10 repeticiones`

Después de terminar los 50 folds por clasificador, el script no se detiene. Hace varias cosas más.

### a. Guarda resultados fold a fold

Se exportan métricas de train y validación por fold:

- `AUC`
- `F1`
- `balanced accuracy`
- `MCC`
- `kappa`
- `sensibilidad`
- `especificidad`
- `PPV`
- `NPV`

También guarda las predicciones de validación y la lista de variables usadas en cada fold.

### b. Construye predicciones out-of-fold

Primero genera una tabla plana con una fila por caso validado. Luego agrega las predicciones repetidas.

Esto es importante porque con `10` repeticiones un mismo caso entra a validación varias veces. El script entonces:

1. junta todas esas predicciones
2. promedia la probabilidad predicha de clase positiva por caso y por clasificador
3. aplica el umbral de clasificación, por defecto `0.5`

El resultado es una predicción agregada `out-of-fold` por caso.

### c. Resume el rendimiento a nivel paciente

Con esas predicciones agregadas calcula métricas globales más estables y genera:

- `summary_metrics.csv`
- `auc_ci_summary.csv`
- curvas ROC agregadas
- matrices de confusión agregadas

Además hace bootstrap estratificado a nivel paciente para obtener intervalos de confianza.

### d. Compara estadísticamente los clasificadores

Si activas `--calculate_differences`, se ejecuta [`2_model_differences.py`](./2_modeling/2_model_differences.py):

- Friedman para diferencia global entre modelos
- Wilcoxon pareado con corrección de Holm para comparaciones por pares

Si quieres comparar `gland` contra `full`, puedes usar [`2a_gland_vs_full_differences.py`](./2_modeling/2a_gland_vs_full_differences.py).

## 6. Qué se hace después con el mejor modelo

Script: [`2_modeling/3_retrain_best_model_and_evaluate.py`](./2_modeling/3_retrain_best_model_and_evaluate.py)

Una vez identificas el clasificador con mejor rendimiento, el pipeline pasa a una etapa final más parecida a evaluación definitiva.

### Flujo de esta etapa

1. Hace un split final `80/20` por grupos (`GroupShuffleSplit`).
2. Vuelve a hacer selección de características usando solo el `train` de ese split final.
3. Restringe `train` y `test` a esas variables.
4. Optimiza hiperparámetros con `BayesSearchCV` usando folds agrupados dentro del train.
5. Guarda el mejor estimador entrenado.
6. Evalúa el modelo no calibrado en el hold-out test.
7. Calcula intervalos de confianza por bootstrap en el test.
8. Calibra probabilidades con Platt scaling (`sigmoid`).
9. Vuelve a evaluar el modelo calibrado.
10. Barre umbrales y reporta el que maximiza `F1`.
11. Genera explicabilidad con `SHAP` y `LIME` sobre train y test.

### Cómo interpretar esta fase

La validación cruzada repetida sirve para comparar modelos de manera robusta. El hold-out final sirve para:

- refinar el mejor clasificador
- obtener una evaluación final separada
- producir un modelo interpretable y exportable

## Salidas principales

Dependiendo de las opciones, en `results/radiomics/<feature_strategy>/<mode>/` aparecen:

- métricas por fold
- predicciones por fold
- predicciones `out-of-fold` planas y agregadas
- selección de características por fold
- resumen agregado con intervalos de confianza
- curvas ROC
- comparación estadística entre modelos
- carpeta del mejor modelo reentrenado con:
  - `best_estimator.pkl`
  - `report.txt`
  - curvas de calibración
  - matrices de confusión
  - explicabilidad `SHAP` y `LIME`

## Ejemplo de ejecución

```bash
python train/radiomics/2_modeling/1_train_and_evaluate.py \
  --csv features_all_gland.csv \
  --data_pre artifacts/radiomics \
  --results_base results/radiomics \
  --feature_strategy most_discriminant \
  --n_splits 5 \
  --n_repeats 10 \
  --bootstrap_iterations 1000 \
  --ci_level 0.95 \
  --classification_threshold 0.5 \
  --min_features 10 \
  --max_features_cap 60 \
  --samples_per_feature 25 \
  --minority_samples_per_feature 8 \
  --fdr_alpha 0.05 \
  --correlation_threshold 0.90 \
  --selection_n_jobs 8 \
  --search_n_jobs 8 \
  --search_iterations 50 \
  --calculate_differences \
  --fine_tune_best_model
```

## Nota metodológica importante

En el script final de hold-out, el barrido de umbral se hace sobre el propio conjunto de test. Eso puede servir como análisis exploratorio, pero si el objetivo es reportar una evaluación final estrictamente no sesgada, el umbral debería fijarse dentro de entrenamiento o en una validación adicional, no en el test final.

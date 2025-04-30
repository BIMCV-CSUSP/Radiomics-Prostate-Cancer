## Concatenación de las características radiómicas 

El notebook `concatenate_data.ipynb` agrupa las características radiómicas extraídas de las diferentes secuencias de RMmp (ADC, DWI, T2) en archivos unificados, organizados por región anatómica.

Toma como entrada los siguientes archivos CSV:

- Región de la glándula prostática: `features_adc_gland.csv`, `features_dwi_gland.csv`, `features_t2_gland.csv`
- Imagen completa: `features_adc_full.csv`, `features_dwi_full.csv`, `features_t2_full.csv`

Y genera como salida:

- `features_all_gland.csv`: Dataset combinado de características de la glándula prostática
- `features_all_full.csv`: Dataset combinado de características de la imagen completa

<div style="margin-top: 25px;"></div>

----

<div style="margin-top: 25px;"></div>

> **Nota:** Los archivos resultantes no están incluidos en este repositorio debido a limitaciones de tamaño de GitHub. Pueden generarse ejecutando el notebook.

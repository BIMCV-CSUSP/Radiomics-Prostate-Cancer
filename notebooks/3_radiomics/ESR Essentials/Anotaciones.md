# ESR Essentials: radiomics—practice recommendations by the European Society of Medical Imaging Informatics

## Radiomics features: extraction parameters, image pre-processing, filters for higher-order features and intuitions for interpretability

### Extraction parameters and image pre-processing

Image pre-processing is performed to standardize image quality, circumvent acquisition susceptibilities, 
and ensure the reproducibility of radiomic features. Intensity normalization is one of
the most important image pre-processing steps.

The open-source Python package PyRadiomics is the
most common framework for radiomics feature extraction. Images and corresponding
segmentations are used to extract radiomics
features for further analysis. Such extraction relies on the
fine-tuning of several parameters to ensure meaningful
feature values. These parameters include intensity discretization,
voxel size resampling, etc., and the reporting
of such parameters is imperative in any radiomics
manuscript to ensure reproducibility.

#### Transformaciones típicas

- Intensity discretization enhances noise reduction and
improves the reproducibility of the extracted features

- Bin width is often favored over bin count due to its independence from the intensity range 
in the selected segment

    As no clear
    guidelines are currently available, it is advised to choose a
    **bin width resulting in a number of bins between 16 and
    128**, which showed good reproducibility for fixed bin
    count with no significant differences in features computed
    within this range

- La resolución de la imagen también es muy importante cuando se trata de un conjunto de datos 
que presenta heterogeneidad en el espaciado de los vóxeles,
ya que la normalización mediante el remuestreo es crucial
para mejorar la reproducibilidad de las características radiómicas.

    It is important to
determine whether images are **isotropic** (los voxels tienen dimensiones iguales en todas las direcciones) 
or **anisotropic** (tiene voxels con dimensiones diferentes en cada dirección)
and choose the resample voxel size accordingly, considering
characteristics like in-plane resolution and slice
thickness.

    Even though isotropic images can be obtained from
    anisotropic ones. a safer approach involves
    **resampling in-plane resolution** and **extracting features in
    2D instead of 3D**, typically used in isotropic images

### Image filters for higher-order radiomics features

Pre-processing filters (e.g.,
Laplacian of Gaussian, Wavelets, Exponential, Logarithm,
Square, Square-Root, etc.) can be applied to images before
feature extraction to enhance quantification of clinically
relevant characteristics and patterns in medical images

While filters expand the number of texture features to
several hundred or even thousands, potentially complicating
the creation of interpretable radiomic models, recent
findings by Demircioglu showed that combining features
extracted from original and filtered images may be beneficial, 
as this approach yielded similar or even superior
predictive performance compared to using only original
image features.

### Intuitions for radiomics feature interpretability

Efforts
must be made to correlate radiomics features with biological
variables to enhance trust in the methodology

For instance, certain features like entropy have direct correlations with biological 
parameters such as tumor heterogeneity

Utilizing color mapping to visualize features in relation to tissue
types of interest further improves interpretability and facilitates clinical adoption

## Radiomics feature univariable analysis

Univariable analyses allow assessing association of each
radiomic feature with the outcome of interest

To assess such associations, methods like 

- Area under the receiver operating characteristic curve (AUC)
- Pearson’s correlation test 
- Chi-square test
- T-test
- Welch’s t-test,
- Mann–Whitney U-test
- variance
- Relief
- and mutual information

can be used.

> Given the challenge posed by a large
number of features and relatively small datasets, these methods
often serve as feature selection tools, aiming to reduce
dimensionality and prevent overfitting

Typically, statistical significance (often
indicated by p < 0.05) guides feature inclusion in multivariate
modeling, accompanied by multiple comparison corrections.

Additionally, unsupervised clustering has been
used to select a single representative feature from each cluster
for use in modeling

> Despite its common usage in radiomics and other
medical sciences, univariable analysis-based feature
selection may be considered inappropriate as it may
produce inaccurate determinations of the contributions of
radiomic features to outcomes

**Shrinkage methods** may be better suited to perform
feature selection as part of the model fitting procedure
and reduce model overfitting

## Best practices for radiomics model development

### Data partitioning

- Correct data partitioning is essential to avoid information
leakage and to avoid biasing the training process by providing
information from test data

- It is imperative that the **data is split on a patient level**.

- **Hyperparameter optimization** can be performed using **validation data**

    Validation data can be obtained from a single partitioning
    (hold-out), but this method is prone to sampling issues.
    Resampling methods are therefore preferable, e.g., bootstrapping
    or k-fold **cross-validation** (CV).

    Results can be averaged across multiple iterations by using different seeds.

- To avoid random sampling issues, a **“temporal” splitting technique is preferred**, which requires acquiring new patients after model development

### Outcome parameter selection

-

### Model comparison

- Depending on the specific target of interest, a
range of model classes can be employed, including but not
limited to regression models, tree-based models, neural
networks, or support vector machines

- single evaluation metric is insufficient for a comprehensive
assessment when comparing different model types.

- several statistical tests are available to evaluate clinically
relevant performance metrics. For classification:

    - McNemar test
    - Cochran’s Q test for confusion matrices
    - DeLong test for area under the curve
    - F-test for variances

    can be employed to compare model

    These tests should be applied to models that have been trained and evaluated using the same process
    to ensure that differences are not due to varying procedures.

- Researchers should weigh the impact of false positives and false negatives independently, considering
the consequences of each type of mistake

### Model fine-tuning and assessment

- Hyperparameter optimization and fine-tuning are crucial for maximizing model performance.

- Optimization must exclusively occur on the validation set

- Once the best model with respective parameters is identified, the test set should be used once to report the final performance metrics.

### Calibration of models/classifier

- Machine learning models output scores between 0 and 1 that are often uncalibrated, meaning they do not accurately reflect the likelihood of aligning with the reference standard.

- Various calibration methods try to mitigate these misalignments:

    - Platt’s scaling (it assumes a logistic relationship)
    - Conformal prediction, which requires a separate hold-out set for calibration and provides mathematical assurances of score-probability alignment

- Calibration curves can be generated to assess the expected calibration error (ECE), indicating the degree to which output scores align with probabilities

- It is essential to examine the histogram of output score bins against underlying probabilities to ensure the distribution is not significantly skewed toward 0 or 1.

### Model explainability

- There’s typically a trade-off between explainability and performance
- Achieving both explainability and performance is an active area of research in Explainable AI
- Some techinques
    - Akaike Information Criteria (AIC)
    - SHapley Additive exPlanations (SHAP)
    - Local Interpretable Model-agnostic Explanations (LIME)


# Conclusiones

- Use CheckList for EvaluAtion of Radiomics research (CLEAR), 
and use quality assessment tools, such as the METhodological RadiomICs Score (METRICS) 
when conducting radiomics research.

- Radiomics feature extraction should be performed with standardized tools that comply with the Image
Biomarker Standardisation Initiative (IBSI) guidelines, such as PyRadiomics.

- Consistent reporting of all preprocessing steps was recommended to improve reproducibility

# Ideas clave por etapa

## 1. Radiomics features

### 1.1. Image pre-processing --> Intensity discretization, bin width, resolución de la imagen (sacar características en 2D)

### 1.2. Image filters --> aplicar filtros genera más características, pero la combinación entre originales y filtros puede ser beneficioso

### 1.3. Radiomics feature interpretability --> correlacionar las características radiómicas con variables biológicas

## 2. Radiomics feature univariable analysis 

- ### Allow assessing association of each radiomic feature with the outcome of interest

## 3. Best practices for radiomics model development

### 3.1. Data partitioning --> división a nivel de paciente, hiperparámetros con datos de validación (bootstrapping o k-fold CV), splitting “temporal” para evitar sesgos aleatorios

### 3.2. Outcome parameter selection --> -

### 3.3. Model comparison --> una única métrica de evaluación es insuficiente, se deben user tests estadísticos (McNemar test, ...), considerar impacto FP y FN de forma independiente

### 3.4. Model fine-tuning and assessment --> optimización exclusivamente en validación, test solo para métricas finales

### 3.5. Calibration of models/classifier --> Platt’s scaling, conformal prediction, calibration curves, histograma de bins de puntuación de salida frente a probabilidades subyacentes 

### 3.6. Model explainability --> Akaike Information Criteria (AIC), SHapley Additive exPlanations (SHAP)
# 1. Image Pre-processing

- Bias field inhomogeneities correction --> Necesita imágenes en 32 bits
- Denoising MRI (a partir de lo obtenido en el punto anterior)

# 2. Radiomics Feature Extraction

Dentro de `Params.yaml` (al parecer habrá que definir un yaml para cada tipo de secuencia, ¿quizá por normalización?):

### Normalization
Since we are working T2w MRI images we will normalize the images by using the parameter `normalize` which we set as `true` 

> **¿Cómo es esto en el resto de secuencias?**

We then scale the standard deviation of the image intensities to be 100 by setting `normalizeScale` to `true`.

The computation of some features (namely Energy, Total Energy and RMS) is affected by the existance of negative value and since we have normalized our images to have mean 0 and standard deviation of 100 we need to ensure that the majority of our normalized image intensities is equal or above 0. We can do this using the `voxelArrayShift`. We selected a value of `300`, which assuming a normal distribution of the image intensities would ensure that 99.7% of them are above 0.

### Memory savings

`preCrop` set to `true` allows us to save some memory during feature computation

### Feature extraction type (2D vs 3D)

We have a dataset where our images are anisotropic (voxel dimensions are not the same for all three axes)

> Because of this, we will perform the extraction of 2D radiomic features instead of 3D. Since our images are T2w Axial images we need to define that we wish to perform a 2D feature extraction by setting `force2D` to `true` and set the in-plane axis to `0` using `force2Ddimension` (Axial - 0, Coronal - 1, Saggital - 2).

### Interpolation/Resampling

To ensure that we are extracting radiomic feature that are comparable among different images we need to standardize the in-plane voxel dimensions. We do this by setting the `interpolator` to `sitkBSpline` and the `resampledPixelSpacing` to a chosen resolution `[0.6, 0.6, 0]`

> **¿Qué resolución indicar?**


## Resegmentation

The resegmentation allows the mask to be redefine to, for example, discard intensities in our mask that are outside $mean\ \pm\ 3\times std$. We can do this by setting `resegmentMode` to `sigma` and `resegmentRange` to `[-3, 3]`.

## Intensity discritization

To ensure that across images the number of bins used for the feature extraction we are between have across patients 30 to 130 performed a first feature extraction to get the `original_firstorder_Range` of our patients and we select the `binWidth` so that it is higher than 30 and lower than 130 accross all patients.

## Filters and Features

We use the `imageType` to define the types of images that from which we which to extract the radiomic features and `featureClass` to select the type of features to be extracted.

# Prostate Cancer Classification with AI and mpMRI

This repository contains the code and artefacts used in an ongoing doctoral research line on clinically significant prostate cancer (`csPCa`) classification from multi-parametric MRI (mpMRI). The project compares classical radiomics pipelines against deep learning models and places special emphasis on reproducibility, grouped evaluation by patient, and interpretability.

## Clinical Target

The binary target is `csPCa`, defined here as ISUP grade group `>= 2`.

The current pipeline works with three axial MRI sequences:

- T2-weighted (`T2W`)
- Apparent diffusion coefficient (`ADC`)
- High b-value diffusion-weighted imaging (`DWI` / `HBV`)

## Repository Structure

```text
├── artifacts/
│   ├── data.csv                     # Cohort table with image paths, labels, and metadata
│   └── radiomics/                  # Extracted modality-specific radiomics CSV files
├── data_analysis/                  # Exploratory notebooks and descriptive analyses
├── data_structuring/               # Notebook used to assemble the cohort CSV
├── results/                        # Model outputs, comparisons, hold-out evaluation, plots
├── train/
│   ├── common/                     # Shared utilities for reproducibility and radiomics helpers
│   ├── compare_approaches/         # Radiomics vs deep learning comparison scripts
│   ├── deep_learning/
│   └── radiomics/
├── z_figures/
└── z_report/
```

## End-to-End Workflow

### 1. Assemble the cohort table

The project starts from `artifacts/data.csv`, which contains:

- patient and study identifiers
- binary label (`case_csPCa`)
- paths to the three MRI sequences
- whole-gland segmentation path
- additional clinical and image metadata

This table is created from the notebooks in `data_structuring/`.

### 2. Extract modality-specific radiomics

Script: `train/radiomics/1_extract_radiomics/extract_radiomics.py`

For each case and for each modality (`T2W`, `ADC`, `DWI`), the script:

1. Loads the MRI volume.
2. Applies preprocessing:
   - float32 conversion
   - N4 bias-field correction
   - curvature anisotropic diffusion denoising
3. Uses the whole-gland mask for the gland-focused analysis.
4. Builds an all-ones mask for the full-volume analysis.
5. Runs PyRadiomics with a modality-specific YAML configuration.

This produces six CSV files in `artifacts/radiomics/`:

- `features_t2_gland.csv`
- `features_adc_gland.csv`
- `features_dwi_gland.csv`
- `features_t2_full.csv`
- `features_adc_full.csv`
- `features_dwi_full.csv`

### 3. Build the concatenated modeling table

Script: `train/radiomics/2_modeling/0_build_concatenated_feature_table.py`

The six modality-specific CSV files are merged into a single modeling table for each spatial setting:

- `features_all_gland.csv`
- `features_all_full.csv`

Important implementation details:

- rows are matched using `patient_id`, `study_id`, and `label`
- feature names are prefixed by modality (`t2_`, `adc_`, `dwi_`)
- shape features are retained from only one reference modality to avoid redundant duplicates
- a unique `sample_id = patient_id + "_" + study_id` is created

### 4. Run repeated grouped cross-validation

Script: `train/radiomics/2_modeling/1_train_and_evaluate.py`

This is the main radiomics benchmarking script. It evaluates six classical classifiers:

- SVM
- Logistic Regression
- Random Forest
- Naive Bayes
- KNN
- Gradient Boosting

The evaluation protocol is:

- grouped by `patient_id`, so studies from the same patient do not leak across train and validation
- stratified at the group level
- repeated `5-fold x 10 repeats` by default, which yields `50` validation folds per classifier

The script first precomputes the grouped split plan once and then reuses that same fold plan across all classifiers so that the comparison is fair.

### 5. Leakage-safe feature selection inside each fold

When `--feature_strategy most_discriminant` is used, feature selection is performed inside each training fold only. The validation fold is never used to choose features.

This is the most important part of the radiomics pipeline:

1. Start from the numeric radiomics matrix only.
   - metadata columns such as `patient_id`, `study_id`, `label`, `sample_id`, and PyRadiomics `diagnostics_*` columns are removed
2. Work only with the training partition of the current fold.
3. Score each feature independently on that training data:
   - invalid or near-constant features are skipped
   - a normality check is attempted
   - if the feature looks Gaussian, a two-sample `t-test` is used
   - otherwise, a `Mann-Whitney U` test is used
   - a univariate ROC AUC is also computed for ranking
   - the best single-feature threshold is estimated with the Youden index
4. Apply false discovery rate control.
   - Benjamini-Hochberg correction is used
   - features with `q <= fdr_alpha` form the preferred candidate pool
   - if none survive FDR, the script falls back to the valid ranked features
5. Infer how many features can be kept in that fold.
   - this is not fixed blindly
   - the cap depends on training sample size and minority-class size
   - the goal is to keep the subset conservative relative to the available data
6. Prune redundancy by correlation.
   - candidate features are sorted by univariate relevance
   - then a greedy pruning step removes features whose absolute Pearson correlation with a previously kept feature is above the threshold
7. Keep the top pruned features up to the inferred cap.
8. Train the classifier on the selected subset and evaluate on the untouched validation fold.

Because this process runs fold by fold, the selected feature subset can change from one fold to another. That is expected and is actually the correct leakage-safe behaviour.

## What Happens After the `5 x 10` Training?

The repeated cross-validation stage does not stop at reporting 50 numbers per model. The script performs several post-processing steps.

### Fold-level outputs

For every classifier and every fold, the pipeline stores:

- train and validation metrics
- the selected feature subset used in that fold
- validation labels, predictions, and probabilities

### Flat out-of-fold predictions

The fold predictions are expanded into a one-row-per-case table:

- classifier
- fold and repeat
- sample, patient, and study identifiers
- true label
- predicted label
- probability of class 1
- selected features for that fold

### Aggregated out-of-fold predictions

Since the cross-validation is repeated 10 times, the same case appears in validation more than once. The script therefore aggregates repeated out-of-fold predictions by averaging the predicted probability for each case and classifier across all its validation appearances.

After that, it:

- applies the classification threshold, default `0.5`
- generates one aggregated prediction per case and classifier
- computes patient-level performance summaries

### Bootstrap confidence intervals

Using the aggregated out-of-fold predictions, the script performs stratified bootstrap resampling at the patient level to estimate confidence intervals for:

- AUC
- accuracy
- balanced accuracy
- F1
- MCC
- kappa
- sensitivity
- specificity
- PPV
- NPV

It also exports ROC curves with confidence bands.

### Statistical comparison between classifiers

If `--calculate_differences` is enabled, the script runs `train/radiomics/2_modeling/2_model_differences.py`, which:

- compares classifiers using the fold-wise metric distributions
- applies a Friedman global test
- if significant, runs pairwise Wilcoxon signed-rank tests with Holm correction

This produces the model-ranking comparison used to justify which classifier should move forward to the final optimization stage.

## Final Hold-Out Optimization of the Best Classifier

Script: `train/radiomics/2_modeling/3_retrain_best_model_and_evaluate.py`

If `--fine_tune_best_model` is enabled, the best classifier according to median validation AUC is retrained in a separate final stage.

The logic is:

1. Create a grouped `80/20` train/test split with `GroupShuffleSplit`.
2. Run feature selection again using only the training split.
3. Restrict both train and test to that training-derived feature subset.
4. Optimize the selected classifier with `BayesSearchCV` using grouped cross-validation inside the training split.
5. Save the best estimator.
6. Evaluate the uncalibrated model on the hold-out test split.
7. Estimate test confidence intervals by patient-level bootstrap.
8. Calibrate predicted probabilities with Platt scaling (`CalibratedClassifierCV`, sigmoid).
9. Re-evaluate the calibrated model.
10. Sweep decision thresholds and report the threshold with the best F1.
11. Run SHAP and LIME analyses on both the training split and the hold-out test split.

This final stage produces the model intended for deeper interpretation and a more realistic final evaluation than the repeated cross-validation benchmark alone.

## Reproducibility Notes

Current reliability-oriented implementation choices include:

- grouped splitting by patient
- fold-wise feature selection to avoid leakage
- shared fold plans across classifiers for fair comparison
- exported selected features per fold
- aggregated out-of-fold predictions at the case level
- bootstrap confidence intervals at the patient level
- project-root-based path resolution instead of fragile relative paths

One methodological caution is worth noting: in the current final hold-out script, the threshold sweep is performed on the hold-out test set itself. That is useful for exploratory analysis, but if the threshold is meant to be locked for a final unbiased evaluation, it should be chosen on a separate validation layer inside training instead of on the test split.

## Typical Commands

### Build the concatenated radiomics table

```bash
python train/radiomics/2_modeling/0_build_concatenated_feature_table.py \
  --radiomics_root artifacts/radiomics \
  --mode gland \
  --keep_shape_from t2 \
  --output artifacts/radiomics/concatenated_data/features_all_gland.csv
```

### Run the main radiomics benchmark

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

### Run the final hold-out optimization directly

```bash
python train/radiomics/2_modeling/3_retrain_best_model_and_evaluate.py \
  --csv artifacts/radiomics/concatenated_data/features_all_gland.csv \
  --model LogisticRegression \
  --feature_strategy most_discriminant \
  --bootstrap_iterations 1000 \
  --ci_level 0.95
```

## Related Modules

- `train/radiomics/README.md`: radiomics-specific methodology in more detail
- `train/deep_learning/README.md`: deep learning branch
- `train/compare_approaches/`: direct radiomics vs deep learning comparison scripts

## Reference

[1] A. Saha, J. S. Bosma, J. J. Twilt, B. van Ginneken, A. Bjartell, A. R. Padhani, D. Bonekamp, G. Villeirs, G. Salomon, G. Giannarini, J. Kalpathy-Cramer, J. Barentsz, K. H. Maier-Hein, M. Rusu, O. Rouvière, R. van den Bergh, V. Panebianco, V. Kasivisvanathan, N. A. Obuchowski, D. Yakar, M. Elschot, J. Veltman, J. J. Fütterer, M. de Rooij, H. Huisman, and the PI-CAI consortium. “Artificial Intelligence and Radiologists in Prostate Cancer Detection on MRI (PI-CAI): An International, Paired, Non-Inferiority, Confirmatory Study”. *The Lancet Oncology* 2024; 25(7): 879-887.

# Prostate Cancer Classification with AI and mpMRI

This repository contains the code and artefacts for a Bachelor's thesis that compares radiomics and deep learning for clinically significant prostate cancer classification using multi-parametric MRI (mpMRI). The study focuses on both predictive performance and interpretability.

## Overview

The project uses the public [PI-CAI](https://pi-cai.grand-challenge.org/) dataset and works with three axial MRI sequences:

- T2-weighted (T2W)
- Diffusion-weighted imaging (DWI / high b-value)
- Apparent diffusion coefficient (ADC)

The target is a binary label indicating clinically significant prostate cancer (`csPCa`), defined as ISUP grade group `>= 2`.

## Repository Structure

```text
├── artifacts/
│   ├── deep_learning/   # Training logs, checkpoints, split files, and intermediate outputs
│   └── radiomics/       # Extracted radiomics features and structured inputs
├── data_analysis/       # Exploratory notebooks and descriptive analysis
├── data_structuring/    # Notebook used to assemble the central dataset CSV
├── results/             # Final analysis outputs, figures, and comparison reports
├── train/
│   ├── common/          # Shared utilities for reproducibility and path handling
│   ├── compare_approaches/
│   ├── deep_learning/
│   └── radiomics/
├── z_figures/
└── z_report/
```

## Methodology

### Radiomics

Radiomics features are extracted from the MRI volumes and used to train classical machine learning models such as logistic regression, SVM, random forest, and gradient boosting.

### Deep Learning

Several 3D neural network backbones are trained directly on the imaging volumes using grouped cross-validation by patient.

### Statistical Comparison

The repository includes scripts to compare classifiers within each family and to compare the best radiomics and deep learning approaches.

### Interpretability

Interpretability scripts generate Grad-CAM, occlusion sensitivity, SHAP, and LIME outputs depending on the selected model family.

## Recent Reliability Improvements

The current codebase now includes:

- Leakage-safe fold-wise feature selection for radiomics experiments
- Persisted grouped validation splits for deep learning experiments
- Deterministic seeding for Python, NumPy, and PyTorch
- Safer checkpoint saving for the best deep learning epoch
- English logs, comments, summaries, and figure labels in the updated scripts
- Project-root-based path resolution instead of fragile relative path chaining

## Installation

1. Clone the repository:

```bash
git clone https://github.com/jose-valero-sanchis/prostate_cancer_TFG.git
cd prostate_cancer_TFG
```

2. Install the project dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Build the dataset CSV

The central dataset table is created from the notebook in `data_structuring/`.

### 2. Run radiomics experiments

Example:

```bash
python train/radiomics/2_modeling/1_train_and_evaluate.py --csv features_all_gland.csv --feature_strategy most_discriminant --n_splits 5 --n_repeats 10 --calculate_differences
```

### 3. Run deep learning experiments

Example:

```bash
python train/deep_learning/1_modeling/train.py --config_key config1 --mode gland --epochs 50 --n_splits 5 --seed 42
```

### 4. Generate deep learning validation predictions

Example:

```bash
python train/deep_learning/2_analyse_results/predict_&_analyse_probs/1_predict.py --mode gland --n_splits 5 --seed 42
```

## Notes

- The repository still contains historical notebooks and archived result folders from earlier experiments.
- Large generated artefacts should remain outside Git whenever possible.
- Some legacy scripts are still being migrated to the updated English and reproducible workflow.

## Reference

[1] A. Saha, J. S. Bosma, J. J. Twilt, B. van Ginneken, A. Bjartell, A. R. Padhani, D. Bonekamp, G. Villeirs, G. Salomon, G. Giannarini, J. Kalpathy-Cramer, J. Barentsz, K. H. Maier-Hein, M. Rusu, O. Rouvière, R. van den Bergh, V. Panebianco, V. Kasivisvanathan, N. A. Obuchowski, D. Yakar, M. Elschot, J. Veltman, J. J. Fütterer, M. de Rooij, H. Huisman, and the PI-CAI consortium. “Artificial Intelligence and Radiologists in Prostate Cancer Detection on MRI (PI-CAI): An International, Paired, Non-Inferiority, Confirmatory Study”. *The Lancet Oncology* 2024; 25(7): 879-887.

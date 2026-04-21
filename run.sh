#!/bin/bash
#SBATCH --job-name=RadiomicaPICAI

#SBATCH --partition=gpuceib

#SBATCH --cpus-per-task 7

#SBATCH --mem 150G

#SBATCH --output=./RadiomicaPICAI.out

export PYTHONUNBUFFERED=1

module load Python/3.11.5-GCCcore-11.2.0 
source /projects/ceib/python_enviroments/radiomics_venv/bin/activate



python train/radiomics/2_modeling/0_build_concatenated_feature_table.py \
  --mode gland \
  --keep_shape_from t2 \
  --output artifacts/radiomics/concatenated_data/features_all_gland.csv

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
  --calculate_differences \
  --fine_tune_best_model


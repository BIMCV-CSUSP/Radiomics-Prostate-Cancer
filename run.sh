#!/bin/bash
#SBATCH --job-name=RadiomicaPICAI

#SBATCH --partition=long

#SBATCH --cpus-per-task 32

#SBATCH --mem 150G

#SBATCH --output=./RadiomicaPICAI.out


module load Python/3.11.5-GCCcore-11.2.0 
source /projects/ceib/python_enviroments/radiomics_venv/bin/activate

export PYTHONUNBUFFERED=1
export MPLBACKEND=Agg

python train/radiomics/2_modeling/1_train_and_evaluate.py \
  --csv features_all_gland.csv \
  --data_pre artifacts/radiomics \
  --results_base results/radiomics \
  --experiment_name more_features_v1_rerun_no_tune \
  --feature_strategy most_discriminant \
  --n_splits 5 \
  --n_repeats 10 \
  --bootstrap_iterations 1000 \
  --ci_level 0.95 \
  --classification_threshold 0.5 \
  --min_features 30 \
  --max_features_cap 100 \
  --samples_per_feature 15 \
  --minority_samples_per_feature 5 \
  --fdr_alpha 0.05 \
  --correlation_threshold 0.95 \
  --selection_n_jobs 32 \
  --search_n_jobs 32 \
  --search_iterations 50 \
  --calculate_differences \
  --fine_tune_best_model


# python -u train/radiomics/2_modeling/1_train_and_evaluate.py \
#   --csv features_all_gland.csv \
#   --data_pre artifacts/radiomics \
#   --results_base results/radiomics \
#   --feature_strategy most_discriminant \
#   --bootstrap_iterations 1000 \
#   --ci_level 0.95 \
#   --classification_threshold 0.5 \
#   --min_features 10 \
#   --max_features_cap 60 \
#   --samples_per_feature 25 \
#   --minority_samples_per_feature 8 \
#   --fdr_alpha 0.05 \
#   --correlation_threshold 0.90 \
#   --selection_n_jobs 7 \
#   --search_n_jobs 7 \
#   --search_iterations 50 \
#   --calculate_differences \
#   --fine_tune_best_model \
#   --postprocess_only
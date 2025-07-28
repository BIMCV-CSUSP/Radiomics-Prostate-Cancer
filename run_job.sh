#!/bin/bash
#SBATCH --job-name=radiomics
 
#SBATCH --partition=gpuceib
 
#SBATCH --cpus-per-task 15
 
#SBATCH --mem 120G
 
#SBATCH --output=extract_radiomics.out
 
#SBATCH --gres=gpu:0
 
source /projects/ceib/python_enviroments/radiomica/bin/activate
 
export LD_LIBRARY_PATH="/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH"
export PATH="/home/jaalzate/.local/bin:/usr/local/cuda-11.7/bin:$PATH"
# export PYTHONPATH="/home/jbernal/.local/lib/python3.9/site-packages:/home/jbernal/code_utils:$PYTHONPATH"
 
# jupyter lab --ip  '0.0.0.0' --port 8889

# python data_analysis/z_get_binWidth/get_binwidth.py
python train/radiomics/1_extract_radiomics/extract_radiomics.py
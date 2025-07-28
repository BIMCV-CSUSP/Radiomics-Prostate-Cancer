#!/bin/bash
#SBATCH --job-name=RadiomicaPICAI

#SBATCH --partition=gpuceib

#SBATCH --cpus-per-task 15

#SBATCH --mem 200G

#SBATCH --output=./RadiomicaPICAI.out

source /projects/ceib/python_enviroments/radiomica/bin/activate
module load GCC
module load CUDA

export PATH="/home/jaalzate/.local/bin:$PATH"
export PATH="/usr/local/cuda-11.7/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH"
export PYTHONPATH="/projects/ceib/python_enviroments/radiomica/lib/python3.10/site-packages:$PYTHONPATH"

python /home/jaalzate/Radiomics-Prostate-Cancer/train/radiomics/1_extract_radiomics/extract_radiomics.py
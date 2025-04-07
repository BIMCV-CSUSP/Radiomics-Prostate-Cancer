#!/bin/bash
#SBATCH --job-name=DeepLearningPICAI

#SBATCH --partition=gpuceib

#SBATCH --cpus-per-task 15

#SBATCH --mem 100G

#SBATCH --output=./DeepLearningPICAI.out

#SBATCH --gres=gpu:1

source /projects/ceib/python_enviroments/monai/bin/activate
module load GCC
module load CUDA

export PATH="/home/jvalero/.local/bin:$PATH"
export PATH="/usr/local/cuda-11.7/bin:$PATH"
export LD_LIBRARY_PATH="/usr/local/cuda-11.7/lib64:$LD_LIBRARY_PATH"
export PYTHONPATH="/projects/ceib/python_enviroments/monai/lib/python3.10/site-packages:$PYTHONPATH"

cd /home/jvalero/...

./run_all.sh
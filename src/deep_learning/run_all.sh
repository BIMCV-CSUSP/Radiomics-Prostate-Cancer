#!/bin/bash

configs=("config1" "config2" "config3" "config4" "config5" "config6")

for config in "${configs[@]}"
do
    echo "Ejecutando entrenamiento con la configuración: ${config}"
    python train.py --config_key "${config}" --epochs 50 --n_splits 5
done
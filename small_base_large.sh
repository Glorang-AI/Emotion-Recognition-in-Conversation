#!/bin/bash
MODELS=("CASE" "CSE")
WANDBNAMEsmall=("CASE-small" "CCSE-small")
WANDBNAMEbase=("CASE-base" "CCSE-base")
WANDBNAMElarge=("CASE-large" "CCSE-large")

echo "MMM"
python3 main.py --model "MMM" --epochs 200 --wandb_project "glorang" --wandb_entity "glorang" --wandb_name "MMM-small" --device "cuda:1" --val_ratio 0 --opt "mean" --lr 2e-5 --mm_type 'add' --size 'small'

for (( i=0; i<2; i++ ))
do
    echo ${MODELS[$i]}
    python3 main.py --model ${MODELS[$i]} --epochs 200 --wandb_project "glorang" --wandb_entity "glorang" --wandb_name ${WANDBNAME[$i]} --device "cuda:1" --val_ratio 0 --opt "mean" --lr 2e-5 --mm_type 'add' --size 'small'
    python3 main.py --model ${MODELS[$i]} --epochs 200 --wandb_project "glorang" --wandb_entity "glorang" --wandb_name ${WANDBNAME[$i]} --device "cuda:1" --val_ratio 0 --opt "mean" --lr 2e-5 --mm_type 'add' --size 'base'
    python3 main.py --model ${MODELS[$i]} --epochs 200 --wandb_project "glorang" --wandb_entity "glorang" --wandb_name ${WANDBNAME[$i]} --device "cuda:1" --val_ratio 0 --opt "mean" --lr 2e-5 --mm_type 'add' --size 'large'
done
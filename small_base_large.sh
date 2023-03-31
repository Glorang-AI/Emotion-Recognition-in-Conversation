#!/bin/bash
MODELS=("CASE" "CSE")
WANDBNAMEsmall=("CASE-small" "CCSE-small")
WANDBNAMEbase=("CASE-base" "CCSE-base")
WANDBNAMElarge=("CASE-large" "CCSE-large")


for (( i=1; i<2; i++ ))
do
    echo ${MODELS[$i]}
    python3 main.py --model ${MODELS[$i]} --epochs 200 --wandb_project "glorang" --wandb_entity "glorang" --wandb_name ${WANDBNAMEsmall[$i]} --device "cuda:0" --val_ratio 0 --opt "mean" --lr 2e-5 --mm_type 'add' --size 'small'
    python3 main.py --model ${MODELS[$i]} --epochs 200 --wandb_project "glorang" --wandb_entity "glorang" --wandb_name ${WANDBNAMEbase[$i]} --device "cuda:0" --val_ratio 0 --opt "mean" --lr 2e-5 --mm_type 'add' --size 'base'
    python3 main.py --model ${MODELS[$i]} --epochs 200 --wandb_project "glorang" --wandb_entity "glorang" --wandb_name ${WANDBNAMElarge[$i]} --device "cuda:2" --val_ratio 0 --opt "mean" --lr 2e-5 --mm_type 'add' --size 'large'
done
#!/bin/bash
MODELS=("CASE" "CSE")
WANDBNAME=("CASE-contrastive" "CCSE-contrastive")

for (( i=0; i<2; i++ ))
do
    echo ${MODELS[$i]}
    python3 main.py --model ${MODELS[$i]} --epochs 200 --wandb_project "glorang" --wandb_entity "glorang" --wandb_name ${WANDBNAME[$i]} --device "cuda:0" --val_ratio 0 --opt "mean" --lr 2e-5 --mm_type 'add' --size 'small' --contrastive True
done
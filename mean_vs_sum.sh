#!/bin/bash
MODELS=("CASE" "CSE")
WANDBNAMEmean=("CASE-mean" "CCSE-mean")
WANDBNAMEsum=("CASE-sum" "CCSE-sum")
for (( i=1; i<2; i++ ))
do
    echo ${MODELS[$i]}
    python3 main.py --model ${MODELS[$i]} --epochs 200 --wandb_project "glorang" --wandb_entity "glorang" --wandb_name ${WANDBNAMEmean[$i]} --device "cuda:1" --val_ratio 0 --opt "mean" --lr 2e-5
    python3 main.py --model ${MODELS[$i]} --epochs 200 --wandb_project "glorang" --wandb_entity "glorang" --wandb_name ${WANDBNAMEsum[$i]} --device "cuda:3" --val_ratio 0 --opt "sum" --lr 2e-5
done

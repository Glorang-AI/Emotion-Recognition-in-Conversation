#!/bin/bash
MODELS=("CCASE" "CASE" "CCE")
WANDBNAME=("CCASE_PET" "CASE_PET" "CCE_PET")

for (( i=0; i<3; i++ ))
do
    echo ${MODELS[$i]}
    python3 main.py --model ${MODELS[$i]} --epochs 50 --wandb_project "ERC" --wandb_entity "21" --wandb_name ${WANDBNAME[$i]} --device "cuda:3" --pet True
done

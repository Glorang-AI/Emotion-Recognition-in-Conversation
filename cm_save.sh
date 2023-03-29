#!/bin/bash

MODELS=("Concat" "MMM"  "CASE" "CCE")
WANDBNAME=("Concat_new_data_all" "MMM_new_data_all" "CASE_new_data_all" "CCE_new_data_all")
LOGFOLDER=("CONCAT_log_all" "MMM_log_all"  "CASE_log_all" "CCE_log_all")
for (( i=0; i<4; i++ ))
do
    echo ${MODELS[$i]}
    python3 main.py --model ${MODELS[$i]} --epochs 50 --wandb_project "ERC" --wandb_entity "21" --wandb_name ${WANDBNAME[$i]} --device "cuda:2" --val_ratio 0
    cp -r wandb/latest-run/files/media data/${LOGFOLDER[$i]}
done
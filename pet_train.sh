#!/bin/bash
WANDBNAME=("CASE_concat_all_2" "CASE_org_all_2")
CONCAT=(True False)


python3 main.py --model "CASE" --epochs 200 --wandb_project "ERC" --wandb_entity "21" --wandb_name "CASE_concat_all_2" --device "cuda:3" --case_concat True --val_ratio 0
python3 main.py --model "CASE" --epochs 200 --wandb_project "ERC" --wandb_entity "21" --wandb_name "CASE_all" --device "cuda:2" --case_concat False --val_ratio 0


SEED=(0 42 2023)
WANDB_GROUP=("0" "42" "2023")
for (( i=0; i<3; i++ ))
do
    echo ${SEED[$i]}
    python3 main.py --model "CSE" --seed ${SEED[$i]} --epochs 200 --wandb_project "glorang" --wandb_entity "glorang" --wandb_group ${WANDB_GROUP[$i]} --wandb_name "CCSE-small" --device "cuda:1" --val_ratio 0 --lr 2e-5 --size "small"
    python3 main.py --model "CSE" --seed ${SEED[$i]} --epochs 200 --wandb_project "glorang" --wandb_entity "glorang" --wandb_group ${WANDB_GROUP[$i]} --wandb_name "CCSE-small-concat" --device "cuda:1" --val_ratio 0 --lr 2e-5 --size "small" --mm_type "concat"

    python3 main.py --model "CASE" --seed ${SEED[$i]} --epochs 200 --wandb_project "glorang" --wandb_entity "glorang" --wandb_group ${WANDB_GROUP[$i]} --wandb_name "CASE-small" --device "cuda:1" --val_ratio 0 --lr 2e-5 --size "small"
    python3 main.py --model "CASE" --seed ${SEED[$i]} --epochs 200 --wandb_project "glorang" --wandb_entity "glorang" --wandb_group ${WANDB_GROUP[$i]} --wandb_name "CASE-small-concat" --device "cuda:1" --val_ratio 0 --lr 2e-5 --size "small" --mm_type "concat"

    python3 main.py --model "MMM" --seed ${SEED[$i]} --epochs 200 --wandb_project "glorang" --wandb_entity "glorang" --wandb_group ${WANDB_GROUP[$i]} --wandb_name "MMM-small" --device "cuda:1" --val_ratio 0 --lr 2e-5 --size "small"

    python3 main.py --model "Concat" --seed ${SEED[$i]} --epochs 200 --wandb_project "glorang" --wandb_entity "glorang" --wandb_group ${WANDB_GROUP[$i]} --wandb_name "Concat-small" --device "cuda:1" --val_ratio 0 --lr 2e-5 --size "small"
done



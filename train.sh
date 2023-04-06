SEED=(0 42 2023)
WANDB_GROUP=("0" "42" "2023")
for (( i=0; i<3; i++ ))
do
    python3 main.py --model "compressing" --seed ${SEED[$i]} --epochs 150 --wandb_project "glorang" --wandb_entity "glorang" --wandb_group ${WANDB_GROUP[$i]} --wandb_name "CASE (compressing)" --device "cuda:0" --val_ratio 0
    python3 main.py --model "compressing" --seed ${SEED[$i]} --epochs 150 --wandb_project "glorang" --wandb_entity "glorang" --wandb_group ${WANDB_GROUP[$i]} --wandb_name "CASE (compressing) concat" --device "cuda:0" --val_ratio 0 --mm_type "concat"

    python3 main.py --model "attention" --seed ${SEED[$i]} --epochs 150 --wandb_project "glorang" --wandb_entity "glorang" --wandb_group ${WANDB_GROUP[$i]} --wandb_name "CASE (attention)" --device "cuda:0" --val_ratio 0
    python3 main.py --model "attention" --seed ${SEED[$i]} --epochs 150 --wandb_project "glorang" --wandb_entity "glorang" --wandb_group ${WANDB_GROUP[$i]} --wandb_name "CASE (attention) concat" --device "cuda:0" --val_ratio 0 --mm_type "concat"

    python3 main.py --model "MMM" --seed ${SEED[$i]} --epochs 150 --wandb_project "glorang" --wandb_entity "glorang" --wandb_group ${WANDB_GROUP[$i]} --wandb_name "MMM" --device "cuda:0" --val_ratio 0

    python3 main.py --model "Concat" --seed ${SEED[$i]} --epochs 150 --wandb_project "glorang" --wandb_entity "glorang" --wandb_group ${WANDB_GROUP[$i]} --wandb_name "Concat" --device "cuda:0" --val_ratio 0

    python3 main.py --model "text_only" --seed ${SEED[$i]} --epochs 150 --wandb_project "glorang" --wandb_entity "glorang" --wandb_group ${WANDB_GROUP[$i]} --wandb_name "text_only" --device "cuda:0" --val_ratio 0
    python3 main.py --model "speech_only" --seed ${SEED[$i]} --epochs 150 --wandb_project "glorang" --wandb_entity "glorang" --wandb_group ${WANDB_GROUP[$i]} --wandb_name "speech_only" --device "cuda:0" --val_ratio 0 
done



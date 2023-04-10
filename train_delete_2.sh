SEED=(42)
WANDB_GROUP=("42")
for (( i=0; i<1; i++ ))
do
    # python3 main.py --model "compressing" --seed ${SEED[$i]} --epochs 150 --wandb_project "glorang" --wandb_entity "glorang" --wandb_group ${WANDB_GROUP[$i]} --wandb_name "Final CASE (compressing)" --device "cuda:1" --val_ratio 0
    python3 main.py --model "compressing" --seed ${SEED[$i]} --epochs 150 --wandb_project "glorang" --wandb_entity "glorang" --wandb_group ${WANDB_GROUP[$i]} --wandb_name "Final CASE (compressing) concat" --device "cuda:1" --val_ratio 0 --mm_type "concat"

    python3 main.py --model "attention" --seed ${SEED[$i]} --epochs 150 --wandb_project "glorang" --wandb_entity "glorang" --wandb_group ${WANDB_GROUP[$i]} --wandb_name "Final CASE (attention)" --device "cuda:1" --val_ratio 0
    python3 main.py --model "attention" --seed ${SEED[$i]} --epochs 150 --wandb_project "glorang" --wandb_entity "glorang" --wandb_group ${WANDB_GROUP[$i]} --wandb_name "Final CASE (attention) concat" --device "cuda:1" --val_ratio 0 --mm_type "concat"

    # python3 main.py --model "MMM" --seed ${SEED[$i]} --epochs 150 --wandb_project "glorang" --wandb_entity "glorang" --wandb_group ${WANDB_GROUP[$i]} --wandb_name "Final MMM" --device "cuda:0" --val_ratio 0

    # python3 main.py --model "Concat" --seed ${SEED[$i]} --epochs 150 --wandb_project "glorang" --wandb_entity "glorang" --wandb_group ${WANDB_GROUP[$i]} --wandb_name "Final Concat" --device "cuda:0" --val_ratio 0

    # python3 main.py --model "text_only" --seed ${SEED[$i]} --epochs 150 --wandb_project "glorang" --wandb_entity "glorang" --wandb_group ${WANDB_GROUP[$i]} --wandb_name "Final text_only" --device "cuda:0" --val_ratio 0
    # python3 main.py --model "speech_only" --seed ${SEED[$i]} --epochs 150 --wandb_project "glorang" --wandb_entity "glorang" --wandb_group ${WANDB_GROUP[$i]} --wandb_name "Final speech_only" --device "cuda:0" --val_ratio 0 
done



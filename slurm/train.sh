#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --job-name='BirdFSD-YOLOv5-train'
#SBATCH --partition='gpu'
#SBATCH --gres=gpu
#SBATCH --constraint=gpu_32gb&gpu_v100
#SBATCH --mem=32gb
#SBATCH --mail-type=ALL
#SBATCH --error=%J.err
#SBATCH --output=%J.out

#-------------------------------------
export WANDB_CACHE_DIR="$WORK/.cache"
export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
#export GITHUB_ACTIONS_RUN_ID= # <<<<<<<<<<<<<<<<<<<< @required
#-------------------------------------
module unload python
module load anaconda
conda activate torch-gpu
module load cuda/10.2
#-------------------------------------
nvidia-smi
#-------------------------------------
#rm -rf dataset-YOLO dataset-YOLO*.tar dataset_config.yml best.pt
#rm -rf yolov5/runs wandb
#-------------------------------------
#gh run download $GITHUB_ACTIONS_RUN_ID
#tar -xf artifacts/dataset-YOLO-*.tar
#mv dataset-YOLO/dataset_config.yml .
#python model_utils/relative_to_abs.py
#-------------------------------------
#python model_utils/download_weights.py --model-version latest
#-------------------------------------
IMAGE_SIZE=640
BATCH_SIZE=64
EPOCHS=300
PRETRAINED_WEIGHTS='BirdFSD-YOLOv5-v1.0.0-alpha.4.1-best_weights.pt'
#-------------------------------------
python yolov5/train.py \
    --img-size $IMAGE_SIZE \
    --batch $BATCH_SIZE \
    --epochs $EPOCHS \
    --data 'dataset_config.yml' \
    --weights "$PRETRAINED_WEIGHTS" \
    --device 0
#-------------------------------------
# python -m torch.distributed.launch \
#     --nproc_per_node 2 \
#     yolov5/train.py \
#     --img-size $IMAGE_SIZE \
#     --batch $BATCH_SIZE \
#     --epochs $EPOCHS \
#     --data 'dataset_config.yml' \
#     --weights "$PRETRAINED_WEIGHTS" \
#     --device 0,1
#-------------------------------------
WANDB_PATH_PARENT="biodiv/train"
DATASET_NAME=$(ls dataset-YOLO-*)

python model_utils/update_run_cfg.py \
    --run-path "$WANDB_PATH_PARENT/$WANDB_RUN_ID" \
    --dataset-name "$DATASET_NAME"

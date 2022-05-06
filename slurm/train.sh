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
export WANDB_RUN_ID=$(python -c "import secrets; print(secrets.token_hex(1000)[-8:])")
export DATASET_NAME= # <<<<<<<<<<<<<<<<<<<< @required
export GITHUB_ACTIONS_RUN_ID= # <<<<<<<<<<<<<<<<<<<< @required
#-------------------------------------
module unload python
module load anaconda
conda activate yolov5
module load cuda/10.2
#-------------------------------------
nvidia-smi
#-------------------------------------
rm -rf dataset-YOLO dataset-YOLO*.tar dataset_config.yml best.pt
rm -rf yolov5/runs wandb
#-------------------------------------
gh run download $GITHUB_ACTIONS_RUN_ID
tar -xf artifacts/dataset-YOLO-*.tar
mv artifacts/dataset-YOLO .
mv dataset-YOLO/dataset_config.yml .
python model_utils/relative_to_abs.py
#-------------------------------------
python model_utils/download_weights.py --model-version latest
#-------------------------------------
BATCH_SIZE=-1
EPOCHS=300
PRETRAINED_WEIGHTS='best.pt'
IMAGE_SIZE=768
#-------------------------------------
python yolov5/train.py \
   --img-size $IMAGE_SIZE \
   --batch $BATCH_SIZE \
   --epochs $EPOCHS \
   --data 'dataset_config.yml' \
   --weights $PRETRAINED_WEIGHTS \
   --device 0
#-------------------------------------
python model_utils/update_run_cfg.py \
   --run-path "biodiv/train/$WANDB_RUN_ID" \
   --dataset-name

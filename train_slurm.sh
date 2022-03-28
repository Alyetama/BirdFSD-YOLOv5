#!/bin/bash
#SBATCH --time=02:00:00
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
#-------------------------------------
module unload python
module load anaconda
conda activate yolov5
module load cuda/10.2
#-------------------------------------
nvidia-smi
#-------------------------------------
rm -rf "dataset-YOLO" "dataset-YOLO.tar"
python json2yolov5.py
python utils/relative_to_abs.py
#-------------------------------------
rm "best.pt"
wget "https://d.aibird.me/best.pt"
#-------------------------------------
BATCH_SIZE=16
EPOCHS=30
PRETRAINED_WEIGHTS="best.pt"
#-------------------------------------
python yolov5/train.py --img-size 768 --batch $BATCH_SIZE --epochs $EPOCHS \
   --data "dataset_config.yml" --weights $PRETRAINED_WEIGHTS --device 0
#-------------------------------------

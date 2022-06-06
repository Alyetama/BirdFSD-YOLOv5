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
nvidia-smi
#-------------------------------------
module unload python
module load anaconda
conda activate yolov5
module load cuda/10.2
#-------------------------------------
rm -rf dist
yes | pip uninstall birdfsd_yolov5
poetry build
pip install dist/*.whl
#-------------------------------------
FILTER_CLASSES_UNDER=20

export WANDB_CACHE_DIR="$WORK/.cache"
export WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
export WANDB_PATH_PARENT= # <<<<<<<<<<<<<<<<<<<< @required
#-------------------------------------
if [[ $WANDB_PATH_PARENT == '' ]]; then
    echo 'Missing `WANDB_PATH_PARENT` value!'
    exit 1
fi
#-------------------------------------
mkdir -p archived
mv dataset-YOLO "archived/dataset-YOLO_$(uuid)"
mv ./dataset-YOLO*.tar dataset_config.yml ./*.pt yolov5/runs wandb archived
#-------------------------------------
python birdfsd_yolov5/preprocessing/json2yolov5.py --enable-s3 \
    --filter-cls-with-instances-under "$FILTER_CLASSES_UNDER"
mv dataset-YOLO/dataset_config.yml .
python birdfsd_yolov5/model_utils/relative_to_abs.py
#-------------------------------------
# PRETRAINED_WEIGHTS='yolov5l.pt'
PRETRAINED_WEIGHTS=$(
    python birdfsd_yolov5/model_utils/download_weights.py -v latest -n
)
echo "PRETRAINED_WEIGHTS: $PRETRAINED_WEIGHTS"
python birdfsd_yolov5/model_utils/download_weights.py --model-version latest \
    -o "$PRETRAINED_WEIGHTS"
#-------------------------------------
IMAGE_SIZE=640
BATCH_SIZE=64
EPOCHS=300
#-------------------------------------
python yolov5/train.py \
    --img-size $IMAGE_SIZE \
    --batch $BATCH_SIZE \
    --epochs $EPOCHS \
    --data 'dataset_config.yml' \
    --weights "$PRETRAINED_WEIGHTS" \
    --device 0
#-------------------------------------
DATASET_NAME=$(ls dataset-YOLO-*)
python birdfsd_yolov5/model_utils/update_run_cfg.py \
    --run-path "$WANDB_PATH_PARENT/$WANDB_RUN_ID" \
    --dataset-name "$DATASET_NAME"

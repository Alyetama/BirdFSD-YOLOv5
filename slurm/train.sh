#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --job-name='train'
#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH --constraint='gpu_32gb&gpu_v100'
#SBATCH --mem=32gb
#SBATCH --error=%J.err
#SBATCH --output=%J.out
#-------------------------------------
nvidia-smi
#-------------------------------------
module unload python
module load anaconda
module load cuda/10.2
conda activate yolov5
#-------------------------------------
if [[ "$1" == "" ]] || [[ "$2" == "" ]]; then
    echo "Usage: $0 <WANDB_PROJECT_PATH> <DATASET_NAME> [optional arguments]"
    exit 1
fi
python slurm/generate_options.py -w "$1" -d "$2" || python generate_options.py -w "$1" -d "$2"
set -o allexport; source '.slurm_train_env'; set +o allexport
#-------------------------------------
rm dist/*.whl >/dev/null 2>&1
yes | pip uninstall birdfsd_yolov5
poetry build
pip install dist/*.whl
#-------------------------------------
export WANDB_CACHE_DIR="$WORK/.cache"
_WANDB_RUN_ID=$(python -c "import wandb; print(wandb.util.generate_id())")
export WANDB_RUN_ID="${_WANDB_RUN_ID}"
#-------------------------------------
mkdir -p archived
_UUID="$(uuid)"
for i in dataset-YOLO dataset-YOLO*.tar dataset_config.yml; do
    mv "${i}" "archived/${i}_${_UUID}"
done
#-------------------------------------
python birdfsd_yolov5/preprocessing/json2yolov5.py --enable-s3 \
  --filter-cls-with-instances-under "$FILTER_CLASSES_UNDER" || exit 1
python birdfsd_yolov5/preprocessing/add_bg_images.py
mv dataset-YOLO/dataset_config.yml .
python birdfsd_yolov5/model_utils/relative_to_abs.py
#-------------------------------------
if [[ "$GET_WEIGHTS_FROM_RUN_ID" != "" ]]; then
  WEIGHTS_PATH="${WANDB_PROJECT_PATH}/run_${GET_WEIGHTS_FROM_RUN_ID}_model:best"
  rm best.pt
  wandb artifact get --root . --type model "$WEIGHTS_PATH"
  PRETRAINED_WEIGHTS="best.pt"
else
  PRETRAINED_WEIGHTS=$(
    python birdfsd_yolov5/model_utils/download_weights.py \
      -v latest --object-name-only
  )
  echo "PRETRAINED_WEIGHTS: $PRETRAINED_WEIGHTS"
  python birdfsd_yolov5/model_utils/download_weights.py \
    --model-version latest -o "$PRETRAINED_WEIGHTS"
fi
#-------------------------------------
# shellcheck disable=SC2086
python yolov5/train.py \
  --data 'dataset_config.yml' \
  --img-size $IMAGE_SIZE \
  --batch $BATCH_SIZE \
  --epochs $EPOCHS \
  --weights "$PRETRAINED_WEIGHTS" \
  --device "$DEVICE" \
  --upload-dataset "$DATASET_NAME" \
  --save-period "$SAVE_PERIOD" \
  --patience "$PATIENCE" \
  --optimizer "$OPTIMIZER" \
  --cache "$CACHE_TO" \
  $ADDITIONAL_OPTIONS || exit 1
#-------------------------------------
WANDB_RUN_PATH="$WANDB_PROJECT_PATH/$WANDB_RUN_ID"
python birdfsd_yolov5/model_utils/update_run_cfg.py \
  --run-path "$WANDB_RUN_PATH" \
  --dataset-name "$DATASET_NAME"
python birdfsd_yolov5/model_utils/wandb_helpers.py \
  --add-f1-score-by-run-path "$WANDB_RUN_PATH"

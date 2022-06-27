#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --job-name='train'
#SBATCH --partition=gpu
#SBATCH --gres=gpu
#SBATCH --constraint='gpu_32gb&gpu_v100'
#SBATCH --mem=32gb
#SBATCH --ntasks-per-node=48
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
mkdir -p logs
unlink latest.err
ln -s "${SLURM_JOB_ID}.err" latest.err
unlink latest.out
ln -s "${SLURM_JOB_ID}.out" latest.out
#-------------------------------------
if [[ "$1" == "" ]]; then
    echo "Usage: sbatch train.sh <WANDB_PROJECT_PATH> [optional arguments]"
    exit 1
fi
python slurm/generate_options.py "${@}"
set -o allexport; source '.slurm_train_env'; set +o allexport
cat '.slurm_train_env'
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
_UUID="$(uuidgen)"
for i in dataset-YOLO dataset-YOLO*.tar dataset_config.yml; do
    mv "${i}" "archived/${i}_${_UUID}" >/dev/null 2>&1
done
#-------------------------------------
python birdfsd_yolov5/label_studio_helpers/sync_tasks.py
python birdfsd_yolov5/preprocessing/json2yolov5.py \
  --filter-rare-classes "$FILTER_CLASSES_UNDER"
python birdfsd_yolov5/preprocessing/add_bg_images.py
mv dataset-YOLO/dataset_config.yml .
python birdfsd_yolov5/model_utils/relative_to_abs.py
DATASET_NAME=$(ls dataset-YOLO*.tar)
#-------------------------------------
if [[ "$GET_WEIGHTS_FROM_RUN_ID" != "" ]]; then
  WEIGHTS_PATH="${WANDB_PROJECT_PATH}/run_${GET_WEIGHTS_FROM_RUN_ID}_model:best"
  wandb artifact get --root . --type model "$WEIGHTS_PATH"
  mv "best.pt" "${GET_WEIGHTS_FROM_RUN_ID}.pt"
  PRETRAINED_WEIGHTS="${GET_WEIGHTS_FROM_RUN_ID}.pt"
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
  --upload-dataset \
  --save-period "$SAVE_PERIOD" \
  --patience "$PATIENCE" \
  --optimizer "$OPTIMIZER" \
  --cache "$CACHE_TO" \
  $ADDITIONAL_OPTIONS
#-------------------------------------
WANDB_RUN_PATH="$WANDB_PROJECT_PATH/$WANDB_RUN_ID"
python birdfsd_yolov5/model_utils/update_run_cfg.py \
  --run-path "$WANDB_RUN_PATH" \
  --dataset-name "$DATASET_NAME"
python birdfsd_yolov5/model_utils/wandb_helpers.py \
  --add-f1-score-by-run-path "$WANDB_RUN_PATH"
#-------------------------------------
mv "${SLURM_JOB_ID}.err" "${SLURM_JOB_ID}.out" logs >/dev/null 2>&1

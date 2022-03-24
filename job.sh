#!/bin/sh

git clone https://github.com/bird-feeder/picam-model-azure.git
cd picam-model-azure

pip install -r requirements.txt

python apply_predictions_with_ray.py --project-id 1 --batch-size 256 --model-path "saved_models/1647175692" --class-names "class_names.npy" --exported-tasks "all_tasks.json"

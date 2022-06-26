#!/usr/bin/env python
# coding: utf-8

import argparse
import multiprocessing
import os
import platform
from pathlib import Path

import pynvml as nv
import torch
import torchvision
import wandb


def update_run_cfg(run_path: str, dataset_dir: str, dataset_name: str) -> None:
    """Adds additional metadata to a finished wandb run.

    This function uploads the classes.txt and hist.jpg files to a finished 
    wandb run. It also updates the run's config with the dataset name, base ML 
    framework version, and system hardware.

    """
    api = wandb.Api()
    run = api.run(run_path)

    cwd = os.getcwd()
    os.chdir(dataset_dir)
    run.upload_file('classes.txt')
    run.upload_file('hist.jpg')
    os.chdir(cwd)

    torch_version = torch.__version__
    torchvision_version = torchvision.__version__
    python_version = platform.python_version()
    cuda_version = os.popen('nvcc --version | grep release').read().split(
        ', ')[1].split('release ')[1]

    version = {
        'Python': python_version,
        'CUDA': cuda_version,
        'Torch': torch_version,
        'Torchvision': torchvision_version
    }

    try:
        nv.nvmlInit()
        gpu_count = nv.nvmlDeviceGetCount()
        gpu_type = [
            nv.nvmlDeviceGetName(nv.nvmlDeviceGetHandleByIndex(i)).decode()
            for i in range(gpu_count)
        ]

        system_hardware = {
            'cpu_count': multiprocessing.cpu_count(),
            'gpu_count': gpu_count,
            'gpu_type': ', '.join(gpu_type),
            'nvidia_driver_version': nv.nvmlSystemGetDriverVersion().decode()
        }
    except nv.NVMLError:
        system_hardware = {'cpu_count': multiprocessing.cpu_count()}

    dict_to_add = {
        'dataset_name': Path(dataset_name).name,
        'base_ml_framework': version,
        'system_hardware': system_hardware
    }

    run.config.update(dict_to_add)
    print(dict_to_add)

    run.update()


def _opts() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-p',
        '--run-path',
        help='Path to the W&B run (i.e., `<entity>/<project>/<run_id>`)',
        type=str,
        required=True)
    parser.add_argument('-d',
                        '--dataset-dir',
                        help='Path to the dataset directory',
                        type=str,
                        default='dataset-YOLO')
    parser.add_argument('-D',
                        '--dataset-name',
                        help='Name of the dataset TAR file',
                        type=str,
                        required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = _opts()
    update_run_cfg(run_path=args.run_path,
                   dataset_dir=args.dataset_dir,
                   dataset_name=args.dataset_name)

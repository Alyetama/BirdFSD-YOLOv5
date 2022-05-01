#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import platform

import torch
import torchvision
import wandb


def main():
    api = wandb.Api()
    run = api.run(args.run_path)

    run.upload_file(f'{args.dataset_dir}/classes.txt')

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

    run.config.update({'base_ml_framework': version})
    run.update()


if __name__ == '__main__':
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
                        required=True)
    args = parser.parse_args()

    main()

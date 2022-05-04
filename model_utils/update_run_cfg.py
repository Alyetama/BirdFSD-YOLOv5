#!/usr/bin/env python
# coding: utf-8

import argparse
import os

import wandb


def main() -> None:
    api = wandb.Api()
    run = api.run(args.run_path)

    cwd = os.getcwd()
    os.chdir(args.dataset_dir)
    run.upload_file('classes.txt')
    run.upload_file('hist.jpg')
    os.chdir(cwd)

    # torch_version = torch.__version__
    # torchvision_version = torchvision.__version__
    # python_version = platform.python_version()
    # cuda_version = os.popen('nvcc --version | grep release').read().split(
    #     ', ')[1].split('release ')[1]

    # version = {
    #     'Python': python_version,
    #     'CUDA': cuda_version,
    #     'Torch': torch_version,
    #     'Torchvision': torchvision_version
    # }

    # try:
    #     nv.nvmlInit()
    #     gpu_count = nv.nvmlDeviceGetCount()
    #     gpu_type = [
    #         nv.nvmlDeviceGetName(nv.nvmlDeviceGetHandleByIndex(i))
    #         for i in range(gpu_count)
    #     ]

    #     system_hardware = {
    #         'cpu_count': multiprocessing.cpu_count(),
    #         'gpu_count': gpu_count,
    #         'gpu_type': ', '.join(gpu_type),
    #         'nvidia_driver_version': nv.nvmlSystemGetDriverVersion()
    #     }
    # except nv.NVMLError:
    #     system_hardware = {'cpu_count': multiprocessing.cpu_count()}

    system_hardware = {
        'cpu_count': 40,
        'gpu_count': 1,
        'gpu_type': ', '.join(['Tesla V100-PCIE-32GB']),
        'nvidia_driver_version': '470.103.01'
    }

    version = {
        'Python': '3.8.13',
        'CUDA': '10.2',
        'Torch': '1.11.0',
        'Torchvision': '0.12.0'
    }

    run.config.update({
        'dataset_name': args.dataset_name,
        'base_ml_framework': version,
        'system_hardware': system_hardware
    })

    run.update()
    return


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
                        default='dataset-YOLO')
    parser.add_argument('-D',
                        '--dataset-name',
                        help='Name of the dataset TAR file',
                        type=str,
                        required=True)
    args = parser.parse_args()

    main()

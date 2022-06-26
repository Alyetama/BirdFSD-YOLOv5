#!/usr/bin/env python
# coding: utf-8

import argparse
import sys


def _write_to_file(parser: argparse.Namespace) -> None:
    opts_dict = vars(parser)
    with open('.slurm_train_env', 'w') as f:
        for k, v in opts_dict.items():
            k = k.upper()
            if isinstance(v, str):
                f.write(f'{k}="{v}"\n')
            else:
                f.write(f'{k}={v}\n')


def options_parser() -> argparse.Namespace:
    options = argparse.ArgumentParser()
    required = options.add_argument_group('required arguments')
    required.add_argument('-w',
                          '--wandb-project-path',
                          type=str,
                          required=True,
                          help='W&B project path (i.e., ENTITY/PROJECT)')
    options.add_argument('-g',
                         '--get-weights-from-run-id',
                         type=str,
                         default='',
                         help='Get best weights from run ID')
    options.add_argument('-f',
                         '--filter-classes-under',
                         type=int,
                         default=20,
                         help='Filter out classes under n')
    options.add_argument('-D',
                         '--device',
                         type=str,
                         default='0',
                         help='CUDA device (i.e., 0 or 0,1,2,3 or cpu)')
    options.add_argument('-i',
                         '--image-size',
                         type=int,
                         default=640,
                         help='Train, val image size in pixels')
    options.add_argument('-b',
                         '--batch-size',
                         type=int,
                         default=64,
                         help='Total batch size, -1 for autobatch')
    options.add_argument('-e',
                         '--epochs',
                         type=int,
                         default=300,
                         help='Number of epochs')
    options.add_argument('-o',
                         '--optimizer',
                         type=str,
                         default='SGD',
                         help='Optimizer to use')
    options.add_argument('-c',
                         '--cache-to',
                         type=str,
                         default='ram',
                         help='Cache images in "ram" or "disk"')
    options.add_argument('-p',
                         '--patience',
                         type=int,
                         default=100,
                         help='Early stopping value')
    options.add_argument('-s',
                         '--save-period',
                         type=int,
                         default=100,
                         help='Checkpoint every n')
    options.add_argument('-a',
                         '--additional-options',
                         type=str,
                         default='',
                         help='Additional train options to pass to '
                         'yolov5/train.py. Must be in quotes.')
    return options.parse_args()


if __name__ == '__main__':
    args = options_parser()
    if '-h' not in sys.argv and '--help' not in sys.argv:
        _write_to_file(args)

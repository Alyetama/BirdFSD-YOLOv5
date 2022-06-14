#!/usr/bin/env python
# coding: utf-8

import random
import shutil
from glob import glob
from pathlib import Path


def split_data(output_dir: str, seed: int = 8) -> None:
    """Split the data into train and validation sets.
        
        Args:
            output_dir (str): Path to the output directory.
            seed (int): Initialize the random number generator with n.

    """

    random.seed(seed)

    imgs_full = glob(f'{output_dir}/ls_images/*')
    imgs = [Path(x).stem for x in imgs_full]
    labels_full = glob(f'{output_dir}/ls_labels/*')
    labels = [Path(x).stem for x in labels_full]

    in_imgs_but_not_in_labels = [x for x in imgs if x not in labels]
    in_labels_but_not_in_images = [x for x in labels if x not in imgs]

    imgs_to_delete = [
        x for x in imgs_full if Path(x).stem in in_imgs_but_not_in_labels
    ]
    labels_to_delete = [
        x for x in labels_full if Path(x).stem in in_labels_but_not_in_images
    ]

    for item in imgs_to_delete + labels_to_delete:
        Path(item).unlink()

    for subdir in ['images/train', 'labels/train', 'images/val', 'labels/val']:
        Path(f'{output_dir}/{subdir}').mkdir(parents=True, exist_ok=True)

    images = sorted(glob(f'{output_dir}/ls_images/*'))
    labels = sorted(glob(f'{output_dir}/ls_labels/*'))
    pairs = list(zip(images, labels))

    train_len = round(len(pairs) * 0.8)
    random.shuffle(pairs)

    train, val = pairs[:train_len], pairs[train_len:]

    for im, label in train:
        shutil.copy(im, f'{output_dir}/images/train')
        shutil.copy(label, f'{output_dir}/labels/train')

    for im, label in val:
        shutil.copy(im, f'{output_dir}/images/val')
        shutil.copy(label, f'{output_dir}/labels/val')

    shutil.rmtree(f'{output_dir}/ls_images', ignore_errors=True)
    shutil.rmtree(f'{output_dir}/ls_labels', ignore_errors=True)

#!/usr/bin/env python
# coding: utf-8

import random
import shutil
import sys
from glob import glob
from pathlib import Path

import ray
from dotenv import load_dotenv
from tqdm import tqdm

from birdfsd_yolov5.model_utils.utils import get_data
from birdfsd_yolov5.preprocessing.json2yolov5 import JSON2YOLO


@ray.remote
def _iter_download(task: dict,
                   output_dir: str = 'dataset-YOLO',
                   bg_imgs_dir_name: str = 'bg_images') -> None:
    """Download a background image and save it to the dataset directory.

    Args:
        task (dict): A dictionary containing the task data.
        output_dir (str): Path to the output directory.
        bg_imgs_dir_name (str): Name of the background images' directory.
    """
    j2y = JSON2YOLO(output_dir=output_dir, imgs_dir_name=bg_imgs_dir_name)
    cur_img_path, _, img_url = j2y.get_assets_info(task)
    j2y.download_image(task, cur_img_path, img_url)


def add_bg_images(background_label: str,
    output_dir: str = 'dataset-YOLO',
                  bg_imgs_dir_name: str = 'bg_images',
                  pct: int = 10,
                  seed: int = 8) -> None:
    """Add n percentage of background images to the dataset.

    Args:
        output_dir (str): The dataset directory (the output of
            `JSON2YOLO.run`).
        bg_imgs_dir_name (str): The background images output directory name.
        pct (int): Percentage of background images to keep.
        seed (int): Seed to initialize the random number generator.
    """

    random.seed(seed)

    Path(f'{output_dir}/{bg_imgs_dir_name}').mkdir()

    tasks = get_data(json_min=True)

    bg_images = []

    for task in tasks:
        if not task.get('label'):
            continue
        for x in task['label']:
            for y in x['rectanglelabels']:
                if y == background_label:
                    bg_images.append(task)

    random.shuffle(bg_images)

    total_images_len = len(glob(f'{output_dir}/images/**/*', recursive=True))

    pct_bg_to_keep = int((pct * total_images_len) / 100)
    bg_tasks_sample = random.sample(bg_images, pct_bg_to_keep)

    futures = [_iter_download.remote(x) for x in bg_tasks_sample]
    for x in tqdm(futures):
        ray.get(x)

    bg_images = glob(f'{output_dir}/{bg_imgs_dir_name}/*')
    random.shuffle(bg_images)

    train_len = round(len(bg_images) * 0.8)

    train, val = bg_images[:train_len], bg_images[train_len:]

    for im in train:
        shutil.copy(im, f'{output_dir}/images/train')

    print(f'Copied {len(train)} image to "{output_dir}/images/train".')

    for im in val:
        shutil.copy(im, f'{output_dir}/images/val')

    print(f'Copied {len(val)} image to "{output_dir}/images/val".')


if __name__ == '__main__':
    load_dotenv()
    if len(sys.argv) > 1:
        add_bg_images(sys.argv[1])
    else:
        add_bg_images('no animal')

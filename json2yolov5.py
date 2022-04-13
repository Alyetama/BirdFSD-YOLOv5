#!/usr/bin/env python
# coding: utf-8

import argparse
import imghdr
import json
import os
import random
import shutil
import tarfile
from glob import glob
from pathlib import Path

import ray
import requests
from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm

from mongodb_helper import get_tasks_from_mongodb
"""This script converts the output of a Label-studio project to a YOLO dataset.

The script will create a folder with the following structure:

dataset-YOLO
├── classes.txt
├── dataset_config.yml
├── images
│   ├── train
│   └── val
└── labels
    ├── train
    └── val

The script will also create a .tar file with the same name as the output folder.

The script will also create a tasks_not_exported.json file with the IDs of the
tasks that failed to export
"""


def to_srv(url):
    return url.replace(f'{os.environ["LS_HOST"]}/data/local-files/?d=',
                       f'{os.environ["SRV_HOST"]}/')


def bbox_ls_to_yolo(x, y, width, height):
    x = (x + width / 2) / 100
    y = (y + height / 2) / 100
    w = width / 100
    h = height / 100
    return x, y, w, h


@ray.remote
def convert_to_yolo(task):
    img_url = to_srv(task['image'])
    cur_img_name = Path(img_url).name
    r = requests.get(img_url)
    with open(f'{imgs_dir}/{cur_img_name}', 'wb') as f:
        f.write(r.content)

    try:
        valid_image = imghdr.what(f'{imgs_dir}/{cur_img_name}')
    except FileNotFoundError:
        logger.error(f'Could not validate {imgs_dir}/{cur_img_name}'
                     f'from {task["id"]}! Skipping...')
        try:
            Path(f'{imgs_dir}/{cur_img_name}').unlink()
            return
        except FileNotFoundError:
            logger.error(f'Could not validate {imgs_dir}/{cur_img_name}'
                         f'from {task["id"]}! Skipping...')
            return

    with open(f'{labels_dir}/{Path(cur_img_name).stem}.txt', 'w') as f:
        try:
            labels = task['label']
        except KeyError:
            tasks_not_exported.append(task['id'])
            logger.warning('>>>>>>>>>> CORRUPTED TASK:', task['id'])
            f.close()
            Path(f'{labels_dir}/{Path(cur_img_name).stem}.txt').unlink()
            Path(f'{imgs_dir}/{cur_img_name}').unlink()
            return

        for label in labels:
            if label['rectanglelabels'][0] not in classes:
                f.close()
                Path(f'{labels_dir}/{Path(cur_img_name).stem}.txt').unlink()
                Path(f'{imgs_dir}/{cur_img_name}').unlink()
                return
            x, y, width, height = [
                v for k, v in label.items()
                if k in ['x', 'y', 'width', 'height']
            ]
            x, y, width, height = bbox_ls_to_yolo(x, y, width, height)

            categories = list(enumerate(classes))  # noqa
            label_idx = [
                k[0] for k in categories if k[1] == label['rectanglelabels'][0]
            ][0]

            f.write(f'{label_idx} {x} {y} {width} {height}')
            f.write('\n')


def split_data(_output_dir):
    for subdir in ['images/train', 'labels/train', 'images/val', 'labels/val']:
        Path(f'{_output_dir}/{subdir}').mkdir(parents=True, exist_ok=True)

    images = sorted(glob(f'{Path(__file__).parent}/{_output_dir}/ls_images/*'))
    labels = sorted(glob(f'{Path(__file__).parent}/{_output_dir}/ls_labels/*'))
    pairs = [(x, y) for x, y in zip(images, labels)]

    len_ = len(images)
    train_len = round(len_ * 0.8)
    random.shuffle(pairs)

    train, val = pairs[:train_len], pairs[train_len:]

    for split, split_str in zip([train, val], ['train', 'val']):
        for n, dtype in zip([0, 1], ['images', 'labels']):
            _ = [
                shutil.copy2(
                    x[n],
                    f'{_output_dir}/{dtype}/{split_str}/{Path(x[n]).name}')
                for x in split
            ]


def get_data():
    global classes
    data = []
    for project_id in tqdm(args.projects.split(','), desc='Projects'):
        data.append(
            get_tasks_from_mongodb(project_id, dump=False, json_min=True))
    data = sum(data, [])

    excluded_labels = [
        'cannot identify', 'no animal', 'distorted image',
        'severe occultation', 'animal other than bird or squirrel'
    ]

    labels = []
    for entry in data:
        try:
            labels.append(
                [label['rectanglelabels'][0] for label in entry['label']][0])
        except KeyError as e:
            logger.warning(f'Current entry raised KeyError {e}! '
                           f'Ignoring entry: {entry}')
    labels = list(set(labels))
    logger.info(labels)
    classes = [label for label in labels if label not in excluded_labels]

    Path('dataset-YOLO').mkdir(exist_ok=True)
    with open(f'{output_dir}/classes.txt', 'w') as f:
        for class_ in classes:
            f.write(f'{class_}\n')
    return data, classes


def opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p',
                        '--projects',
                        help='Comma-seperated projects ID',
                        type=str,
                        default=os.environ['PROJECTS_ID'])
    parser.add_argument('-o',
                        '--output-dir',
                        help='Path to the output directory',
                        type=str,
                        default='dataset-YOLO')
    parser.add_argument(
        '--remove',
        help='Remove the output folder and only keep the .tar file',
        action="store_true")
    return parser.parse_args()


def main():
    global classes
    random.seed(8)

    data, classes = get_data()

    Path(imgs_dir).mkdir(parents=True, exist_ok=True)
    Path(labels_dir).mkdir(parents=True, exist_ok=True)

    futures = []
    for task in data:
        futures.append(convert_to_yolo.remote(task))

    results = []
    for future in tqdm(futures):
        result = ray.get(future)
        if result:
            results.append(result)

    if tasks_not_exported:
        with open('tasks_not_exported.json', 'w') as f:
            json.dump(tasks_not_exported, f)

    assert len(glob(f'{output_dir}/images/*')) == len(
        glob(f'{output_dir}/labels/*'))

    split_data(output_dir)
    shutil.rmtree(imgs_dir, ignore_errors=True)
    shutil.rmtree(labels_dir, ignore_errors=True)

    d = {
        'path': f'{output_dir}',
        'train': 'images/train',
        'val': 'images/val',
        'test': '',
        'nc': len(classes),
        'names': classes
    }

    with open(f'{output_dir}/dataset_config.yml', 'w') as f:
        for k, v in d.items():
            f.write(f'{k}: {v}\n')

    folder_name = Path(output_dir).name
    with tarfile.open(f'{folder_name}.tar', 'w') as tar:
        tar.add(output_dir, folder_name)

    if args.remove:
        shutil.rmtree(output_dir, ignore_errors=True)


if __name__ == '__main__':
    load_dotenv()
    args = opts()
    output_dir = args.output_dir
    imgs_dir = f'{output_dir}/ls_images'
    labels_dir = f'{output_dir}/ls_labels'

    classes = None
    tasks_not_exported = []

    main()

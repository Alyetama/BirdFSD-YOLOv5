#!/usr/bin/env python
# coding: utf-8

import argparse
import collections
import imghdr
import json
import os
import random
import shutil
import tarfile
import time
from glob import glob
from pathlib import Path
from typing import Union

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import ray
import requests
import seaborn as sns
from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm

from model_utils.handlers import catch_keyboard_interrupt
from model_utils.minio_helper import BucketDoesNotExist, MinIO
from model_utils.mongodb_helper import get_tasks_from_mongodb
from model_utils.utils import add_logger, upload_logs


class JSON2YOLO:
    """Converts the output of a Label-studio project to a YOLO dataset.

    The output is a folder with the following structure:

    dataset-YOLO
    ├── classes.txt
    ├── dataset_config.yml
    ├── images
    │   ├── train
    │   └── val
    └── labels
        ├── train
        └── val

    The output will also be stored in a tarball with the same name as the output
    folder.

    The ID of the tasks that failed to export for any reason, will be written
    to a file named `tasks_not_exported.json`.
    """

    def __init__(self,
                 projects: str,
                 output_dir: str = 'dataset-YOLO',
                 only_tar_file: bool = False,
                 enable_s3: bool = False):
        self.projects = projects
        self.output_dir = output_dir
        self.only_tar_file = only_tar_file
        self.enable_s3 = enable_s3
        self.imgs_dir = f'{self.output_dir}/ls_images'
        self.labels_dir = f'{self.output_dir}/ls_labels'
        self.classes = None
        self.tasks_not_exported = []

    # @staticmethod
    # def to_srv(url: str) -> str:
    #     return url.replace(f'{os.environ["LS_HOST"]}/data/local-files/?d=',
    #                        f'{os.environ["SRV_HOST"]}/')

    @staticmethod
    def bbox_ls_to_yolo(x: float, y: float, width: float,
                        height: float) -> str:
        x = (x + width / 2) / 100
        y = (y + height / 2) / 100
        w = width / 100
        h = height / 100
        return x, y, w, h

    def get_data(self) -> list:
        data = []
        projects_id = self.projects.split(',')
        for project_id in tqdm(projects_id, desc='Projects'):
            data.append(
                get_tasks_from_mongodb(project_id, dump=False, json_min=True))
        data = sum(data, [])

        excluded_labels = os.getenv('EXCLUDE_LABELS')
        if excluded_labels:
            excluded_labels = excluded_labels.split(',')
        else:
            excluded_labels = []

        labels = []
        for entry in data:
            try:
                labels.append([
                    label['rectanglelabels'][0] for label in entry['label']
                ][0])
            except KeyError as e:
                logger.warning(f'Current entry raised KeyError {e}! '
                               f'Ignoring entry: {entry}')
        labels = list(set(labels))

        self.classes = [
            label for label in labels if label not in excluded_labels
        ]

        logger.debug(f'Number of classes: {len(self.classes)}')
        logger.debug(f'Classes: {self.classes}')

        Path('dataset-YOLO').mkdir(exist_ok=True)
        with open(f'{self.output_dir}/classes.txt', 'w') as f:
            for class_ in self.classes:
                f.write(f'{class_}\n')
        return data

    @ray.remote
    def convert_to_yolo(self, task: dict) -> Union[str, None]:
        # img_url = self.to_srv(task['image'])
        img_url = task['image']
        cur_img_name = Path(img_url).name
        r = requests.get(img_url)
        with open(f'{self.imgs_dir}/{cur_img_name}', 'wb') as f:
            f.write(r.content)

        try:
            valid_image = imghdr.what(f'{self.imgs_dir}/{cur_img_name}')
        except FileNotFoundError:
            logger.error(f'Could not validate {self.imgs_dir}/{cur_img_name}'
                         f'from {task["id"]}! Skipping...')
            try:
                Path(f'{self.imgs_dir}/{cur_img_name}').unlink()
                return
            except FileNotFoundError:
                logger.error(
                    f'Could not validate {self.imgs_dir}/{cur_img_name}'
                    f'from {task["id"]}! Skipping...')
                return

        with open(f'{self.labels_dir}/{Path(cur_img_name).stem}.txt',
                  'w') as f:
            try:
                labels = task['label']
            except KeyError:
                self.tasks_not_exported.append(task['id'])
                logger.warning('>>>>>>>>>> CORRUPTED TASK:', task['id'])
                f.close()
                Path(f'{self.labels_dir}/{Path(cur_img_name).stem}.txt'
                     ).unlink()
                Path(f'{self.imgs_dir}/{cur_img_name}').unlink()
                return

            for label in labels:
                if label['rectanglelabels'][0] not in self.classes:
                    f.close()
                    Path(f'{self.labels_dir}/{Path(cur_img_name).stem}.txt'
                         ).unlink()
                    Path(f'{self.imgs_dir}/{cur_img_name}').unlink()
                    return
                x, y, width, height = [
                    v for k, v in label.items()
                    if k in ['x', 'y', 'width', 'height']
                ]
                x, y, width, height = self.bbox_ls_to_yolo(x, y, width, height)

                categories = list(enumerate(self.classes))  # noqa
                label_idx = [
                    k[0] for k in categories
                    if k[1] == label['rectanglelabels'][0]
                ][0]

                f.write(f'{label_idx} {x} {y} {width} {height}')
                f.write('\n')
        return label['rectanglelabels'][0]  # to create a count table

    @staticmethod
    def split_data(_output_dir: str) -> None:
        for subdir in [
                'images/train', 'labels/train', 'images/val', 'labels/val'
        ]:
            Path(f'{_output_dir}/{subdir}').mkdir(parents=True, exist_ok=True)

        images = sorted(
            glob(f'{Path(__file__).parent}/{_output_dir}/ls_images/*'))
        labels = sorted(
            glob(f'{Path(__file__).parent}/{_output_dir}/ls_labels/*'))
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
        return

    def plot_results(self, results: list) -> None:
        matplotlib.use('Agg')
        plt.subplots(figsize=(12, 8), dpi=300)
        plt.xticks(rotation=90)

        sns.histplot(data=results, kde=True)
        plt.title('Distribution Of Classes With A Kernel Density Estimate')
        plt.savefig(f'{self.output_dir}/bar.jpg', bbox_inches='tight')
        plt.cla()

        data_count = dict(collections.Counter(results))
        df = pd.DataFrame(data_count.items(), columns=['label', 'count'])
        df = df.groupby('label').sum().reset_index()

        ax = sns.barplot(data=df,
                         x='label',
                         y='count',
                         hue='label',
                         dodge=False)
        plt.xticks(rotation=90)
        plt.title('Instances Per Class')
        ax.get_legend().remove()
        plt.savefig(f'{self.output_dir}/hist.jpg', bbox_inches='tight')
        return

    def run(self):
        logs_file = add_logger(__file__)
        catch_keyboard_interrupt()
        random.seed(8)

        data = self.get_data()

        Path(self.imgs_dir).mkdir(parents=True, exist_ok=True)
        Path(self.labels_dir).mkdir(parents=True, exist_ok=True)

        futures = []
        for task in data:
            futures.append(self.convert_to_yolo.remote(self, task))

        results = []
        for future in tqdm(futures):
            result = ray.get(future)
            if result:
                results.append(result)

        if results:
            self.plot_results(results)

        if self.tasks_not_exported:
            with open('tasks_not_exported.json', 'w') as f:
                json.dump(self.tasks_not_exported, f)

        assert len(glob(f'{self.output_dir}/images/*')) == len(
            glob(f'{self.output_dir}/labels/*'))

        self.split_data(self.output_dir)

        shutil.rmtree(self.imgs_dir, ignore_errors=True)
        shutil.rmtree(self.labels_dir, ignore_errors=True)

        d = {
            'path': f'{self.output_dir}',
            'train': 'images/train',
            'val': 'images/val',
            'test': '',
            'nc': len(self.classes),
            'names': self.classes
        }

        with open(f'{self.output_dir}/dataset_config.yml', 'w') as f:
            for k, v in d.items():
                f.write(f'{k}: {v}\n')

        folder_name = Path(self.output_dir).name
        with tarfile.open(f'{folder_name}.tar', 'w') as tar:
            tar.add(self.output_dir, folder_name)

        if self.only_tar_file:
            shutil.rmtree(self.output_dir, ignore_errors=True)

        if self.enable_s3:
            minio = MinIO()
            if not minio.client.bucket_exists('dataset'):
                raise BucketDoesNotExist('Bucket `dataset` does not exist!')

            objs = list(minio.client.list_objects('dataset'))
            latest_ts = max([o.last_modified for o in objs])
            latest_obj = [o for o in objs if o.last_modified == latest_ts][0]

            if latest_obj.size != Path(f'{folder_name}.tar').stat().st_size:
                logger.info('Uploading dataset...')
                minio.client.fput_object('dataset',
                                         f'{folder_name}-{time.time()}.tar',
                                         f'{folder_name}.tar')

        upload_logs(logs_file)
        return


if __name__ == '__main__':
    load_dotenv()

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
    parser.add_argument('--only-tar-file',
                        help='Only output a TAR file',
                        action="store_true")
    parser.add_argument('--enable-s3',
                        help='Upload the output to an S3 bucket',
                        action="store_true")
    args = parser.parse_args()

    json2yolo = JSON2YOLO(projects=args.projects,
                          output_dir=args.output_dir,
                          only_tar_file=args.only_tar_file,
                          enable_s3=args.enable_s3)
    json2yolo.run()

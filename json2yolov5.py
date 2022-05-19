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
from datetime import datetime, timedelta
from glob import glob
from pathlib import Path
from typing import Optional

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ray
import requests
import seaborn as sns
from dotenv import load_dotenv
from loguru import logger
from requests.structures import CaseInsensitiveDict
from tqdm import tqdm

from model_utils.handlers import catch_keyboard_interrupt
from model_utils.mongodb_helper import get_tasks_from_mongodb
from model_utils.s3_helper import BucketDoesNotExist, S3
from model_utils.utils import tasks_data, get_labels_count, get_project_ids_str


class FailedToParseImageURL(Exception):
    pass


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

    The output will also be stored in a tarball with the same name as the
     output folder.

    The tasks that failed to export for any reason, will be logged at the 
    end of the run.
    """

    def __init__(self,
                 projects: str,
                 output_dir: str = 'dataset-YOLO',
                 only_tar_file: bool = False,
                 enable_s3: bool = True,
                 copy_data_from: str = None,
                 filter_underrepresented_cls: bool = False,
                 filter_cls_with_instances_under: Optional[int] = None,
                 get_tasks_with_api: bool = False,
                 force_update: bool = False):
        self.projects = projects
        self.output_dir = output_dir
        self.only_tar_file = only_tar_file
        self.enable_s3 = enable_s3
        self.copy_data_from = copy_data_from
        self.filter_underrepresented_cls = filter_underrepresented_cls
        self.filter_cls_with_instances_under = filter_cls_with_instances_under
        self.get_tasks_with_api = get_tasks_with_api
        self.force_update = force_update
        self.imgs_dir = f'{self.output_dir}/ls_images'
        self.labels_dir = f'{self.output_dir}/ls_labels'
        self.classes = None
        self.tasks_not_exported = []

    @staticmethod
    def bbox_ls_to_yolo(x: float, y: float, width: float,
                        height: float) -> tuple:
        x = (x + width / 2) / 100
        y = (y + height / 2) / 100
        w = width / 100
        h = height / 100
        return x, y, w, h

    def get_data(self) -> list:

        @ray.remote
        def iter_projects(proj_id):
            return get_tasks_from_mongodb(proj_id, dump=False, json_min=True)

        @ray.remote
        def iter_projects_api(proj_id):
            headers = CaseInsensitiveDict()
            headers['Content-type'] = 'application/json'
            headers['Authorization'] = f'Token {os.environ["TOKEN"]}'
            ls_host = os.environ["LS_HOST"]
            q = 'exportType=JSON&download_all_tasks=true'
            proj_tasks = []
            url = f'{ls_host}/api/projects/{proj_id}/export?{q}'
            resp = requests.get(url, headers=headers)
            proj_tasks.append(resp.json())
            return proj_tasks

        if self.projects:
            project_ids = self.projects.split(',')
        else:
            project_ids = get_project_ids_str().split(',')

        futures = []
        for project_id in project_ids:
            if self.get_tasks_with_api:
                futures.append(iter_projects_api.remote(project_id))
            else:
                futures.append(iter_projects.remote(project_id))

        data = []
        for future in tqdm(futures, desc='Projects'):
            data.append(ray.get(future))

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

        unique, counts = np.unique(labels, return_counts=True)

        if (self.filter_underrepresented_cls
                or self.filter_cls_with_instances_under):
            if self.filter_underrepresented_cls:
                min_instances = np.median(counts)
            else:
                min_instances = self.filter_cls_with_instances_under
            self.classes = sorted([
                label for label, count in zip(unique, counts)
                if label not in excluded_labels and count >= min_instances
            ])
        else:
            self.classes = sorted(
                [label for label in unique if label not in excluded_labels])

        logger.debug(f'Number of classes: {len(self.classes)}')
        logger.debug(f'Classes: {self.classes}')

        Path(self.output_dir).mkdir(exist_ok=True)
        with open(f'{self.output_dir}/classes.txt', 'w') as f:
            for class_ in self.classes:
                f.write(f'{class_}\n')
        return data

    def convert_to_yolo(self, task: dict) -> Optional[str]:
        if self.copy_data_from or self.enable_s3:
            img = task['image']
            if img.startswith('s3://'):
                object_name = img.split('s3://data/')[-1]
            elif img.startswith('http'):
                object_url = img.split('?')[0]
                object_name = '/'.join(Path(object_url).parts[-2:])
            else:
                raise FailedToParseImageURL(img)
            cur_img_name = Path(object_name).name
        else:
            if 's3://' in task['image'] and not self.enable_s3:
                raise TypeError('You need to pass the flag `--enable-s3` '
                                'for S3 objects!')
            object_name = None
            cur_img_name = Path(task['image']).name

        cur_img_path = f'{self.imgs_dir}/{cur_img_name}'
        cur_label_path = f'{self.labels_dir}/{Path(cur_img_name).stem}.txt'

        if self.copy_data_from:
            shutil.copy(f'{self.copy_data_from}/{object_name}', cur_img_path)
        else:
            if self.enable_s3:
                img_url = S3().client.presigned_get_object(
                    'data', object_name, expires=timedelta(hours=6))
            else:
                img_url = task['image']
            r = requests.get(img_url)
            if '<Error>' in r.text:
                logger.error(
                    f'Could not download the image `{img_url}`! Skipping...')
                return

            with open(cur_img_path, 'wb') as f:
                f.write(r.content)

        try:
            valid_image = imghdr.what(cur_img_path)
            if not valid_image:
                logger.error(f'{cur_img_path} is not valid (task'
                             f' id: {task["id"]})! Skipping...')
                Path(cur_img_path).unlink()
        except FileNotFoundError:
            logger.error(
                f'Could not validate {cur_img_path} from {task["id"]}! '
                'Skipping...')
            return

        with open(cur_label_path, 'w') as f:
            try:
                labels = task['label']
            except KeyError:
                self.tasks_not_exported.append(task)
                logger.error(f'>>>>>>>>>> CORRUPTED TASK: {task}')
                f.close()
                Path(cur_label_path).unlink()
                Path(cur_img_path).unlink()
                return

            for label in labels:
                if label['rectanglelabels'][0] not in self.classes:
                    f.close()
                    Path(cur_label_path).unlink()
                    Path(cur_img_path).unlink()
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

        @ray.remote
        def iter_convert_to_yolo(t):
            return self.convert_to_yolo(t)

        s3_client = S3().client
        catch_keyboard_interrupt()
        random.seed(8)

        tasks = self.get_data()

        Path(self.imgs_dir).mkdir(parents=True, exist_ok=True)
        Path(self.labels_dir).mkdir(parents=True, exist_ok=True)

        futures = []
        for task in tasks:
            futures.append(iter_convert_to_yolo.remote(task))
        results = []
        for future in tqdm(futures, desc='Tasks'):
            results.append(ray.get(future))

        if results:
            self.plot_results(results)

        if self.tasks_not_exported:
            logger.error(f'Corrupted tasks: {self.tasks_not_exported}')

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

        tasks_data(f'tasks.json')

        with open('classes.json', 'w') as f:
            if not os.getenv('EXCLUDE_LABELS'):
                os.environ['EXCLUDE_LABELS'] = []

            classes_json = {
                k: v
                for k, v in get_labels_count().items()
                if k not in os.getenv('EXCLUDE_LABELS')
            }
            if self.filter_cls_with_instances_under:
                classes_json = {
                    k: v
                    for k, v in classes_json.items()
                    if v > self.filter_cls_with_instances_under
                }
            json.dump(classes_json, f, indent=4)

        folder_name = Path(self.output_dir).name
        ts = datetime.now().strftime('%m-%d-%Y_%H.%M.%S')
        dataset_name = f'{folder_name}-{ts}.tar'

        with tarfile.open(dataset_name, 'w') as tar:
            tar.add(self.output_dir, folder_name)

        if self.only_tar_file:
            shutil.rmtree(self.output_dir, ignore_errors=True)

        if self.enable_s3:
            if not s3_client.bucket_exists('dataset'):
                raise BucketDoesNotExist('Bucket `dataset` does not exist!')

            upload_dataset = False
            objs = list(s3_client.list_objects('dataset'))
            if objs:
                latest_ts = max(
                    [o.last_modified for o in objs if o.last_modified])
                latest_obj = [o for o in objs
                              if o.last_modified == latest_ts][0]
                if latest_obj.size != Path(
                        dataset_name).stat().st_size or self.force_update:
                    upload_dataset = True
            else:
                upload_dataset = True
            if upload_dataset:
                if self.copy_data_from:
                    logger.debug('Copying the dataset to the bucket...')
                    ds_path = f'{Path(self.copy_data_from).parent}/dataset'
                    shutil.copy(dataset_name, f'{ds_path}/{dataset_name}')
                else:
                    logger.info('Uploading the dataset...')
                    s3_client.fput_object('dataset', dataset_name,
                                          dataset_name)
        return


if __name__ == '__main__':
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-p',
        '--projects',
        help='Comma-seperated projects ID. If empty, it will select all '
        'projects',
        type=str)
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
    parser.add_argument(
        '--copy-data-from',
        help='If running on the same host serving the S3 objects, you can '
        'use this option to specify a path to copy the data from '
        '(i.e., the local path to the S3 bucket where the data is '
        'stored) instead of downloading it',
        type=str)
    parser.add_argument('--filter-underrepresented-cls',
                        help='Only include classes with instances equal or '
                        'above the overall median',
                        action="store_true")
    parser.add_argument(
        '--filter-cls-with-instances-under',
        help='Remove the class from the dataset if the annotation instances '
        'is lower than n',
        type=int)
    parser.add_argument('--get-tasks-with-api',
                        help='Use label-studio API to get tasks data',
                        action="store_true")
    parser.add_argument(
        '--force-update',
        help='Update the dataset even when it appears to be identical to the '
        'latest dataset',
        action="store_true")
    args = parser.parse_args()

    json2yolo = JSON2YOLO(
        projects=args.projects,
        output_dir=args.output_dir,
        only_tar_file=args.only_tar_file,
        enable_s3=args.enable_s3,
        copy_data_from=args.copy_data_from,
        filter_underrepresented_cls=args.filter_underrepresented_cls,
        filter_cls_with_instances_under=args.filter_cls_with_instances_under,
        get_tasks_with_api=args.get_tasks_with_api,
        force_update=args.force_update)
    json2yolo.run()

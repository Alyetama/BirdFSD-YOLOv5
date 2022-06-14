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
from datetime import datetime, timedelta
from glob import glob
from pathlib import Path
from typing import List, Optional, Tuple, Union

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

from birdfsd_yolov5.model_utils import (handlers, mongodb_helper, s3_helper,
                                        utils)
from birdfsd_yolov5.preprocessing.split_data import split_data


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
    ....├── train
    ....└── val

    The output will also be stored in a tarball with the same name as the
    output folder.

    The tasks that failed to export for any reason, will be logged at the 
    end of the run.
    """

    def __init__(self,
                 output_dir: str = 'dataset-YOLO',
                 projects: Optional[str] = None,
                 copy_data_from: Optional[str] = None,
                 filter_rare_classes: Optional[str] = None,
                 get_tasks_with_api: bool = False,
                 force_update: bool = False,
                 background_label: str = 'no animal',
                 upload_dataset: bool = False,
                 alt_data_endpoint: Optional[str] = None,
                 excluded_labels: Union[list, str] = None,
                 seed: int = 8,
                 overwrite: bool = False,
                 verbose: bool = False,
                 imgs_dir_name: str = 'ls_images',
                 labels_dir_name: str = 'ls_labels'):
        self.projects = projects
        self.output_dir = str(Path(output_dir).absolute())
        self.copy_data_from = copy_data_from
        self.filter_rare_classes = filter_rare_classes
        self.get_tasks_with_api = get_tasks_with_api
        self.force_update = force_update
        self.background_label = background_label
        self.upload_dataset = upload_dataset
        self.alt_data_endpoint = alt_data_endpoint
        self.excluded_labels = excluded_labels
        self.seed = seed
        self.overwrite = overwrite
        self.verbose = verbose
        self.imgs_dir_name = imgs_dir_name
        self.labels_dir_name = labels_dir_name
        self.imgs_dir = f'{self.output_dir}/{imgs_dir_name}'
        self.labels_dir = f'{self.output_dir}/{labels_dir_name}'
        self.classes = None

    @staticmethod
    def bbox_ls_to_yolo(x: float, y: float, width: float,
                        height: float) -> tuple:
        """From label-studio's xywh to yolov5's xywh.

        Converts a bounding box from the format used by the labelme tool to
        the format used by the yolo tool.

        Args:
            x: The x coordinate of the top left corner of the bounding box.
            y: The y coordinate of the top left corner of the bounding box.
            width: The width of the bounding box.
            height: The height of the bounding box.

        Returns:
            tuple: A tuple containing the x, y, width and height of the 
            bounding box in the format used by the yolov5.

        """
        x = (x + width / 2) / 100
        y = (y + height / 2) / 100
        w = width / 100
        h = height / 100
        return x, y, w, h

    def count_labels(self, data: list) -> None:
        excluded_labels = self.get_excluded_labels()
        labels = []
        for entry in data:
            if not entry.get('label'):
                continue
            try:
                labels.append([
                    label['rectanglelabels'][0] for label in entry['label']
                ][0])
            except KeyError as e:
                logger.warning(f'Current entry raised KeyError "{e}"! '
                               f'Ignoring entry: {entry}')

        unique, counts = np.unique(labels, return_counts=True)

        min_instances = 1
        if self.filter_rare_classes:
            if self.filter_rare_classes.isdigit():
                min_instances = int(self.filter_rare_classes)
            elif self.filter_rare_classes.lower() == 'median':
                min_instances = np.median(counts)
            elif self.filter_rare_classes.lower() == 'mean':
                min_instances = np.mean(counts)

        self.classes = sorted([
            label for label, count in zip(unique, counts)
            if label not in excluded_labels and count >= min_instances
        ])

        logger.debug(f'Number of classes: {len(self.classes)}')
        logger.debug(f'Classes: {self.classes}')

    def get_data(self) -> list:
        """This function is used to get data from the database.

        Returns:
            list: A list of data.

        """

        @ray.remote
        def iter_projects(proj_id):
            return mongodb_helper.get_tasks_from_mongodb(proj_id,
                                                         dump=False,
                                                         json_min=True)

        @ray.remote
        def iter_projects_api(proj_id):
            headers = CaseInsensitiveDict()
            headers['Content-type'] = 'application/json'
            headers['Authorization'] = f'Token {os.environ["TOKEN"]}'
            ls_host = os.environ['LS_HOST']
            q = 'exportType=JSON&download_all_tasks=true'
            proj_tasks = []
            url = f'{ls_host}/api/projects/{proj_id}/export?{q}'
            resp = requests.get(url, headers=headers)
            proj_tasks.append(resp.json())
            return proj_tasks

        if self.projects:
            project_ids = self.projects.split(',')
        else:
            project_ids = utils.get_project_ids_str().split(',')

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
        self.count_labels(data)

        Path(self.output_dir).mkdir(exist_ok=True)
        with open(f'{self.output_dir}/classes.txt', 'w') as f:
            for class_ in self.classes:
                f.write(f'{class_}\n')
        return data

    def get_assets_info(self, task: dict) -> tuple:
        is_s3 = False
        img = task['image']
        if 's3://' in task['image']:
            is_s3 = True
        # Get name and URL of the image.
        if self.copy_data_from or is_s3:
            img = task['image']
            if img.startswith('s3://'):
                object_name = img.split('s3://data/')[-1]
            elif img.startswith('http'):
                img = img.split('?')[0]
                object_name = '/'.join(Path(img).parts[-2:])
            else:
                raise FailedToParseImageURL(img)
            cur_img_name = Path(object_name).name
        else:
            object_name = None
            cur_img_name = Path(task['image']).name

        # Define the path to which the image and label will be written.
        if '?' in cur_img_name:
            cur_img_name = cur_img_name.split('?')[0]
        cur_img_path = f'{self.imgs_dir}/{cur_img_name}'
        cur_label_path = f'{self.labels_dir}/{Path(cur_img_name).stem}.txt'

        # Write the image to local disk.
        if self.copy_data_from:
            shutil.copy(f'{self.copy_data_from}/{object_name}', cur_img_path)
        else:
            if self.alt_data_endpoint:
                if '?' in img:
                    img = img.split('?')[0]
                src_img_endpoint = task['image'].split('/data')[0]
                img = img.replace(src_img_endpoint, self.alt_data_endpoint)
            elif is_s3:
                img = s3_helper.S3().client.presigned_get_object(
                    'data', object_name, expires=timedelta(hours=6))
        return cur_img_path, cur_label_path, img

    def download_image(self, task: dict, cur_img_path: str,
                       img_url: str) -> Optional[bool]:
        if self.verbose:
            print(f'Downloading {img_url}...')

        if not self.copy_data_from:
            r = requests.get(img_url)
            if '<Error>' in r.text or r.status_code != 200:
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
                return
        except FileNotFoundError:
            logger.error(
                f'Could not validate {cur_img_path} from {task["id"]}! '
                'Skipping...')
            return
        return True

    def convert_to_yolo(
        self, task: dict
    ) -> Union[None, Tuple[List[str], str], Tuple[List[str], None]]:
        """Convert the task to YOLO format.

        Args:
            task (dict): The task to be converted.

        Returns:
            Optional[Tuple[list, list]]: A tuple with a list of the labels in
            the task and a list of background image path if the task is
            labeled as a background image.

        Raises:
            FailedToParseImageURL: If the image URL is not valid.
            TypeError: If the image URL is not valid.

        """
        cur_img_path, cur_label_path, img_url = self.get_assets_info(task)
        valid_download = self.download_image(task, cur_img_path, img_url)
        if not valid_download:
            return


        if task.get('label'):
            labels = task['label']
        else:
            try:
                Path(cur_img_path).unlink()
            except FileNotFoundError:
                pass
            return

        label_names = []
        label_file_content = ''

        # Iterate through annotations in a single task
        for label in labels:
            if label['rectanglelabels'][0] not in self.classes:
                Path(cur_img_path).unlink()
                return

            label_names.append(label['rectanglelabels'][0])
            x, y, width, height = [
                v for k, v in label.items()
                if k in ['x', 'y', 'width', 'height']
            ]
            x, y, width, height = self.bbox_ls_to_yolo(x, y, width, height)

            categories = list(enumerate(self.classes))
            label_idx = [
                k[0] for k in categories if k[1] == label['rectanglelabels'][0]
            ][0]

            label_file_content += f'{label_idx} {x} {y} {width} {height}\n'

        with open(cur_label_path, 'w') as f:
            f.write(label_file_content)
        return label_names

    def plot_results(self, results: list) -> None:
        """Plots the results of the classification.

        Args:
            results (list): The results of the classification.

        Returns:
            None

        """
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

    def upload_dataset_file(self, dataset_name: str) -> None:
        s3_client = s3_helper.S3().client

        if not s3_client.bucket_exists('dataset'):
            raise s3_helper.BucketDoesNotExist(
                'Bucket `dataset` does not exist!')

        # upload_dataset = False
        # objs = list(s3_client.list_objects('dataset'))
        # if objs:
        #     latest_ts = max([o.last_modified for o in objs if o.last_modified])
        #     latest_obj = [o for o in objs if o.last_modified == latest_ts][0]
        #     if latest_obj.size != Path(
        #             dataset_name).stat().st_size or self.force_update:
        #         upload_dataset = True
        # else:
        #     upload_dataset = True

        if upload_dataset:
            if self.copy_data_from:
                logger.debug('Copying the dataset to the bucket...')
                ds_path = f'{Path(self.copy_data_from).parent}/dataset'
                shutil.copy(dataset_name, f'{ds_path}/{dataset_name}')
            else:
                logger.info('Uploading the dataset...')
                s3_client.fput_object('dataset', dataset_name, dataset_name)

    def _create_metadata_files(self) -> None:
        excluded_labels = self.get_excluded_labels()
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

        utils.tasks_data(f'{self.output_dir}/tasks.json')

        with open(f'{self.output_dir}/classes.json', 'w') as f:

            classes_json = {
                k: v
                for k, v in utils.get_labels_count().items()
                if k not in excluded_labels
            }
            if self.filter_rare_classes.isdigit():
                classes_json = {
                    k: v
                    for k, v in classes_json.items()
                    if v > int(self.filter_rare_classes)
                }
            json.dump(classes_json, f, indent=4)

    def get_excluded_labels(self):
        if self.excluded_labels:
            excluded_labels = self.excluded_labels
        elif not self.excluded_labels and os.getenv('EXCLUDE_LABELS'):
            excluded_labels = os.getenv('EXCLUDE_LABELS')
        else:
            excluded_labels = []
        if isinstance(excluded_labels, str):
            excluded_labels = excluded_labels.split(',')
        return excluded_labels

    def run(self) -> None:
        """Runs the preprocessing pipeline.

        This method is used to run main preprocessing pipeline and convert
        the data to the yolov5 format.

        Returns:
            None

        Raises:
            BucketDoesNotExist: If the dataset S3 bucket does not exist.

        """

        @ray.remote
        def iter_convert_to_yolo(t):
            return self.convert_to_yolo(t)

        random.seed(self.seed)
        handlers.catch_keyboard_interrupt()

        if Path(self.output_dir).exists():
            if self.overwrite:
                shutil.rmtree(self.output_dir, ignore_errors=True)
            else:
                raise FileExistsError('The output folder already exists!')

        tasks = self.get_data()

        Path(self.imgs_dir).mkdir(parents=True, exist_ok=True)
        Path(self.labels_dir).mkdir(parents=True, exist_ok=True)

        with open(f'{self.output_dir}/notes.json', 'w') as j:
            json.dump({'seed': self.seed}, j, indent=4)

        futures = [iter_convert_to_yolo.remote(task) for task in tasks]
        results = []
        for future in tqdm(futures, desc='Tasks'):
            try:
                result = ray.get(future)
            except requests.exceptions.ChunkedEncodingError:
                time.sleep(2)
                result = ray.get(future)
            results.append(result)
            time.sleep(0.01)

        results_labels = sum([x for x in results if x], [])

        split_data(self.output_dir, seed=self.seed)
        self.plot_results(results_labels)
        self._create_metadata_files()

        folder_name = Path(self.output_dir).name
        ts = datetime.now().strftime('%m-%d-%Y_%H.%M.%S')
        dataset_name = f'{folder_name}-{ts}.tar'

        with tarfile.open(dataset_name, 'w') as tar:
            tar.add(self.output_dir, folder_name)

        if self.upload_dataset:
            self.upload_dataset_file(dataset_name)


def _opts() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-o',
                        '--output-dir',
                        help='Path to the output directory',
                        type=str,
                        default='dataset-YOLO')
    parser.add_argument(
        '-p',
        '--projects',
        help='Comma-seperated projects ID. If empty, it will select all '
        'projects',
        type=str)
    parser.add_argument(
        '--copy-data-from',
        help='If running on the same host serving the S3 objects, you can '
        'use this option to specify a path to copy the data from '
        '(i.e., the local path to the S3 bucket where the data is '
        'stored) instead of downloading it',
        type=str)
    parser.add_argument('-f',
                        '--filter-rare-classes',
                        help='Only include classes with instances equal or '
                        'above the median (default), mean, or an integer',
                        default='median')
    parser.add_argument('--get-tasks-with-api',
                        help='Use label-studio API to get tasks data',
                        action='store_true')
    parser.add_argument(
        '-F',
        '--force-update',
        help='Update the dataset even when it appears to be identical to the '
        'latest dataset',
        action='store_true')
    parser.add_argument('-u',
                        '--upload-dataset',
                        help='Upload the output dataset to the data server ('
                        'S3 only)',
                        action='store_true')
    parser.add_argument('-B',
                        '--background-label',
                        help='Label of background images',
                        type=str,
                        default='no animal')
    parser.add_argument('-a',
                        '--alt-data-endpoint',
                        help='Alternative endpoint to use for data files',
                        type=str)
    parser.add_argument('-e',
                        '--excluded-labels',
                        help='Labels to exclude from the output dataset ('
                        'as a comma-seperated string of labels)',
                        type=str)
    parser.add_argument('-s',
                        '--seed',
                        help='Initialize the random number generator',
                        type=int,
                        default=8)
    parser.add_argument('--overwrite',
                        help='Overwrite the output folder if exists',
                        action='store_true')
    parser.add_argument('-v', '--verbose', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    load_dotenv()
    args = _opts()
    json2yolo = JSON2YOLO(output_dir=args.output_dir,
                          projects=args.projects,
                          copy_data_from=args.copy_data_from,
                          filter_rare_classes=args.filter_rare_classes,
                          get_tasks_with_api=args.get_tasks_with_api,
                          force_update=args.force_update,
                          upload_dataset=args.upload_dataset,
                          alt_data_endpoint=args.alt_data_endpoint,
                          excluded_labels=args.excluded_labels,
                          seed=args.seed,
                          overwrite=args.overwrite,
                          verbose=args.verbose)
    json2yolo.run()

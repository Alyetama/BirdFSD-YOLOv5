#!/usr/bin/env python
# coding: utf-8

import hashlib
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import ray
from dotenv import load_dotenv
from tqdm import tqdm

from mongodb_helper import get_tasks_from_mongodb
from s3_helper import S3
from utils import get_project_ids_str


def get_all_tasks_from_mongodb():

    @ray.remote
    def _get_all_tasks_from_mongodb(proj_id):
        return get_tasks_from_mongodb(proj_id, dump=False, json_min=True)

    project_ids = get_project_ids_str().split(',')

    futures = []
    for project_id in project_ids:
        futures.append(_get_all_tasks_from_mongodb.remote(project_id))
    tasks = []
    for future in tqdm(futures, desc='Projects'):
        tasks.append(ray.get(future))
    return sum(tasks, [])


@ray.remote
def simplify(task):
    labels = []
    label_val = task.get('label')
    if not label_val:
        print(task, 'is corrupted!', 'Skipping...')
        return
    for n, entry in enumerate(label_val):
        xywh = (entry['x'], entry['y'], entry['width'], entry['height'])
        label = entry['rectanglelabels']
        image_width = entry['original_width']
        image_height = entry['original_height']
        labels.append({
            n: {
                'class': label,
                'xywh': xywh,
                'image_width': image_width,
                'image_height': image_height
            }
        })

    object_url = task['data']['image'].split('?')[0]
    object_name = '/'.join(Path(object_url).parts[-2:])
    img_data = S3().client.get_object('data', object_name).read()

    simple_task = {
        'image': {
            'data': img_data,
            'md5_hash': hashlib.md5(img_data).hexdigest()
        },
        'labels': labels
    }
    return simple_task


def main():
    tasks = get_all_tasks_from_mongodb()

    futures = [simplify.remote(task) for task in tasks]
    results = []
    for future in tqdm(futures):
        try:
            results.append(ray.get(future))
        except ConnectionResetError as e:
            print('ERROR:', e)
            futures.append(future)
            time.sleep(10)

    d = {k: v for k, v in enumerate(results) if v}

    df = pd.DataFrame.from_dict(d)
    ts = datetime.now().strftime('%m-%d-%Y_%H.%M.%S')
    ds_name = f'{ts}.parquet'
    df.to_parquet(ds_name)


if __name__ == '__main__':
    load_dotenv()
    main()

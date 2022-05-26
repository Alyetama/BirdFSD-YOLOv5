#!/usr/bin/env python
# coding: utf-8

import hashlib
import time
from datetime import datetime
from pathlib import Path

import pandas as pd
import ray
from dotenv import load_dotenv
from ray.exceptions import RayTaskError
from tqdm import tqdm

from birdfsd_yolov5.model_utils import mongodb_helper, s3_helper, utils


def get_all_tasks_from_mongodb():
    """This function is used to get all the tasks from mongodb.

    Returns:
        list: A list of all the tasks in the database.
    """

    @ray.remote
    def _get_all_tasks_from_mongodb(proj_id):
        return mongodb_helper.get_tasks_from_mongodb(proj_id,
                                                     dump=False,
                                                     json_min=True)

    project_ids = utils.get_project_ids_str().split(',')

    futures = []
    for project_id in project_ids:
        futures.append(_get_all_tasks_from_mongodb.remote(project_id))
    tasks = []
    for future in tqdm(futures, desc='Projects'):
        tasks.append(ray.get(future))
    return sum(tasks, [])


@ray.remote
def simplify(task):
    """This function takes a task from the original dataset and simplifies it
    to a format that is easier to work with.

    Args:
        task (dict): A task from the original dataset.

    Returns:
        dict: A simplified task.
    """
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
            str(n): {
                'class': label,
                'xywh': xywh,
                'image_width': image_width,
                'image_height': image_height
            }
        })

    object_url = task['data']['image'].split('?')[0]
    object_name = '/'.join(Path(object_url).parts[-2:])
    img_data = s3_helper.S3().client.get_object('data', object_name).read()

    simple_task = {
        'image': {
            'data': img_data,
            'md5_hash': hashlib.md5(img_data).hexdigest()
        },
        'labels': labels
    }
    return simple_task


def main():
    """This function is the main function of the program. It gets all the tasks
    from mongodb, and then sends them to the ray cluster for processing. It
    then waits for the results to come back, and saves them to a parquet file.

    Returns:
        None
    """
    tasks = get_all_tasks_from_mongodb()

    futures = [simplify.remote(task) for task in tasks]
    results = []
    for future in tqdm(futures):
        try:
            results.append(ray.get(future))
        except (ConnectionResetError, RayTaskError) as e:
            print('ERROR:', e)
            futures.append(future)
            time.sleep(10)

    d = {str(k): v for k, v in enumerate(results) if v}

    df = pd.DataFrame.from_dict(d).T
    ts = datetime.now().strftime('%m-%d-%Y_%H.%M.%S')
    ds_name = f'{ts}.parquet'
    df.to_parquet(ds_name)


if __name__ == '__main__':
    load_dotenv()
    main()

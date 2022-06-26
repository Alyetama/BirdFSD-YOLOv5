#!/usr/bin/env python
# coding: utf-8

import os

import ray
from loguru import logger
from tqdm import tqdm

from birdfsd_yolov5.model_utils.utils import api_request


@ray.remote
def get_pred_details(url):
    return api_request(url)


def get_project_pred_ids(db, project_id, tasks):
    existing_ids = db[f'project_{project_id}_preds'].find().distinct('_id')

    all_pred_ids = []
    for task in tasks:
        for pred_id in task['predictions']:
            if pred_id not in existing_ids:
                all_pred_ids.append(pred_id)
    return all_pred_ids


def process_preds(db, project_id, tasks):
    prediction_ids = get_project_pred_ids(db, project_id, tasks)
    if not prediction_ids:
        logger.debug(f'All predictions in project {project_id} are up-to-date')
        return

    p_url = f'{os.environ["LS_HOST"]}/api/predictions'
    futures = [
        get_pred_details.remote(f'{p_url}/{pred_id}/')
        for pred_id in prediction_ids
    ]
    results = [ray.get(future) for future in tqdm(futures, desc='Futures')]
    for result in results:
        result.update({'_id': result['id']})  # noqa: PyTypeChecker
    col = db[f'project_{project_id}_preds']
    col.drop()
    col.insert_many(results)
    return

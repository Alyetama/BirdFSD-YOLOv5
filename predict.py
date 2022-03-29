#!/usr/bin/env python
# coding: utf-8

import argparse
import json
import os
from pathlib import Path

import numpy as np
import requests
import torch
from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm


def make_headers():
    TOKEN = os.environ['TOKEN']
    headers = requests.structures.CaseInsensitiveDict()
    headers['Content-type'] = 'application/json'
    headers['Authorization'] = f'Token {TOKEN}'
    return headers


def to_srv(url):
    return url.replace(f'{os.environ["LS_HOST"]}/data/local-files/?d=',
                       f'{os.environ["SRV_HOST"]}/')


def get_task(_task_id):
    url = f'{os.environ["LS_HOST"]}/api/tasks/{_task_id}'
    resp = requests.get(url, headers=headers)
    data = resp.json()
    data['data']['image'] = to_srv(data['data']['image'])
    return data


def download_image(img_url):
    cur_img_name = Path(img_url).name
    r = requests.get(img_url)
    with open(f'/tmp/{cur_img_name}', 'wb') as f:
        f.write(r.content)
    img_local_path = f'/tmp/{cur_img_name}'
    logger.debug(img_local_path)
    return img_local_path


def label_score(x):
    _x = x.split('0')
    score = float('0' + _x[-1])
    label = _x[0].rstrip()
    return label, score


def yolo_to_ls(x, y, width, height, score, n):
    global names
    x = (x - width / 2) * 100
    y = (y - height / 2) * 100
    w = width * 100
    h = height * 100
    x, y, w, h, score = [float(i) for i in [x, y, w, h, score]]
    return x, y, w, h, round(score, 2), names[int(n)]


def get_all_tasks():
    logger.debug('Fetching all tasks. This might take few minutes...')
    q = 'exportType=JSON&download_all_tasks=true'
    url = f'{os.environ["LS_HOST"]}/api/projects/1/export?{q}'
    resp = requests.get(url, headers=headers)
    return resp.json()


def selected_tasks(tasks, start, end):
    return [t for t in tasks if t['id'] in range(start, end + 1)]


def pred_result(x, y, w, h, score, label):
    return {
        "type": "rectanglelabels",
        "score": score,
        "value": {
            "x": x,
            "y": y,
            "width": w,
            "height": h,
            "rectanglelabels": [label]
        },
        "to_name": "image",
        "from_name": "label"
    }


def pred_post(results, scores, task_id):
    return {
        'model_version': MODEL_VERSION,  # global
        'result': results,
        'score': np.mean(scores),
        'cluster': 0,
        'neighbors': {},
        'mislabeling': 0,
        'task': task_id
    }


def post_prediction(task, dry_run=False):
    task_id = task['id']
    img = download_image(get_task(task_id)['data']['image'])
    model_preds = model(img)
    pred_xywhn = model_preds.xywhn[0]
    if pred_xywhn.shape[0] == 0:
        logger.debug('No predictions...')
        url = F'{os.environ["LS_HOST"]}/api/tasks/20318'
        resp = requests.delete(url, headers=headers)
        logger.debug({'response': resp.text})
        logger.debug('Deleted image.')
        return

    results = []
    scores = []

    for pred in pred_xywhn:
        result = yolo_to_ls(*pred)
        scores.append(result[-2])
        results.append(pred_result(*result))
        logger.debug(result)

    if not dry_run:
        _post = pred_post(results, scores, task_id)

        url = F'{os.environ["LS_HOST"]}/api/predictions/'
        resp = requests.post(url, headers=headers, data=json.dumps(_post))
        logger.debug({'response': resp.json()})


def opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('-w',
                        '--weights',
                        help='Path to the model weights',
                        type=str,
                        required=True)
    parser.add_argument('-r',
                        '--tasks-range',
                        help='Selected range of tasks (e.g., "10,18")',
                        type=str)
    parser.add_argument('-v',
                        '--model-version',
                        help='Name of the model version',
                        type=str)
    parser.add_argument('-a',
                        '--predict-all',
                        help='Predict all tasks even if predictions exist',
                        action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    load_dotenv()
    logger.add('logs.log')
    args = opts()
    headers = make_headers()

    if not args.model_version:
        MODEL_VERSION = 'BirdFSD-YOLOv5-v1.0.0-unknown'
        logger.warning(
            f'Model version was not specified! Defaulting to {MODEL_VERSION}')
    else:
        MODEL_VERSION = args.model_version

    model = torch.hub.load('ultralytics/yolov5', 'custom', args.weights)
    names = model.names

    tasks = get_all_tasks()

    if not args.predict_all and not args.tasks_range:
        tasks = [t for t in tasks if not t['predictions']]

    if args.tasks_range:
        logger.debug(f'Selected range of tasks: {args.tasks_range}')
        tasks_range = [int(n) for n in args.tasks_range.split(',')]
        tasks = selected_tasks(tasks, *tasks_range)

    logger.debug(f'Tasks to predict: {len(tasks)}')

    for task in tqdm(tasks):
        logger.debug(task)
        post_prediction(task)

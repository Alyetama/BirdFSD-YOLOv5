#!/usr/bin/env python
# coding: utf-8

import argparse
import json
import os
import signal
import sys
from pathlib import Path

import numpy as np
import requests
import torch
from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm


class _Headers:

    def __init__(self, token):
        pass

    def make_headers(self):
        headers = requests.structures.CaseInsensitiveDict()
        headers['Content-type'] = 'application/json'
        headers['Authorization'] = f'Token {os.environ["TOKEN"]}'
        return headers


class LoadModel:

    def __init__(self, weights):
        self.weights = weights

    def model(self):
        return torch.hub.load('ultralytics/yolov5', 'custom', self.weights)


class Predict(LoadModel, _Headers):

    def __init__(self,
                 weights,
                 tasks_range='',
                 predict_all=False,
                 model_version=None,
                 debug=False):
        super().__init__(weights)
        self.headers = super().make_headers()
        self.model = super().model()
        self.tasks_range = tasks_range
        self.predict_all = predict_all
        self.model_version = model_version
        self.debug = debug

    @staticmethod
    def keyboard_interrupt_handler(sig, frame):
        logger.warning(f'KeyboardInterrupt (ID: {sig}) has been caught...')
        sys.exit(1)

    def get_model_version(self):
        if not self.model_version:
            MODEL_VERSION = 'BirdFSD-YOLOv5-v1.0.0-unknown'
            logger.warning(
                f'Model version was not specified! Defaulting to {MODEL_VERSION}'
            )
        else:
            MODEL_VERSION = self.model_version
        return MODEL_VERSION

    @staticmethod
    def to_srv(url):
        return url.replace(f'{os.environ["LS_HOST"]}/data/local-files/?d=',
                           f'{os.environ["SRV_HOST"]}/')

    def get_task(self, _task_id):
        logger.debug(f'Processing task: {_task_id}')
        url = f'{os.environ["LS_HOST"]}/api/tasks/{_task_id}'
        resp = requests.get(url, headers=self.headers)
        data = resp.json()
        try:
            data['data']['image'] = self.to_srv(data['data']['image'])
            return data
        except KeyError as e:
            logger.error(e)
            logger.error(
                f'Could not find image file for task {_task_id}! Skipping...')
            return

    @staticmethod
    def download_image(img_url):
        cur_img_name = Path(img_url).name
        r = requests.get(img_url)
        with open(f'/tmp/{cur_img_name}', 'wb') as f:
            f.write(r.content)
        img_local_path = f'/tmp/{cur_img_name}'
        logger.debug(img_local_path)
        return img_local_path

    def yolo_to_ls(self, x, y, width, height, score, n):
        x = (x - width / 2) * 100
        y = (y - height / 2) * 100
        w = width * 100
        h = height * 100
        x, y, w, h, score = [float(i) for i in [x, y, w, h, score]]
        return x, y, w, h, round(score, 2), self.model.names[int(n)]

    def get_all_tasks(self):
        logger.debug('Fetching all tasks. This might take few minutes...')
        q = 'exportType=JSON&download_all_tasks=true'
        url = f'{os.environ["LS_HOST"]}/api/projects/1/export?{q}'
        if self.debug:
            url = f'{os.environ["LS_HOST"]}/api/tasks/7409'
        resp = requests.get(url, headers=self.headers)
        if self.debug:
            return [resp.json()]
        return resp.json()

    @staticmethod
    def selected_tasks(tasks, start, end):
        return [t for t in tasks if t['id'] in range(start, end + 1)]

    @staticmethod
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

    def pred_post(self, results, scores, task_id):
        return {
            'model_version': self.model_version,
            'result': results,
            'score': np.mean(scores),
            'cluster': 0,
            'neighbors': {},
            'mislabeling': 0,
            'task': task_id
        }

    def post_prediction(self, task, dry_run=False):
        task_id = task['id']
        task_ = self.get_task(task_id)
        if not task_:
            return
        img = self.download_image(task_['data']['image'])
        model_preds = self.model(img)
        pred_xywhn = model_preds.xywhn[0]
        if pred_xywhn.shape[0] == 0:
            logger.debug('No predictions...')
            url = f'{os.environ["LS_HOST"]}/api/tasks/{task_id}'
            resp = requests.delete(url, headers=self.headers)
            logger.debug({'response': resp.text})
            logger.debug(f'Deleted task {task_id}.')
            return

        results = []
        scores = []

        for pred in pred_xywhn:
            result = self.yolo_to_ls(*pred)
            scores.append(result[-2])
            results.append(self.pred_result(*result))
            logger.debug(result)

        if not dry_run:
            _post = self.pred_post(results, scores, task_id)
            logger.debug({'request': _post})
            url = F'{os.environ["LS_HOST"]}/api/predictions/'
            resp = requests.post(url,
                                 headers=self.headers,
                                 data=json.dumps(_post))
            logger.debug({'response': resp.json()})

    def apply_predictions(self):
        logger.add('logs.log')
        signal.signal(signal.SIGINT, self.keyboard_interrupt_handler)

        tasks = self.get_all_tasks()

        if not self.predict_all and not self.tasks_range and not self.debug:
            tasks = [t for t in tasks if not t['predictions']]

        if self.tasks_range:
            logger.info(f'Selected range of tasks: {self.tasks_range}')
            tasks_range = [int(n) for n in self.tasks_range.split(',')]
            tasks = self.selected_tasks(tasks, *tasks_range)

        logger.info(f'Tasks to predict: {len(tasks)}')

        for task in tqdm(tasks):
            self.post_prediction(task)


if __name__ == '__main__':
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument('-w',
                        '--weights',
                        help='Path to the model weights',
                        type=str,
                        required=True)
    parser.add_argument(
        '-r',
        '--tasks-range',
        help=
        'Comma-separated range of tasks by task ID number (e.g., "10,18").',
        type=str)
    parser.add_argument('-a',
                        '--predict-all',
                        help='Predict all tasks even if predictions exist',
                        action='store_true')
    parser.add_argument('-t',
                        '--one-task',
                        help='Predict a single task.',
                        type=int)
    parser.add_argument('-v',
                        '--model-version',
                        help='Name of the model version',
                        type=str)
    parser.add_argument(
        '-d',
        '--debug',
        help='Run in debug mode (will run on one task for debugging)',
        action='store_true')
    args = parser.parse_args()

    predict = Predict(args.weights, args.tasks_range, args.predict_all,
                      args.one_task, args.model_version, args.debug)
    predict.apply_predictions()

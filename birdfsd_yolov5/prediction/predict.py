#!/usr/bin/env python
# coding: utf-8

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional, List, Union

import concurrent.futures
import numpy as np
import ray
import requests
import torch
from PIL import UnidentifiedImageError
from dotenv import load_dotenv
from loguru import logger
from requests.structures import CaseInsensitiveDict
from tqdm import tqdm

from birdfsd_yolov5.model_utils import handlers, mongodb_helper, utils


class _Headers:
    """This class is used to make headers for requests call."""

    def __init__(self) -> None:
        pass

    @staticmethod
    def make_headers() -> CaseInsensitiveDict:
        """Make headers for the request.

        Returns:
            CaseInsensitiveDict: A dictionary of headers.
        """
        load_dotenv()
        headers = CaseInsensitiveDict()
        headers['Content-type'] = 'application/json'
        headers['Authorization'] = f'Token {os.environ["TOKEN"]}'
        return headers


class LoadModel:
    """This class is used to load a model from a given path."""

    def __init__(self, weights: str, model_version: str) -> None:
        self.weights = weights
        self.model_version = model_version

    def model(self) -> torch.nn.Module:
        """Loads a pretrained YOLOv5 model for a given dataset.

        The model is loaded using the torch.hub.load function.

        Returns:
            torch.nn.Module: a YOLOv5 model.
        """
        return torch.hub.load('ultralytics/yolov5',
                              'custom',
                              path=self.weights)


class Predict(LoadModel, _Headers):
    """Prediction and preprocessing class.

    This class is used to predict bounding boxes and classes for images in
    a given project. It uses the YOLOv5 model to predict bounding boxes and
    then posts the predictions to the Label Studio server.

    Attributes:
        headers (dict): Headers for the requests.
        model (torch.nn.Module): YOLOv5 model.
        model_version (str): Model version to use.
        project_ids (Optional[str]): Ids of the projects to predict.
        tasks_range (Optional[str]): Range of tasks to predict.
        predict_all (bool): Predict all tasks in the project(s).
        one_task (Optional[int]): Predict a single task.
        multithreading (bool): Use multithreading.
        delete_if_no_predictions (bool): Delete tasks that have no predictions.
        if_empty_apply_label (str): Apply a label to tasks that have no
            predictions.
        get_tasks_with_api (bool): Get tasks with label-studio API instead of
            MongoDB.
        db (pymongo.database.Database): An instance of mongoDB client (if
            connection string exists in .env).
        flush (list): Used to flush temp files written to disk during
            prediction.
    """

    def __init__(self,
                 weights: str,
                 model_version: str,
                 project_ids: Optional[str] = None,
                 tasks_range: Optional[str] = None,
                 predict_all: bool = False,
                 one_task: Optional[int] = None,
                 multithreading: bool = True,
                 delete_if_no_predictions: bool = True,
                 if_empty_apply_label: str = None,
                 get_tasks_with_api: bool = False,
                 verbose: bool = False) -> None:
        super().__init__(weights, model_version)
        self.headers = super().make_headers()
        self.model = super().model()
        self.project_ids = project_ids
        self.tasks_range = tasks_range
        self.predict_all = predict_all
        self.one_task = one_task
        self.multithreading = multithreading
        self.delete_if_no_predictions = delete_if_no_predictions
        self.if_empty_apply_label = if_empty_apply_label
        self.get_tasks_with_api = get_tasks_with_api
        self.verbose = verbose
        self.db = mongodb_helper.mongodb_db()
        self.flush = []

    def download_image(self, img_url: str) -> str:
        """Download an image from a URL and save it to a local file.

        Args:
          img_url(str): The URL of the image to download.

        Returns:
            str: Path to the temporary image file.
        """
        cur_img_name = Path(img_url.split('?')[0]).name
        r = requests.get(img_url)
        img_local_path = f'/tmp/{cur_img_name}'
        self.flush.append(img_local_path)
        with open(img_local_path, 'wb') as f:
            f.write(r.content)
        logger.debug(img_local_path)
        return img_local_path

    def yolo_to_ls(self, x: float, y: float, width: float, height: float,
                   score: float, n: int) -> tuple:
        """Converts YOLOv5 output to a tuple of the form: (x, y, w, h, conf, n)

        Args:
            x (float): The x coordinate of the center of the bounding box.
            y (float): The y coordinate of the center of the bounding box.
            width (float): The width of the bounding box.
            height (float): The height of the bounding box.
            score (float): The confidence score of the bounding box.
            n (int): The index of the class label.

        Returns:
            tuple: (x, y, width, height, score, label)
        """
        x = (x - width / 2) * 100
        y = (y - height / 2) * 100
        w = width * 100
        h = height * 100
        x, y, w, h, score = [float(i) for i in [x, y, w, h, score]]
        try:
            label = self.model.names[int(n)]
        except ValueError:
            label = n
        return x, y, w, h, round(score, 2), label

    def get_task(self, _task_id: int) -> dict:
        """This function returns a single task from a project.

        Args:
            _task_id (int): The id of the task to be returned.

        Returns:
            dict: A dictionary containing the task data.
        """
        url = f'{os.environ["LS_HOST"]}/api/tasks/{_task_id}'
        resp = requests.get(url, headers=self.headers)
        data = resp.json()
        return data

    def get_all_tasks(self, project_ids: List[str]) -> list:
        """Fetch all tasks from the project.
        
        This function fetches all tasks from the project.

        Args:
          project_ids(str): Comma-seperated string of project ids.
          project_ids: List[str]: 

        Returns:
            list: A list of tasks.
        """
        logger.debug('Fetching all tasks. This might take few minutes...')
        q = 'exportType=JSON&download_all_tasks=true'
        ls_host = os.environ["LS_HOST"]

        all_tasks = []
        for project_id in tqdm(project_ids, desc='Projects'):
            url = f'{ls_host}/api/projects/{project_id}/export?{q}'
            resp = requests.get(url, headers=self.headers)
            all_tasks.append(resp.json())

        return sum(all_tasks, [])

    @staticmethod
    def _get_all_tasks_from_mongodb(project_ids: List[str]) -> list:
        """Fetch all tasks from the project.

        This function fetches all tasks from the project.

        Args:
            project_ids: List[str]: Comma-seperated string of project ids.

        Returns:
            list: A list of tasks.
        """

        @ray.remote
        def _get_all_tasks_from_mongodb_remote(proj_id: str):
            return mongodb_helper.get_tasks_from_mongodb(proj_id)

        futures = []
        for project_id in project_ids:
            futures.append(
                _get_all_tasks_from_mongodb_remote.remote(project_id))
        tasks = []
        for future in tqdm(futures, desc='Projects'):
            tasks.append(ray.get(future))
        return sum(tasks, [])

    @staticmethod
    def selected_tasks(tasks: list, start: int, end: int) -> list:
        """Selects label-studio tasks in a user-specified range

        Creates a list of tasks from the given list of tasks,
        whose id is in the range [start, end].

        Args:
            tasks (list): A list of tasks id.
            start (int): The start of the range.
            end (int): The end of the range.

        Returns:
            list: A list of tasks.
        """
        return [t for t in tasks if t['id'] in range(start, end + 1)]

    def single_task(self, task_id: int) -> list:
        """Get a single task by its id.

        Args:
            task_id (int): The id of the task to get.

        Returns:
            list: A list containing the task data.
        """
        url = f'{os.environ["LS_HOST"]}/api/tasks/{task_id}'
        resp = requests.get(url, headers=self.headers)
        return [resp.json()]

    @staticmethod
    def pred_result(x: float, y: float, w: float, h: float, score: float,
                    label: str) -> dict:
        """Creates a dictionary for an individual prediction result.

        This function takes in the x, y, width, height, score, and label of
        a prediction and returns a dictionary with the prediction's
        information.

        Args:
            x (float): The x coordinate of the prediction.
            y (float): The y coordinate of the prediction.
            w (float): The width of the prediction.
            h (float): The height of the prediction.
            score (float): The confidence score of the prediction.
            label (str): The label of the prediction.

        Returns:
            dict: A dictionary with the prediction's data.
        """
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

    def pred_post(self, results: list, scores: list, task_id: int) -> dict:
        """Creates a dictionary for a prediction POST data.

        This function is used to create an API POST request of a single
        prediction results.

        Args:
            results (list): the results of the model for all predictions in a
                single image.
            scores (list): the scores of all detections predicted by the model.
            task_id (int): the task id.

        Returns:
            dict: The prediction results.
        """
        return {
            'model_version': self.model_version,
            'result': results,
            'score': np.mean(scores),
            'cluster': 0,
            'neighbors': {},
            'mislabeling': 0,
            'task': task_id
        }

    def _preds_exist(self, proj: str) -> list:
        tasks_with_no_preds = []
        preds_col = self.db[f'project_{proj}_preds']
        project_preds = list(preds_col.find({}))
        for x in project_preds:
            if self.model_version != x.get('model_version'):
                tasks_with_no_preds.append(x.get('task'))
        return tasks_with_no_preds

    def post_prediction(self, task: dict) -> Optional[Union[str, dict]]:
        """Prepares the POST request data of the prediction results.

        This function is called by the `predict` method. It takes a task as
        an argument and performs the following steps:
        
        1. It downloads the image from the task's `data` field.
        2. It runs the image through the model and gets the predictions.
        3. It converts the predictions to the format required by Label Studio.
        4. It posts the predictions to Label Studio.
        
        If the task has no data, it skips the task.
        
        If the task has no predictions, it deletes the task if
        `delete_if_no_predictions` is set to `True`.
        
        If `if_empty_apply_label` is set to a label, it applies the string of
        `if_empty_apply_label` if not set to `None`.

        Args:
            task (dict): A dictionary with the task data.

        Returns:
            Optional[Union[str, dict]]: _skipped or task dictionary.

        Raises:
            UnidentifiedImageError: If image is not identified.
            OSError: If image is not downloaded.
            Exception: If any other exception occurs.
        """
        task_id = task['id']
        if not self.delete_if_no_predictions and not self.if_empty_apply_label:
            logger.error(
                'Action for tasks without detections is not specified!')
            logger.error('Available actions: [\'--delete-if-no-predictions\', '
                         '\'--if-empty-apply-label\']')
            sys.exit(1)
        try:
            try:
                img = self.download_image(
                    self.get_task(task_id)['data']['image'])
            except KeyError:
                logger.error(f'Task {task_id} had no data in the response '
                             '(could be a deleted task). Skipping...')
                return
            model_preds = self.model(img)
            pred_xywhn = model_preds.xywhn[0]
            if pred_xywhn.shape[0] == 0:
                logger.debug('No predictions...')
                url = f'{os.environ["LS_HOST"]}/api/tasks/{task_id}'

                if self.delete_if_no_predictions \
                        and not self.if_empty_apply_label:
                    resp = requests.delete(url, headers=self.headers)
                    logger.debug({'response': resp.text})
                    logger.debug(f'Deleted task {task_id}.')
                    return
                elif self.if_empty_apply_label:
                    pred_xywhn = [[
                        .0, .0, .0, .0, .0, self.if_empty_apply_label
                    ]]  # hardcoded zeros

            results = []
            scores = []

            for pred in pred_xywhn:
                result = self.yolo_to_ls(*pred)
                scores.append(result[-2])
                results.append(self.pred_result(*result))
                logger.debug(result)

            _post = self.pred_post(results, scores, task_id)
            if self.verbose:
                logger.debug({'request': _post})
            url = F'{os.environ["LS_HOST"]}/api/predictions/'
            resp = requests.post(url,
                                 headers=self.headers,
                                 data=json.dumps(_post))
            if self.verbose:
                logger.debug({'response': resp.json()})

        except UnidentifiedImageError as _e:
            logger.error(_e)
            logger.error(f'Skipped {task_id}...')
            logger.error(f'Skipped task {task_id}: {task}')
        except OSError as _e:
            logger.error(_e)
            logger.error(f'Temporarily skipped {task_id}...')
            logger.error(f'Skipped task {task_id}: {task}')
            return task
        except Exception as _e:
            logger.error('>>>>>>>>>>>>>>>>>>>>>>>>>> UNEXPECTED EXCEPTION!')
            logger.exception(_e)
            logger.error('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        return

    def apply_predictions(self) -> None:
        """This function applies predictions to label studio tasks."""
        start = time.time()
        logs_file = utils.add_logger(__file__)
        handlers.catch_keyboard_interrupt()

        if self.delete_if_no_predictions and self.if_empty_apply_label:
            logger.error('Can\'t have both --delete-if-no-predictions and '
                         '--if-empty-apply-label!')
            sys.exit(1)

        if self.project_ids:
            project_ids = self.project_ids.split(',')
        else:
            project_ids = utils.get_project_ids_str().split(',')

        if self.one_task:
            tasks = self.single_task(self.one_task)
        else:
            if self.get_tasks_with_api:
                logger.info('Getting tasks with label-studio API...')
                tasks = self.get_all_tasks(project_ids)
            else:
                logger.info('Getting tasks from MongoDB...')
                tasks = self._get_all_tasks_from_mongodb(project_ids)

            if not self.predict_all and not self.tasks_range:
                logger.debug('Predicting tasks with no predictions...')
                tasks = [t for t in tasks if not t['predictions']]

        if self.tasks_range:
            logger.info(f'Selected range of tasks: {self.tasks_range}')
            tasks_range = [int(n) for n in self.tasks_range.split(',')]
            tasks = self.selected_tasks(tasks, *tasks_range)

        with concurrent.futures.ThreadPoolExecutor() as executor:
            tasks_ids = list(
                tqdm(executor.map(self._preds_exist, project_ids),
                     total=len(project_ids),
                     desc='Checking if preds exists'))
            tasks_ids = sum(tasks_ids, [])

        tasks = [task for task in tasks if task['id'] in tasks_ids]

        logger.info(f'Number of tasks to scan: {len(tasks)}')

        if self.multithreading:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                results = list(
                    tqdm(executor.map(self.post_prediction, tasks),
                         total=len(tasks),
                         desc='Prediction'))
        else:
            results = []
            for task in tqdm(tasks):
                results.append(self.post_prediction(task))

        error_tasks = [x for x in results if isinstance(x, dict)]
        logger.debug('Attempting to process temporarily skipped tasks...')
        _results = []
        for task in tqdm(error_tasks):
            result = self.post_prediction(task)
            if isinstance(result, dict):
                logger.error(f'Could not process task {task["id"]}: {task}')
            _results.append(result)

        logger.debug('Flushing temp files...')
        for tmp_file in self.flush:
            try:
                Path(tmp_file).unlink()
            except FileNotFoundError:
                continue

        if tasks:
            num_preds = len([x for x in results + _results if not x])
            task_with_errors = [x for x in _results if isinstance(x, dict)]
            logger.info(f'Made {num_preds} prediction(s)')

            if task_with_errors:
                logger.info(f'Could not process {len(task_with_errors)} '
                            f'task(s) (see logs for details)')

        logger.info(f'Took: {round(time.time() - start, 2)}s')
        utils.upload_logs(logs_file)
        return


def _opts() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Namespace object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-w',
                        '--weights',
                        help='Path to the model weights',
                        type=str)
    parser.add_argument('-v',
                        '--model-version',
                        help='Model version including its name [*-vX.Y.Z*]',
                        type=str,
                        required=True)
    parser.add_argument(
        '-p',
        '--project-ids',
        default=None,
        help='Comma-seperated project ids. If empty, it will select all '
        'projects',
        type=str)
    parser.add_argument(
        '-r',
        '--tasks-range',
        help='Comma-separated range of task ids (e.g., "10,18")',
        type=str,
        default='')
    parser.add_argument('-a',
                        '--predict-all',
                        help='Predict all tasks even if predictions exist',
                        action='store_true')
    parser.add_argument('-t',
                        '--one-task',
                        help='Predict a single task',
                        type=int,
                        default=None)
    parser.add_argument('-m',
                        '--multithreading',
                        help='Enable multithreading',
                        action='store_true')
    parser.add_argument('--get-tasks-with-api',
                        help='Use label-studio API to get tasks data',
                        action='store_true')
    parser.add_argument(
        '-D',
        '--delete-if-no-predictions',
        help='Delete tasks where the model could not predict anything',
        action='store_true')
    parser.add_argument(
        '-L',
        '--if-empty-apply-label',
        help='Label to apply for tasks where the model could not predict '
        'anything',
        type=str,
        default=None)
    parser.add_argument('-d',
                        '--debug',
                        help='Run in debug mode (runs on one task)',
                        action='store_true')
    parser.add_argument('-V',
                        '--verbose',
                        help='Log additional details',
                        action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    load_dotenv()
    args = _opts()

    predict = Predict(weights=args.weights,
                      model_version=args.model_version,
                      project_ids=args.project_ids,
                      tasks_range=args.tasks_range,
                      predict_all=args.predict_all,
                      one_task=args.one_task,
                      multithreading=args.multithreading,
                      delete_if_no_predictions=args.delete_if_no_predictions,
                      if_empty_apply_label=args.if_empty_apply_label,
                      get_tasks_with_api=args.get_tasks_with_api,
                      verbose=args.verbose)

    predict.apply_predictions()

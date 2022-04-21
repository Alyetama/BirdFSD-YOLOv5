#!/usr/bin/env python
# coding: utf-8

import argparse
import concurrent.futures
import json
import os
import signal
import sys
from pathlib import Path
from typing import Union

import numpy as np
import requests
from PIL import UnidentifiedImageError
from dotenv import load_dotenv
from tqdm import tqdm

from model_utils.mongodb_helper import get_tasks_from_mongodb

try:
    import torch
except RuntimeError as e:
    sys.exit(1)

if 'google.colab' in sys.modules:
    sys.path.insert(0, '/content/BirdFSD-YOLOv5/utils')
    from colab_logger import logger  # noqa
else:
    from loguru import logger


def keyboard_interrupt_handler(sig, _):
    """This function handles the KeyboardInterrupt (CTRL+C) signal.
    It's a handler for the signal, which means it's called when the OS sends the signal.
    The signal is sent when the user presses CTRL+C.

    Parameters
    ----------
    The function takes two arguments:
    sig:
        The ID of the signal that was sent.
    frame:
        The current stack frame.
    """
    logger.warning(f'KeyboardInterrupt (ID: {sig}) has been caught...')
    logger.info('Terminating the session gracefully...')
    sys.exit(1)


class _Headers:
    """This class is used to make headers for the requests.
    It is a static class.
    It has only one method: make_headers()
    It returns a dictionary with the headers.
    """

    def __init__(self):
        pass

    @staticmethod
    def make_headers():
        """Make headers for the request.

        Returns:
            headers (dict): A dictionary of headers.
        """
        load_dotenv()
        headers = requests.structures.CaseInsensitiveDict()  # noqa
        headers['Content-type'] = 'application/json'
        headers['Authorization'] = f'Token {os.environ["TOKEN"]}'
        return headers


class LoadModel:
    """This class is used to load a model from a given path.
    The model is a YOLOv5 model.
    The model is loaded using the torch.hub.load function.
    The model is loaded with the following parameters:
        - ultralytics/yolov5
        - custom
        - path=self.weights
    The model is returned by the model method.
    """

    def __init__(self, weights: str):
        self.weights = weights

    def model(self):
        """Loads a pretrained YOLOv5 model for a given dataset.

        Returns:
            (nn.Module): a YOLOv5 model.
        """
        return torch.hub.load('ultralytics/yolov5',
                              'custom',
                              path=self.weights)


class Predict(LoadModel, _Headers):
    """This class is used to predict bounding boxes for images in a given project.
    It uses the YOLOv5 model to predict bounding boxes and then posts the
    predictions to the Label Studio server.

    Parameters
    ----------
    weights : str
        Path to the weights file.
    project_id : int
        ID of the project to predict.
    tasks_range : str, optional
        Range of tasks to predict.
        Example: '1,10' will predict tasks 1 to 10.
        Default: ''
    predict_all : bool, optional
        Predict all tasks in the project.
        Default: False
    one_task : Union[None, int], optional
        Predict a single task.
        Default: None
    model_version : Union[None, str], optional
        Model version to use.
        Default: None
    multithreading : bool, optional
        Use multithreading.
        Default: True
    delete_if_no_predictions : bool, optional
        Delete tasks that have no predictions.
        Default: True
    if_empty_apply_label : str, optional
        Apply a label to tasks that have no predictions.
        Default: None
    debug : bool, optional
        Debug mode.
        Default: False

    Attributes
    ----------
    headers : dict
        Headers for the requests.
    model : YOLOv5
        YOLOv5 model.
    project_id : int
        ID of the project to predict.
    tasks_range : str
        Range of tasks to predict.
    predict_all : bool
        Predict all tasks in the project.
    one_task : Union[None, int]
        Predict a single task.
    model_version : Union[None, str]
        Model version to use.
    multithreading : bool
        Use multithreading.
    delete_if_no_predictions : bool
        Delete tasks that have no predictions.
    if_empty_apply_label : str
        Apply a label to tasks that have no predictions.
    debug : bool
        Debug mode.
    counter : int
        Counter for the progress bar.
    total_tasks : Union[None, int]
        Total number of tasks that have been predicted.

    Methods
    -------
    get_model_version()
        Get the model version.
    get_task(task_id)
        Get a task from the Label Studio server.
    download_image(img_url)
        Download an image from the Label Studio server.
    yolo_to_ls(x, y, width, height, score, n)
        Convert YOLOv5 predictions to Label Studio format.
    get_all_tasks()
        Get all tasks from the Label Studio server.
    selected_tasks(tasks, start, end)
        Select a range of tasks.
    single_task(task_id)
        Get a single task from the Label Studio server.
    pred_result(x, y, w, h, score, label)
        Create a prediction result.
    pred_post(results, scores, task_id)
        Create a prediction post.
    post_prediction(task)
        Post a prediction to the Label Studio server.
    apply_predictions()
        Apply predictions to the Label Studio server.
    """

    def __init__(self,
                 weights: str,
                 project_id: int,
                 tasks_range: str = '',
                 predict_all: bool = False,
                 one_task: Union[None, int] = None,
                 model_version: Union[None, str] = None,
                 multithreading: bool = True,
                 delete_if_no_predictions: bool = True,
                 if_empty_apply_label: str = None,
                 debug: bool = False):
        super().__init__(weights)
        self.headers = super().make_headers()
        self.model = super().model()
        self.project_id = project_id
        self.tasks_range = tasks_range
        self.predict_all = predict_all
        self.one_task = one_task
        self.model_version = model_version
        self.multithreading = multithreading
        self.delete_if_no_predictions = delete_if_no_predictions
        self.if_empty_apply_label = if_empty_apply_label
        self.debug = debug
        self.counter = 0
        self.total_tasks = None

    def get_model_version(self):
        """Get the model version.
    
        If the model version is not specified, a default value is used.
        
        Parameters
        ----------
        self : object
            The object instance.
        
        Returns
        -------
        model_version : str
            The model version.
        """
        if not self.model_version:
            model_version = 'BirdFSD-YOLOv5-v1.0.0-unknown'
            logger.warning(f'Model version was not specified! Defaulting to '
                           f'{model_version}')
        else:
            model_version = self.model_version
        return model_version

    @staticmethod
    def to_srv(url):
        """Converts a private file URL to a public server URL.
    
        Parameters
        ----------
        url : str
            The label-studio URL to convert.
        
        Returns
        -------
        str
            The server URL.
        """
        return url.replace(f'{os.environ["LS_HOST"]}/data/local-files/?d=',
                           f'{os.environ["SRV_HOST"]}/')

    def get_task(self, _task_id: int) -> dict:
        """This function returns a task from the local server.
        
        Parameters
        ----------
        _task_id : int
            The id of the task to be returned.
        
        Returns
        -------
        dict
            A dictionary containing the task data.
        
        Examples
        --------
        >>> get_task(1)
        {'id': 1, 'data': {'image': 'http://localhost:8000/data/local-files/1.jpg'}}
        """
        url = f'{os.environ["LS_HOST"]}/api/tasks/{_task_id}'
        resp = requests.get(url, headers=self.headers)
        data = resp.json()
        data['data']['image'] = self.to_srv(data['data']['image'])
        return data

    @staticmethod
    def download_image(img_url):
        """Download an image from a URL and save it to a local file.
        
        Parameters
        ----------
        img_url : str
            The URL of the image to download.
        
        Returns
        -------
        img_local_path : str
            The path to the local file containing the downloaded image.
        """
        cur_img_name = Path(img_url).name
        r = requests.get(img_url)
        with open(f'/tmp/{cur_img_name}', 'wb') as f:
            f.write(r.content)
        img_local_path = f'/tmp/{cur_img_name}'
        logger.debug(img_local_path)
        return img_local_path

    def yolo_to_ls(self, x, y, width, height, score, n):
        """Converts YOLOv5 output to a tuple of the form:
        [x, y, width, height, score, label]

        Parameters
        ----------
        x : float
            The x coordinate of the center of the bounding box.
        y : float
            The y coordinate of the center of the bounding box.
        width : float
            The width of the bounding box.
        height : float
            The height of the bounding box.
        score : float
            The confidence score of the bounding box.
        n : int
            The index of the class label.

        Returns
        -------
        list
            A tuple of the form:
            [x, y, width, height, score, label]
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

    def get_all_tasks(self):
        """Fetch all tasks from the project.

        This function fetches all tasks from the project.

        Parameters
        ----------
        self : object
            The object instance.

        Returns
        -------
        list
            A list of tasks.
        """
        logger.debug('Fetching all tasks. This might take few minutes...')
        q = 'exportType=JSON&download_all_tasks=true'
        ls_host = os.environ["LS_HOST"]
        url = f'{ls_host}/api/projects/{self.project_id}/export?{q}'
        if self.debug:
            url = f'{os.environ["LS_HOST"]}/api/tasks/7409'  # hardcoded task ID
        resp = requests.get(url, headers=self.headers)
        if self.debug:
            return [resp.json()]
        return resp.json()

    @staticmethod
    def selected_tasks(tasks, start, end):
        """Returns a list of tasks from the given list of tasks,
        whose id is in the range [start, end].

        Parameters
        ----------
        tasks : list
            A list of tasks.
        start : int
            The start of the range.
        end : int
            The end of the range.

        Returns
        -------
        list
            A list of tasks.
        """
        return [t for t in tasks if t['id'] in range(start, end + 1)]

    def single_task(self, task_id):
        """Get a single task by id.

        :param task_id: The id of the task to get.
        :type task_id: int
        :return: A list containing the task.
        :rtype: list
        """
        url = f'{os.environ["LS_HOST"]}/api/tasks/{task_id}'
        resp = requests.get(url, headers=self.headers)
        return [resp.json()]

    @staticmethod
    def pred_result(x, y, w, h, score, label):
        """This function takes in the x, y, width, height, score, and label of a prediction and returns a dictionary with the prediction's information.

        Parameters
        ----------
        x : int
            The x coordinate of the prediction.
        y : int
            The y coordinate of the prediction.
        w : int
            The width of the prediction.
        h : int
            The height of the prediction.
        score : float
            The confidence score of the prediction.
        label : str
            The label of the prediction.

        Returns
        -------
        dict
            A dictionary with the prediction's information.
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

    def pred_post(self, results, scores, task_id):
        """This function is used to create an API POST request of a single prediction results.

        Parameters
        ----------
        results: list
            the results of the model for all predictions in a single image.
        scores: list
            the scores of all detections predicted by the model.
        task_id: int
            the task id.

        Returns
        -------
        dict
            The prediction results.
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

    def post_prediction(self, task):
        """This function is called by the `predict` method. It takes a task as an argument and performs the following steps:

        1. It downloads the image from the task's `data` field.
        2. It runs the image through the model and gets the predictions.
        3. It converts the predictions to the format required by Label Studio.
        4. It posts the predictions to Label Studio.

        If the task has no data, it skips the task.

        If the task has no predictions, it deletes the task if `delete_if_no_predictions` is set to `True`.

        If `if_empty_apply_label` is set to a label, it applies a the string of `if_empty_apply_label` if not set to `None`.

        Parameters
        ----------
        task: dict
            The labe-studio API response of a single task.

        Returns
        -------
        dict
            A dictionary with the prediction's information.
        """
        try:
            task_id = task['id']
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
                        0.5, 0.5, 0.05, 0.05, 0, self.if_empty_apply_label
                    ]]  # hardcoded arbiturary x, y, w, h values

            results = []
            scores = []

            for pred in pred_xywhn:
                result = self.yolo_to_ls(*pred)
                scores.append(result[-2])
                results.append(self.pred_result(*result))
                logger.debug(result)

            _post = self.pred_post(results, scores, task_id)
            logger.debug({'request': _post})
            url = F'{os.environ["LS_HOST"]}/api/predictions/'
            resp = requests.post(url,
                                 headers=self.headers,
                                 data=json.dumps(_post))
            logger.debug({'response': resp.json()})

        except UnidentifiedImageError as _e:
            logger.error(_e)
            logger.error(f'Skipped {task}')
        except Exception as _e:
            logger.error('>>>>>>>>>>>>>>>>>>>>>>>>>> UNEXPECTED EXCEPTION!')
            logger.exception(_e)
            logger.error('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')
        finally:
            if self.multithreading:
                self.counter += 1
                logger.info(
                    f'🏃‍♂️ Progress: {self.counter} / {self.total_tasks} 🟢')

    def apply_predictions(self):
        """This function applies predictions to tasks.

        Returns
        -------
        list
            A list of tasks with predictions applied.
        """
        if self.delete_if_no_predictions and self.if_empty_apply_label:
            logger.error('Can\'t have both --delete-if-no-predictions and '
                         '--if-empty-apply-label!')
            sys.exit(1)
        if self.one_task:
            tasks = self.single_task(self.one_task)
        else:
            if not Path('tasks.json').exists():
                tasks = self.get_all_tasks()
            else:
                logger.debug('Loading tasks from a local file...')
                with open('tasks.json') as j:
                    tasks = json.load(j)

            if not self.predict_all and not self.tasks_range and not self.debug:
                tasks = [t for t in tasks if not t['predictions']]

        if self.tasks_range:
            logger.info(f'Selected range of tasks: {self.tasks_range}')
            tasks_range = [int(n) for n in self.tasks_range.split(',')]
            tasks = self.selected_tasks(tasks, *tasks_range)

        if not Path('tasks.json').exists():
            logger.debug('Writing tasks to a file...')
            with open('tasks.json', 'w') as j:
                json.dump(tasks, j)

        logger.info(f'Tasks to predict: {len(tasks)}')
        self.total_tasks = len(tasks)

        if self.multithreading:
            with concurrent.futures.ThreadPoolExecutor() as executor:
                futures = []
                results = [
                    executor.submit(self.post_prediction, x) for x in tasks
                ]
                for future in concurrent.futures.as_completed(results):
                    futures.append(future.result())

        else:
            for task in tqdm(tasks):
                self.post_prediction(task)


if __name__ == '__main__':
    load_dotenv()
    logger.add('logs.log')

    parser = argparse.ArgumentParser()
    parser.add_argument('-w',
                        '--weights',
                        help='Path to the model weights',
                        type=str,
                        required=True)
    parser.add_argument('-p',
                        '--project_id',
                        help='Label-studio project ID',
                        type=int,
                        required=True)
    parser.add_argument(
        '-r',
        '--tasks-range',
        help='Comma-separated range of task IDs (e.g., "10,18")',
        type=str,
        default='')
    parser.add_argument('-a',
                        '--predict-all',
                        help='Predict all tasks even if predictions exist',
                        action='store_true')
    parser.add_argument('-t',
                        '--one-task',
                        help='Predict a single task',
                        type=Union[None, int],
                        default=None)
    parser.add_argument('-v',
                        '--model-version',
                        help='Name of the model version',
                        type=Union[None, str],
                        default=None)
    parser.add_argument('-m',
                        '--multithreading',
                        help='Enable multithreading',
                        action='store_true')
    parser.add_argument('-M',
                        '--mongodb',
                        help='Get tasks from MongoDB',
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
        type=Union[None, str],
        default=None)
    parser.add_argument('-d',
                        '--debug',
                        help='Run in debug mode (runs on one task)',
                        action='store_true')
    args = parser.parse_args()

    signal.signal(signal.SIGINT, keyboard_interrupt_handler)

    if args.mongodb:
        logger.info('Getting tasks from MongoDB...')
        get_tasks_from_mongodb(args.project_id)

    predict = Predict(args.weights, args.project_id, args.tasks_range,
                      args.predict_all, args.one_task, args.model_version,
                      args.multithreading, args.delete_if_no_predictions,
                      args.if_empty_apply_label, args.debug)
    predict.apply_predictions()

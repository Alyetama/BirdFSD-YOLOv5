#!/usr/bin/env python
# coding: utf-8

import functools
import json
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional, Union

import numpy as np
import ray
import requests
import sys
from loguru import logger
from minio.error import S3Error
from requests.structures import CaseInsensitiveDict
from tqdm.auto import tqdm

from birdfsd_yolov5.model_utils import handlers, s3_helper, mongodb_helper


def add_logger(current_file: str) -> str:
    """Adds a logger to the current file.

    Args:
        current_file (str): The name of the current file.

    Returns:
        str: The name of the log file.

    """
    logger.remove()
    Path('logs').mkdir(exist_ok=True)
    ts = datetime.now().strftime('%m-%d-%Y_%H.%M.%S')
    logs_file = f'logs/{Path(current_file).stem}_{ts}.log'
    logger.add(
        sys.stderr,
        format='{level.icon} <fg #3bd6c6>{time:HH:mm:ss}</fg #3bd6c6> | '
        '<level>{level: <8}</level> | '
        '<fg #f1fa8c>{function}</fg #f1fa8c>:'
        '<fg #f1fa8c>{line}</fg #f1fa8c> - <lvl>{message}</lvl>',
        level='DEBUG')
    logger.level('WARNING', color='<yellow><bold>', icon='🚧')
    logger.level('INFO', color='<bold>', icon='🚀')
    logger.add(logs_file)
    return logs_file


def upload_logs(logs_file: str) -> None:
    """Uploads the logs file to S3.

    Args:
        logs_file: The path to the logs file.

    Returns:
        None

    Raises:
        S3Error: If the upload fails.

    """
    s3 = s3_helper.S3()
    try:
        logger.debug('Uploading logs...')
        resp = s3.upload('logs', logs_file)
        logger.debug(f'Uploaded log file: `{resp.object_name}`')
    except S3Error as e:
        logger.error('Could not upload logs file!')
        logger.error(e)
    return


def requests_download(url: str, filename: str) -> None:
    """Downloads a file from a given URL to a given filename.
    
    Notes:
        tqdm integration snippet from: https://stackoverflow.com/a/63831344

    Args:
        url (str): The URL of the file to download.
        filename (str): The name of the file to save the downloaded file to.

    Returns:
        None

    Raises:
        RuntimeError: If the returned status code is not 200.

    """

    handlers.catch_keyboard_interrupt()
    r = requests.get(url, stream=True, allow_redirects=True)
    if r.status_code != 200:
        r.raise_for_status()
        raise RuntimeError(f'Returned status code: {r.status_code}')
    file_size = int(r.headers.get('Content-Length', 0))
    r.raw.read = functools.partial(r.raw.read, decode_content=True)
    with tqdm.wrapattr(r.raw,
                       'read',
                       total=file_size,
                       desc='Download progress') as r_raw:
        with open(filename, 'wb') as f:
            shutil.copyfileobj(r_raw, f)
    return


def api_request(url: str,
                method: str = 'get',
                data: Optional[dict] = None,
                return_text: bool = False) -> Union[dict, list, str]:
    """Makes an API request to the given url with the given method and data.

    Args:
        url (str): The url to make the request to.
        method (str): The HTTP method to use. Defaults to 'get'.
        data (Optional[dict]): The data to send with the request. Defaults to 
            None.
        return_text (bool): Return the response as literal string.

    Returns:
        The response from the API.

    """
    headers = CaseInsensitiveDict()
    headers['Content-type'] = 'application/json'
    headers['Authorization'] = f'Token {os.environ["TOKEN"]}'
    method = method.lower()
    if method == 'post':
        resp = requests.post(url, headers=headers, data=json.dumps(data))
    elif method == 'patch':
        resp = requests.patch(url, headers=headers, data=json.dumps(data))
    elif method == 'delete':
        resp = requests.delete(url, headers=headers)
    else:
        resp = requests.get(url, headers=headers)

    if return_text or method == 'delete':
        return resp.text
    return resp.json()


def get_project_ids_str(exclude_ids: Optional[str] = None) -> str:
    """Get a comma separated string of project ids.

    Args:
        exclude_ids (Optional[str]): A comma separated string of project ids to 
            exclude.

    Returns:
        str: A comma separated string of project ids.

    """
    projects = api_request(
        f'{os.environ["LS_HOST"]}/api/projects?page_size=1000')
    projects_results: list = projects['results']
    project_ids = sorted([project['id'] for project in projects_results])
    project_ids = [str(p) for p in project_ids]
    if exclude_ids:
        exclude_ids = list(exclude_ids.split(','))
        project_ids = [p for p in project_ids if p not in exclude_ids]
    return ','.join(project_ids)


def get_data(json_min: bool) -> list:
    """Get all tasks data from all projects.

    Args:
        json_min: Whether to download the tasks in JSON_MIN format.

    Returns:
        list: A list of all tasks.

    """

    @ray.remote
    def iter_db(proj_id: str, j_min: bool) -> list:
        return mongodb_helper.get_tasks_from_mongodb(proj_id,
                                                     dump=False,
                                                     json_min=j_min)

    project_ids = get_project_ids_str().split(',')
    futures = []
    for project_id in project_ids:
        futures.append(iter_db.remote(project_id, json_min))
    tasks = []
    for future in tqdm(futures):
        tasks.append(ray.get(future))
    return sum(tasks, [])


def tasks_data(output_path: Optional[str]) -> None:
    """Get all tasks data of all projects.

    Args:
        output_path (str): Path to the output JSON file.

    Returns:
        None

    """
    with open(output_path, 'w') as j:
        json.dump(get_data(False), j)
    return


def get_labels_count() -> dict:
    """Creates a dictionary of labels and their frequency.

    Returns:
       dict:  A dictionary of labels and their frequency.
        
    """
    tasks = get_data(True)
    labels = []
    for d in tasks:
        if d.get('label'):
            for label in d['label']:
                labels.append(label['rectanglelabels'])
    unique, counts = np.unique(labels, return_counts=True)
    labels_freq = {k: int(v) for k, v in np.asarray((unique, counts)).T}
    return labels_freq

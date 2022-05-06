#!/usr/bin/env python
# coding: utf-8

import functools
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import ray
import requests
from loguru import logger
from minio.error import S3Error
from requests.structures import CaseInsensitiveDict
from tqdm.auto import tqdm

try:
    from . import handlers, s3_helper, mongodb_helper
except ImportError:
    import handlers
    import s3_helper
    import mongodb_helper


def add_logger(current_file: str) -> str:
    Path('logs').mkdir(exist_ok=True)
    ts = datetime.now().strftime('%m-%d-%Y_%H.%M.%S')
    logs_file = f'logs/{Path(current_file).stem}_{ts}.log'
    logger.add(logs_file)
    return logs_file


def upload_logs(logs_file: str) -> None:
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
    """https://stackoverflow.com/a/63831344"""
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


def api_request(url: str, method: str = 'get', data: dict = None) -> dict:
    headers = CaseInsensitiveDict()
    headers['Content-type'] = 'application/json'
    headers['Authorization'] = f'Token {os.environ["TOKEN"]}'
    if method == 'get':
        resp = requests.get(url, headers=headers)
    elif method == 'post':
        resp = requests.post(url, headers=headers, data=json.dumps(data))
    elif method == 'patch':
        resp = requests.patch(url, headers=headers, data=json.dumps(data))
    return resp.json()  # noqa


def get_project_ids(exclude_ids: str = None) -> str:
    projects = api_request(
        f'{os.environ["LS_HOST"]}/api/projects?page_size=10000')
    project_ids = sorted([project['id'] for project in projects['results']])
    project_ids = [str(p) for p in project_ids]
    if exclude_ids:
        exclude_ids = [p for p in exclude_ids.split(',')]
        project_ids = [p for p in project_ids if p not in exclude_ids]
    return ','.join(project_ids)


def get_data(json_min):

    @ray.remote
    def iter_db(proj_id, j_min):
        return mongodb_helper.get_tasks_from_mongodb(proj_id,
                                                     dump=False,
                                                     json_min=j_min)

    project_ids = get_project_ids().split(',')
    futures = []
    for project_id in project_ids:
        futures.append(iter_db.remote(project_id, json_min))
    tasks = []
    for future in tqdm(futures):
        tasks.append(ray.get(future))
    return sum(tasks, [])


def tasks_data(output_path):
    with open(output_path, 'w') as j:
        json.dump(get_data(False), j)
    return


def get_labels_count():
    tasks = get_data(True)
    labels = []
    for d in tasks:
        if d.get('label'):
            for label in d['label']:
                labels.append(label['rectanglelabels'])
    unique, counts = np.unique(labels, return_counts=True)
    labels_freq = {k: int(v) for k, v in np.asarray((unique, counts)).T}
    return labels_freq

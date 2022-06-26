#!/usr/bin/env python
# coding: utf-8

import copy
import json
import os
import random
import shlex
import subprocess
import sys
import time
from datetime import date
from datetime import datetime
from glob import glob
from pathlib import Path

import matplotlib
import requests
import schedule
from dotenv import load_dotenv
from loguru import logger
from requests.structures import CaseInsensitiveDict

from birdfsd_yolov5.model_utils.handlers import catch_keyboard_interrupt
from birdfsd_yolov5.model_utils.utils import add_logger, upload_logs


def _run(cmd):
    p = subprocess.run(shlex.split(cmd),
                       shell=False,
                       check=True,
                       capture_output=True,
                       text=True)
    logger.info(f'Process exit code: {p.returncode}')
    logger.debug(f'stdout: {p.stdout}')
    logger.debug(f'stderr: {p.stderr}')
    return p.stdout


def make_headers():
    headers = CaseInsensitiveDict()
    headers['Content-type'] = 'application/json'
    headers['Authorization'] = f'Token {os.environ["TOKEN"]}'
    return headers


def handle_project():
    headers = make_headers()
    url = f'{os.environ["LS_HOST"]}/api/projects'
    resp = requests.get(url, headers=headers)
    response = resp.json()
    logger.debug(response)

    projects = []
    for p in response['results']:
        if 'project' in p['title']:
            projects.append((p['id'], p['title'], p['task_number']))

    cmd_size = f'rclone {os.environ["IS_SHARED"]} size ' \
               f'{os.environ["REMOTE_PATH"]}'
    num_of_new_files = int(
        _run(cmd_size).split('Total objects:')[1].split(' (')[1].split(')')[0])
    logger.debug(f'num_of_new_files: {num_of_new_files}')

    projects = sorted(projects)[::-1]

    last_project = projects[0]
    size_if_added = last_project[-1] + num_of_new_files
    logger.debug(f'size_if_added: {size_if_added}')

    if size_if_added > 1000:
        logger.debug('Creating new project...')
        template = copy.deepcopy(
            [x for x in response['results'] if x['id'] == projects[0][0]][0])
        template = {
            k: v
            for k, v in template.items() if k in ['title', 'label_config']
        }
        new_project_title = 'project-' + str(int(template['title'][-3:]) +
                                             1).zfill(3)
        color = random.choice([
            x for x in list(matplotlib.colors.cnames.values())
            if x != '#FFFFFF'
        ])
        template.update({'title': new_project_title, 'color': color})

        url = f'{os.environ["LS_HOST"]}/api/projects'
        resp = requests.post(url, headers=headers, data=json.dumps(template))
        _response = resp.json()
        logger.debug(_response)
        proj_id_to_use = _response['id']
    else:
        proj_id_to_use = projects[0][0]

    return proj_id_to_use


def sync_project(project_id):
    headers = make_headers()
    dt = datetime.today().strftime('%m-%d-%Y')
    NEW_FOLDER_NAME = f'downloaded_{dt}'

    if not Path(f'{os.environ["PATH_TO_SRC_DIR"]}/{NEW_FOLDER_NAME}').exists():
        logger.error(
            f'{os.environ["PATH_TO_SRC_DIR"]}/{NEW_FOLDER_NAME} does not '
            'exist!')
        raise FileNotFoundError

    url = f'{os.environ["LS_HOST"]}/api/storages/localfiles?' \
          f'project={project_id}'
    logger.debug(f'Request: {url}')
    resp = requests.get(url, headers=headers)
    response = resp.json()
    logger.debug(response)
    if isinstance(response, dict):
        if response.get('status_code') == 404:
            raise ConnectionError

    _PATH = f'{os.environ["PATH_TO_SRC_DIR_ON_CONTAINER"]}/{NEW_FOLDER_NAME}'
    logger.debug(_PATH)
    EXISTS = False

    for x in response:
        if x['path'] == _PATH:
            logger.warning('Storage folder already exists!')
            logger.debug(f'Existing storage: {x}')
            EXISTS = True
            break

    if not EXISTS:
        data = "{" + f'"path":"{_PATH}","title":"{NEW_FOLDER_NAME}",' \
                     f'"use_blob_urls":"true","project":{project_id}' + "}"
        resp = requests.post(url, headers=headers, data=data)
        response = resp.json()
        logger.debug(f'Request URL: {url}')
        logger.debug(f'Request data: {data}')
        if response.get('status_code') == 400:
            raise Exception('Something is wrong.')
        logger.info(f'Create new local storage response: {response}')
        storage_id = response['id']
    else:
        storage_id = x['id']  # noqa: PyTypeChecker

    logger.debug('Running sync...')
    url = f'{os.environ["LS_HOST"]}/api/storages/localfiles/{storage_id}/sync'
    resp = requests.post(url, headers=headers)
    logger.info(f'Sync response: {resp.text}')


def rclone_files_handler(project_id):
    ts = f'downloaded_{date.today().strftime("%m-%d-%Y")}'
    source_path = f'{os.environ["PATH_TO_SRC_DIR"]}/{ts}'
    Path(source_path).mkdir(exist_ok=True)

    logger.debug('Copying images from google drive to local storage')
    cmd_copy = f'rclone copy {os.environ["IS_SHARED"]} ' \
               f'{os.environ["REMOTE_PATH"]} {source_path} -P ' \
               f'--stats-one-line --ignore-existing --transfers 32'
    _run(cmd_copy)

    imgs = glob(f'{source_path}/*.jpg')
    logger.info(f'Copied {len(imgs)} image(s).')

    for _ in range(2):
        sync_project(project_id)
        logger.debug('Running again just in case...')
        time.sleep(2)

    logger.debug('Creating -downloaded directory in remote')
    cmd_mkdir = f'rclone {os.environ["IS_SHARED"]} mkdir ' \
                f'{os.environ["REMOTE_PATH"]}-downloaded/{ts}'
    _run(cmd_mkdir)

    logger.debug('Moving images between remote folders')
    cmd_move = f'rclone {os.environ["IS_SHARED"]} move ' \
               f'{os.environ["REMOTE_PATH"]} {os.environ["REMOTE_PATH"]} ' \
               f'-downloaded/{ts} -P --stats-one-line --transfers 32'
    _run(cmd_move)


def sync_local_storage():
    logs_file = add_logger(__file__)
    catch_keyboard_interrupt()

    logger.debug('--------------------START--------------------')
    proj_id_to_use = handle_project()
    rclone_files_handler(proj_id_to_use)
    logger.debug('--------------------END--------------------')

    upload_logs(logs_file)


if __name__ == '__main__':
    load_dotenv()

    if '--once' in sys.argv:
        sync_local_storage()
        sys.exit(0)

    schedule.every().day.do(sync_local_storage)

    while True:
        schedule.run_pending()
        time.sleep(1)

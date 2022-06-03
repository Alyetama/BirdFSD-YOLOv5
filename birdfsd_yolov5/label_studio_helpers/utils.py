#!/usr/bin/env python
# coding: utf-8

import json
import os
from typing import Optional, Union

import ray
from loguru import logger
from tqdm import tqdm

from birdfsd_yolov5.model_utils.handlers import catch_keyboard_interrupt
from birdfsd_yolov5.model_utils.mongodb_helper import (get_tasks_from_mongodb,
                                                       mongodb_db)
from birdfsd_yolov5.model_utils.utils import get_project_ids_str, api_request


def update_model_version_in_all_projects(new_model_version: str) -> None:
    """Updates the selected default model version in all projects.

    Args:
        new_model_version (str): The new model version to be updated.

    Returns:
        None

    """
    project_ids = get_project_ids_str().split(',')

    for project_id in tqdm(project_ids):
        project = api_request(
            f'{os.environ["LS_HOST"]}/api/projects/{project_id}')
        project.update({'model_version': new_model_version})
        project.pop('created_by')

        patched_project = api_request(
            f'{os.environ["LS_HOST"]}/api/projects/{project_id}',
            method='patch',
            data=project)
        if patched_project.get('status_code'):
            logger.error(patched_project)
    return


def get_all_projects_tasks(dump: Optional[Union[bool, str]] = None,
                           get_predictions_instead: bool = False):
    """Get all the tasks from all projects from the database.

    The function gets all the tasks from the MongoDB database. returns a
    list of dictionaries, each dictionary containing the project
    name and the tasks associated with it.

    Args:
        dump (Optional[Union[bool, str]]): The JSON file containing the
            database dump.
        get_predictions_instead (bool): If True, the function returns the
            predictions instead of the actual tasks.

    Returns:
        list: A list of dictionaries, each dictionary containing the project
            name and the tasks associated with it.

    """

    @ray.remote
    def _iter_projects(proj_id, get_preds_instead=get_predictions_instead):
        if get_preds_instead:
            _tasks = get_tasks_from_mongodb(proj_id,
                                            dump=dump,
                                            get_predictions=True)
        else:
            _tasks = get_tasks_from_mongodb(proj_id)
        for task in _tasks:
            task.pop('_id')
        return _tasks

    project_ids = get_project_ids_str().split(',')

    futures = []
    for project_id in project_ids:
        futures.append(_iter_projects.remote(project_id))

    tasks = []
    for future in tqdm(futures):
        tasks.append(ray.get(future))

    if dump:
        with open(dump, 'w') as j:
            json.dump(sum(tasks, []), j)

    return sum(tasks, [])


def _drop_all_projects_from_mongodb():
    catch_keyboard_interrupt()
    CONFIRMED = False
    q = input('Are you sure (y/N)? ')
    if q.lower() in ['y', 'yes']:
        confirm = input('Confirm by typing: "I confirm": ')
        if confirm == 'I confirm':
            CONFIRMED = True
    if not CONFIRMED:
        logger.warning('Cancelled...')
        return

    project_ids = get_project_ids_str().split(',')

    for project_id in tqdm(project_ids):
        db = mongodb_db()
        for name in ['', '_min', '_preds']:
            col = db[f'project_{project_id}{name}']
            col.drop()
    logger.info('Dropped all projects from MongoDB.')

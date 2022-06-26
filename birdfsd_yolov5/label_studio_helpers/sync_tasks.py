#!/usr/bin/env python
# coding: utf-8

import argparse
import os
from typing import Union

import numpy as np
import ray
from dotenv import load_dotenv
from loguru import logger
from tqdm import tqdm

from birdfsd_yolov5.label_studio_helpers.utils import get_all_projects_tasks
from birdfsd_yolov5.model_utils.handlers import catch_keyboard_interrupt
from birdfsd_yolov5.model_utils.mongodb_helper import mongodb_db
from birdfsd_yolov5.model_utils.utils import api_request, get_project_ids_str


@ray.remote
def _insert_many_chunks(chunk: np.ndarray, col_name: str) -> None:
    db = mongodb_db()
    col = db[col_name]
    col.insert_many(chunk.tolist())  # noqa: PyTypeChecker


def sync_all_to_single_files(splits: int = 50) -> None:
    """Creates one file for all tasks and one file for all predictions.

    Args:
        splits (int): Number of chunks to split the list into before inserting
            to MongoDB.

    Returns:
        None

    """
    logger.debug('Running `sync_all()`...')

    tasks = get_all_projects_tasks()
    for task in tasks:
        task.update({'_id': task['id']})
    preds = get_all_projects_tasks(get_predictions_instead=True)
    for pred in preds:
        pred.update({'_id': pred['id']})

    for res, res_name in zip([tasks, preds], ['tasks', 'preds']):
        logger.debug(f'Syncing all {res_name} to one collection...')
        db = mongodb_db()
        col_name = f'all_projects_{res_name}'
        col = db[col_name]
        col.drop()
        chunks = np.array_split(res, splits)
        desc = f'{res_name.capitalize()} chunks'

        futures = [
            _insert_many_chunks.remote(chunk, col_name) for chunk in chunks
        ]
        _ = [ray.get(future) for future in tqdm(futures, desc=desc)]

        logger.debug(f'Finished syncing all {res_name} to one collection...')

    logger.debug('Finished running `sync_all()`...')
    return


@ray.remote
def sync_project(project_id: Union[int, str],
                 json_min: bool = False,
                 force_update: bool = False) -> None:
    """Runs the sync process for a given project.

    Updates the database with the latest data from the server. It takes the 
    project id as an argument and then makes an API request to the server to 
    get the number of tasks and annotations in the project. It then connects to 
    the database and gets the number of tasks and annotations in the database. 
    If the number of tasks and annotations in the database is not equal to the 
    number of tasks and annotations in the server, then it makes another API 
    request to get the data of all the tasks in the project. It then updates 
    the database with the latest data.

    Args:
        project_id (int or str): The project id.
        json_min (bool): If True, the json_min format will be used.
        force_update (bool): If True, the project will be updated even if no
            changes were detected.

    Returns:
        None

    """

    def _msg(x):
        return f'Difference in {x} number'

    ls_host = os.environ["LS_HOST"]
    project_data = api_request(f'{ls_host}/api/projects/{project_id}/')

    tasks_len_ls = project_data['task_number']
    if tasks_len_ls == 0:
        logger.warning(f'(project: {project_id}) Empty project! Skipping...')
        return
    anno_len_ls = project_data['num_tasks_with_annotations']
    pred_len_ls = project_data['total_predictions_number']

    ls_lens = (tasks_len_ls, anno_len_ls, pred_len_ls)
    logger.debug(f'(project: {project_id}) Tasks: {tasks_len_ls}')
    logger.debug(f'(project: {project_id}) Annotations: {anno_len_ls}')
    logger.debug(f'(project: {project_id}) Predictions: {pred_len_ls}')

    db = mongodb_db()
    if json_min:
        col = db[f'project_{project_id}_min']
    else:
        col = db[f'project_{project_id}']

    tasks_len_mdb = len(list(col.find({})))
    anno_len_mdb = len(list(col.find({"annotations": {'$ne': []}})))
    pred_len_mdb = len(list(db[f'project_{project_id}_preds'].find({})))

    mdb_lens = (tasks_len_mdb, anno_len_mdb, pred_len_mdb)

    if force_update or ((not json_min and ls_lens != mdb_lens) or
                        (json_min and anno_len_ls != anno_len_mdb)):
        logger.debug(
            f'(project: {project_id}) Project has changed. Updating...')
        if not tasks_len_ls - tasks_len_mdb == 0:
            logger.debug(f'(project: {project_id}) {_msg("tasks")}: '
                         f'{abs(tasks_len_ls - tasks_len_mdb)}')
        if not anno_len_ls - anno_len_mdb == 0:
            logger.debug(f'(project: {project_id}) {_msg("annotations")}: '
                         f'{abs(anno_len_ls - anno_len_mdb)}')
        if not pred_len_ls - pred_len_mdb == 0:
            logger.debug(f'(project: {project_id}) {_msg("predictions")}: '
                         f'{abs(pred_len_ls - pred_len_mdb)}')

        if json_min:
            data = api_request(f'{ls_host}/api/projects/{project_id}/export'
                               '?exportType=JSON_MIN&download_all_tasks=true')
            if not data:
                logger.debug(f'No tasks found in project {project_id} '
                             f'(json_min: {json_min}). Skipping...')
                return
        else:
            data = api_request(f'{ls_host}/api/projects/{project_id}/export'
                               '?exportType=JSON&download_all_tasks=true')

        existing_ids = []

        for task in data:
            if json_min:
                img = task['image']  # noqa: PyTypeChecker
            else:
                img = task['data']['image']  # noqa: PyTypeChecker
            task.update({
                '_id': task['id'],  # noqa: PyTypeChecker
                'data': {
                    'image': img
                }
            })  # noqa: PyTypeChecker
            if task['id'] in existing_ids:  # noqa: PyTypeChecker
                logger.error(
                    f'Duplicate annotation in task {task["id"]}! '  # noqa: PyTypeChecker
                    'Fix manually...')
                data.remove(task)
            else:
                existing_ids.append(task['id'])  # noqa: PyTypeChecker

        col.drop()
        col.insert_many(data)

    if not json_min and (force_update or pred_len_ls != pred_len_mdb):
        logger.debug(f'(project: {project_id}) Syncing predictions...')
        proj_preds = api_request(
            f'{ls_host}/api/predictions?task__project={project_id}')
        if proj_preds:
            for pred in proj_preds:
                pred.update({'_id': pred['id']})  # noqa: PyTypeChecker
            col = db[f'project_{project_id}_preds']
            col.drop()
            col.insert_many(proj_preds)
        else:
            logger.debug(
                f'No predictions found in project {project_id}. Skipping...')

    logger.info(f'(project: {project_id}) Finished (json_min: {json_min}).')
    return


def _opts() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-p',
                        '--project-ids',
                        help='Comma-seperated project ids',
                        type=str)
    parser.add_argument('-f',
                        '--force',
                        help='Force update',
                        action='store_true')
    return parser.parse_args()


def sync_tasks(force_update: bool = False) -> None:
    """Synchronizes the tasks of all projects in the database.

    Args:
        force_update (bool): If True, the tasks will be updated even if they
            were already up-to-date.

    Returns:
        None

    """
    catch_keyboard_interrupt()

    if not args.project_ids:
        project_ids = get_project_ids_str().split(',')
    else:
        project_ids = args.project_ids.split(',')

    futures = [
        sync_project.remote(project_id,
                            json_min=False,
                            force_update=force_update)
        for project_id in project_ids
    ]
    _ = [ray.get(future) for future in tqdm(futures, desc='Projects')]

    futures_min = [
        sync_project.remote(project_id,
                            json_min=True,
                            force_update=force_update)
        for project_id in project_ids
    ]
    _ = [ray.get(future) for future in tqdm(futures_min, desc='Projects min')]
    return


if __name__ == '__main__':
    load_dotenv()
    args = _opts()
    if args.force:
        logger.info('Invoked force update!')
    sync_tasks(force_update=args.force)
    # sync_all_to_single_files()

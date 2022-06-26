#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import re
from typing import Optional, Union

import ray
from dotenv import load_dotenv
from tqdm import tqdm

from birdfsd_yolov5.model_utils.handlers import catch_keyboard_interrupt
from birdfsd_yolov5.model_utils.utils import api_request, get_project_ids_str


@ray.remote
def _delete_pred(pred_id: int) -> None:
    resp = api_request(f'{os.environ["LS_HOST"]}/api/predictions/{pred_id}/',
                       method='delete')
    if resp:
        print(f'>>>>>>>>>>> Error in delete response of {pred_id}: {resp}')


def clear_preds_history(model_version_to_keep: str,
                        project_id: Optional[Union[int, str]] = None,
                        all_projects: bool = False) -> None:

    catch_keyboard_interrupt()

    if not project_id and not all_projects:
        raise AssertionError(
            'Pass a project id number or set `all_projects` to True!')

    if all_projects:
        project_ids = get_project_ids_str().split(',')
    else:
        project_ids = [project_id]

    version = model_version_to_keep.split('-v')[1]

    # RegEx pattern source: https://semver.org
    match = re.match(
        r'^(0|[1-9]\d*)\.(0|[1-9]\d*)\.(0|[1-9]\d*)(?:-((?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*)(?:\.(?:0|[1-9]\d*|\d*[a-zA-Z-][0-9a-zA-Z-]*))*))?(?:\+([0-9a-zA-Z-]+(?:\.[0-9a-zA-Z-]+)*))?$',  # noqa
        version)

    if not match:
        raise AssertionError('Not a valid model version!')

    for project_id in tqdm(project_ids, desc='Projects'):

        print(f'Keeping model: {model_version_to_keep}...')
        print(f'Project id: {project_id}')

        _predictions: list = api_request(
            f'{os.environ["LS_HOST"]}/api/predictions'
            f'?task__project={project_id}')

        predictions = [
            pred['id'] for pred in _predictions
            if pred['model_version'] != model_version_to_keep
        ]

        if not predictions:
            print('All predictions are up-to-date!')
            return

        futures = [_delete_pred.remote(pred) for pred in predictions]

        for future in tqdm(futures, desc='Predictions'):
            ray.get(future)


def _opts() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--project-id', help='Project id')
    parser.add_argument('-a',
                        '--all-projects',
                        help='Process all project',
                        action='store_true')
    parser.add_argument('-m',
                        '--model-version-to-keep',
                        help='Model version to keep. All predictions from '
                        'other models will be removed.',
                        type=str,
                        required=True)
    return parser.parse_args()


if __name__ == '__main__':
    load_dotenv()
    args = _opts()
    clear_preds_history(model_version_to_keep=args.model_version_to_keep,
                        project_id=args.project_id,
                        all_projects=args.all_projects)

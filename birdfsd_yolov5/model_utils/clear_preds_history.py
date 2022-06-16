#!/usr/bin/env python
# coding: utf-8

import argparse
import os

import ray
from tqdm import tqdm

from birdfsd_yolov5.model_utils.handlers import catch_keyboard_interrupt
from birdfsd_yolov5.model_utils.utils import api_request


@ray.remote
def _delete_pred(pred: dict, model_version_to_keep: str) -> None:
    if pred['model_version'] != model_version_to_keep:
        pred_id = pred['id']
        resp = api_request(
            f'{os.environ["LS_HOST"]}/api/predictions/{pred_id}/',
            method='delete')
        if resp:
            print(f'>>>>>>>>>>> Error in delete response of {pred_id}: {resp}')


def _opts() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--project-id', help='Poject id', required=True)
    parser.add_argument('-m',
                        '--model-version-to-keep',
                        help='Model version to keep. All predictions from '
                        'other models will be removed.',
                        type=str,
                        required=True)
    return parser.parse_args()


def clear_preds_history(project_id, model_version_to_keep: str) -> None:
    catch_keyboard_interrupt()
    print(f'Project id: {project_id}')

    predictions = api_request(f'{os.environ["LS_HOST"]}/api/predictions'
                              f'?task__project={project_id}')

    futures = [
        _delete_pred.remote(pred, model_version_to_keep)
        for pred in predictions
    ]

    for future in tqdm(futures):
        ray.get(future)


if __name__ == '__main__':
    args = _opts()
    clear_preds_history(project_id=args.project_id,
                        model_version_to_keep=args.model_version_to_keep)


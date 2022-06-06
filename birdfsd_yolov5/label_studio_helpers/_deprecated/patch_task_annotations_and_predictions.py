#!/usr/bin/env python
# coding: utf-8

import argparse
import os

import ray
from dotenv import load_dotenv
from tqdm import tqdm

from birdfsd_yolov5.label_studio_helpers.utils import get_all_projects_tasks
from birdfsd_yolov5.model_utils.handlers import catch_keyboard_interrupt
from birdfsd_yolov5.model_utils.utils import api_request


@ray.remote
def patch_anno(task, _from, to):
    for _entry in task['annotations']:
        entry_id = _entry['id']
        for entry in _entry['result']:
            value = entry['value']
            if not _from == value['rectanglelabels'][0]:
                print(f'Could not find the label `{_from}` in task '
                      f'`{task["id"]}`! Skipping...')
                return

            entry['value']['rectanglelabels'] = [to]
            url = f'{os.environ["LS_HOST"]}/api/annotations/{entry_id}/'
            api_request(url, method='patch', data=_entry)
    return


@ray.remote
def patch_pred(pred, _from, to):
    for result in pred['result']:
        label = result['value']['rectanglelabels']

        if not _from == label[0]:
            print(f'Could not find the label `{_from}` in pred '
                  f'`{pred["id"]}`! Skipping...')
            return

        result['value']['rectanglelabels'] = [to]
        url = f'{os.environ["LS_HOST"]}/api/predictions/{pred["id"]}/'
        api_request(url, method='patch', data=pred)
    return


def check_if_label_exists_in_task_annotations(task, label):
    labels = []
    if not task.get('annotations'):
        return
    results = sum([x['result'] for x in task['annotations']], [])
    for result in results:
        labels.append(result['value']['rectanglelabels'])
    labels = sum(labels, [])
    if label in labels:
        return task
    return


def opts() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-f',
                        '--from-label',
                        help='Label to find and change (i.e., old label)',
                        type=str,
                        required=True)
    parser.add_argument(
        '-t',
        '--to-label',
        help='Label to use instead of the old label (i.e., new label)',
        type=str,
        required=True)
    return parser.parse_args()


def patch(from_label, to_label):
    catch_keyboard_interrupt()
    # --------------------------------------------------------------
    tasks = get_all_projects_tasks()

    tasks_with_label = []
    for task in tqdm(tasks, desc='Scan tasks'):
        task = check_if_label_exists_in_task_annotations(task,
                                                         label=from_label)
        if task:
            tasks_with_label.append(task)

    futures = []
    for task in tasks_with_label:
        futures.append(patch_anno.remote(task, from_label, to_label))

    for future in tqdm(futures, desc='Futures'):
        ray.get(future)
    # --------------------------------------------------------------
    preds = get_all_projects_tasks(get_predictions_instead=True)

    preds_with_label = []
    for pred in tqdm(preds, desc='Scan preds'):
        for result in pred['result']:
            label = result['value']['rectanglelabels']
            if from_label in label:
                preds_with_label.append(pred)

    futures = []
    for pred in preds_with_label:
        futures.append(patch_pred.remote(pred, from_label, to_label))

    for future in tqdm(futures, desc='Futures'):
        ray.get(future)
    # --------------------------------------------------------------
    ray.shutdown()


if __name__ == '__main__':
    load_dotenv()
    args = opts()
    patch(from_label=args.from_label, to_label=args.to_label)

#!/usr/bin/env python
# coding: utf-8

import argparse
import copy
import json
import os
from typing import Optional

import numpy as np
import requests
from dotenv import load_dotenv
from loguru import logger
from requests.structures import CaseInsensitiveDict

from birdfsd_yolov5.model_utils.handlers import catch_keyboard_interrupt
from birdfsd_yolov5.model_utils.mongodb_helper import mongodb_db
from birdfsd_yolov5.model_utils.utils import (add_logger, get_project_ids_str,
                                              upload_logs)


class CreateRareClassesView:
    """Create a label-studio view tab with filters that shows rare classes."""

    def __init__(self,
                 project_id: int,
                 model_version: str,
                 method: str = 'median'):
        """
        Args:
            project_id (str): The project id of the model to be used.
            model_version (str): The version of the model to be used.
            method (str): The method to be used for imputation.
                Options:
                    - 'median': Use the median of the column to impute missing 
                    values.
                    - 'mean': Use the mean of the column to impute missing 
                    values.
        
        Returns:
            None

        """
        self.project_id = project_id
        self.model_version = model_version
        self.method = method

    @staticmethod
    def _make_headers() -> CaseInsensitiveDict:
        """Creates a dictionary of headers for the API requests.

        Returns:
            headers (dict): A dictionary of headers for the API requests.

        Raises:
            None

        """
        headers = CaseInsensitiveDict()
        headers['Authorization'] = f'Token {os.environ["TOKEN"]}'
        headers['Content-type'] = 'application/json'
        return headers

    def create_view(self) -> Optional[dict]:
        """Creates a new view in the project with the rare classes.

        Returns:
            A dictionary containing the new view.

        """
        catch_keyboard_interrupt()

        db = mongodb_db(os.environ['DB_CONNECTION_STRING'])
        if self.model_version == 'latest':
            latest_model_ts = max(db.model.find().distinct('added_on'))
            d = db.model.find_one({'added_on': latest_model_ts})
        else:
            d = db.model.find_one({'_id': self.model_version})
        logger.debug(f'Model version: {d["_id"]}')

        labels_vals = list(d['labels'].values())
        if self.method == 'mean':
            count_m = np.mean(labels_vals)
        elif self.method == 'median':
            count_m = np.median(labels_vals)
        else:
            count_m = 10

        excluded_labels = os.getenv('EXCLUDE_LABELS')
        if excluded_labels:
            excluded_labels = excluded_labels.split(',')
        else:
            excluded_labels = []

        labels_with_few_annos = []
        for k, v in d['labels'].items():
            if count_m > v and k not in excluded_labels:
                labels_with_few_annos.append(k)

        headers = self._make_headers()

        view_template = {
            'data': {
                'type': 'list',
                'title': '',
                'target': 'tasks',
                'gridWidth': 4,
                'columnsWidth': {},
                'hiddenColumns': {
                    'explore': [
                        'tasks:annotations_results', 'tasks:annotations_ids',
                        'tasks:predictions_score', 'tasks:predictions_results',
                        'tasks:file_upload', 'tasks:created_at',
                        'tasks:updated_at'
                    ],
                    'labeling': [
                        'tasks:id', 'tasks:completed_at',
                        'tasks:cancelled_annotations',
                        'tasks:total_predictions', 'tasks:annotators',
                        'tasks:annotations_results', 'tasks:annotations_ids',
                        'tasks:predictions_score',
                        'tasks:predictions_model_versions',
                        'tasks:predictions_results', 'tasks:file_upload',
                        'tasks:created_at', 'tasks:updated_at'
                    ]
                },
                'columnsDisplayType': {},
                'filters': {
                    'conjunction':
                    'or',
                    'items': [{
                        'filter': 'filter:tasks:predictions_results',
                        'operator': 'equal',
                        'type': 'String',
                        'value': 'placeholder_a'
                    }, {
                        'filter': 'filter:tasks:predictions_results',
                        'operator': 'equal',
                        'type': 'String',
                        'value': 'placeholder_b'
                    }]
                }
            }
        }

        default_view = copy.deepcopy(view_template)

        filtered_labels = []
        for label in labels_with_few_annos:
            filtered_labels.append({
                'filter': 'filter:tasks:predictions_results',
                'operator': 'contains',
                'type': 'String',
                'value': label
            })

        view_template['data']['filters']['conjunction'] = 'or'  # noqa: PyTypeChecker
        view_template['data']['filters']['items'] = filtered_labels
        view_template['data']['title'] = 'rare_classes'

        view_template.update({'project': self.project_id})

        url = f'{os.environ["LS_HOST"]}/api/dm/views?project={self.project_id}'
        resp = requests.get(url, headers=headers)

        default_tab = [
            x for x in resp.json() if x['data']['title'] == 'Default'
        ]

        if not default_tab:
            logger.debug(
                f'Creating default view for project {self.project_id}')
            default_view.update({'project': self.project_id})
            default_view['data']['title'] = 'Default'
            default_view['data'].pop('filters')
            url = f'{os.environ["LS_HOST"]}/api/dm/views/'
            new_view_resp = requests.post(url,
                                          headers=headers,
                                          data=json.dumps(default_view))
            new_default_view = new_view_resp.json()
            logger.debug(f'Response: {new_default_view}')

        existing_rare_classes_tab = [
            x for x in resp.json() if x['data']['title'] == 'rare_classes'
        ]

        if existing_rare_classes_tab:
            version_col = 'tasks:predictions_model_versions'
            explore_dict = existing_rare_classes_tab[0]['data'][
                'hiddenColumns']['explore']
            if existing_rare_classes_tab[0]['data']['filters'][
                    'items'] == filtered_labels and (version_col
                                                     in explore_dict):
                logger.debug(
                    'An identical `rare_classes` view already exists for '
                    f'project {self.project_id}. Skipping...')
                return
            else:
                logger.debug(
                    'The list of rare classes has changed! Replacing...')
                existing_view_id = existing_rare_classes_tab[0]['id']
                url = f'{os.environ["LS_HOST"]}/api/dm/views/' \
                      f'{existing_view_id}'
                _ = requests.delete(url, headers=headers)

        url = f'{os.environ["LS_HOST"]}/api/dm/views/'
        logger.debug(f'Request: {url} -d {view_template}')
        resp = requests.post(url,
                             headers=headers,
                             data=json.dumps(view_template))
        new_view = resp.json()
        logger.debug(f'Response: {new_view}')
        return new_view


def _opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--project-ids', help='Project ids')
    parser.add_argument('-v',
                        '--model-version',
                        help='Model version',
                        type=str,
                        required=True)
    parser.add_argument(
        '-m',
        '--method',
        type=str,
        help='The method used to calculate underrepresented classes',
        choices=['mean', 'median'],
        default='median')
    return parser.parse_args()


if __name__ == '__main__':
    load_dotenv()
    logs_file = add_logger(__file__)

    args = _opts()

    if not args.project_ids:
        project_ids = get_project_ids_str().split(',')
    else:
        project_ids = args.project_ids.split(',')

    for proj_id in project_ids:
        create_rare_classes_view = CreateRareClassesView(
            project_id=proj_id,
            model_version=args.model_version,
            method=args.method)
        _ = create_rare_classes_view.create_view()

    upload_logs(logs_file)

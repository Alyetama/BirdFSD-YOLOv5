#!/usr/bin/env python
# coding: utf-8

import argparse
import json
import os

import numpy as np
import requests
from dotenv import load_dotenv

from ..mongodb_helper import mongodb_db


class CreateRareLabelsView:

    def __init__(self, project_id, model_version):
        self.project_id = project_id
        self.model_version = model_version

    def create_view(self, method='median'):
        db = mongodb_db()
        d = self.db.model.find_one({'_id': self.model_version})

        labels_vals = list(d['labels'].values())
        if method == 'mean':
            count_m = np.mean(labels_vals)
        elif method == 'median':
            count_m = np.median(labels_vals)

        excluded_labels = [
            'cannot identify', 'no animal', 'distorted image',
            'severe occultation', 'animal other than bird or squirrel'
        ]

        labels_with_few_annos = []
        for k, v in d['labels'].items():
            if count_m > v and k not in excluded_labels:
                labels_with_few_annos.append(k)

        headers = requests.structures.CaseInsensitiveDict()  # noqa
        headers['Authorization'] = f'Token {os.environ["TOKEN"]}'
        headers['Content-type'] = 'application/json'

        template_view_id = 68  # hard-coded view id to use as a template
        url = f'{os.environ["LS_HOST"]}/api/dm/views/{template_view_id}'
        request_data = {'project': f'{self.project_id}'}
        resp = requests.get(url,
                            headers=headers,
                            data=json.dumps(request_data))
        view_template = resp.json()

        filterd_labels = []
        for label in labels_with_few_annos:
            filterd_labels.append({
                'filter': 'filter:tasks:predictions_results',
                'operator': 'contains',
                'type': 'String',
                'value': label
            })

        view_template['data']['filters']['conjunction'] = 'or'
        view_template['data']['filters']['items'] = filterd_labels
        view_template['data']['title'] = 'rare_birds'

        view_template.pop('id')
        view_template.pop('user')

        view_template.update({'project': self.project_id})

        url = f'{os.environ["LS_HOST"]}/api/dm/views/'
        resp = requests.post(url,
                             headers=headers,
                             data=json.dumps(view_template))
        new_view = resp.json()
        return new_view


if __name__ == '__main__':
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--project-id', help='Project ID', required=True)
    parser.add_argument('-v',
                        '--model-version',
                        help='Model version',
                        type=str,
                        required=True)
    args = parser.parse_args()

    create_rare_labels_view = CreateRareLabelsView(
        project_id=args.project_id, model_version=args.model_version)
    _ = create_rare_labels_view.create_view(method='median')

#!/usr/bin/env python
# coding: utf-8

import argparse
import json
import os

import requests
from dotenv import load_dotenv
from loguru import logger


def update_view():
    url = f'{os.environ["LS_HOST"]}/api/dm/views/{args.view_id}/'
    resp = requests.get(url, headers=headers)
    view = resp.json()

    with open(args.classes_path) as f:
        _classes = [x.rstrip() for x in f.readlines() if x != 'squirrels\n']
    logger.debug(_classes)

    filter_items = [{
        'filter': 'filter:tasks:total_annotations',
        'operator': 'equal',
        'type': 'Number',
        'value': 0
    }, {
        'filter': 'filter:tasks:predictions_results',
        'operator': 'contains',
        'type': 'String',
        'value': 'squirrels'
    }]

    for _class in _classes:
        filter_items.append({
            'filter': 'filter:tasks:predictions_results',
            'operator': 'not_contains',
            'type': 'String',
            'value': _class
        })

    view['data']['filters']['items'] = filter_items

    logger.debug(f'URL: {url} ; DATA: {view}')
    resp = requests.put(url, data=json.dumps(view), headers=headers)
    logger.debug(resp.json())


def opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p',
                        '--project-id',
                        help='Label-studio project ID',
                        type=int,
                        required=True)
    parser.add_argument('-v',
                        '--view-id',
                        help='View to filter',
                        type=int,
                        required=True)
    parser.add_argument('-c',
                        '--classes-path',
                        help='Path to classes.txt',
                        type=str,
                        required=True)
    return parser.parse_args()


if __name__ == '__main__':
    logger.add('logs.log')
    load_dotenv()
    args = opts()

    headers = requests.structures.CaseInsensitiveDict()  # noqa
    headers['Authorization'] = f'Token {os.environ["TOKEN"]}'
    headers['Content-type'] = 'application/json'

    update_view()
    print('-' * 40)
    logger.info('Delete all tasks under: https://ls.aibird.me/projects/'
                f'{args.project_id}/data?tab={args.view_id}')

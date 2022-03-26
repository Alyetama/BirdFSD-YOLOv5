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

    resp = requests.put(url, data=json.dumps(view), headers=headers)
    logger.debug(resp.json())


def opts():
    parser = argparse.ArgumentParser()
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
    load_dotenv()
    args = opts()

    headers = requests.structures.CaseInsensitiveDict()
    headers['Authorization'] = f'Token {os.environ["TOKEN"]}'
    headers['Content-type'] = 'application/json'

    update_view()
    logger.info(
        f'Delete all tasks under: https://ls.aibird.me/projects/1/data?tab={args.view_id}'
    )

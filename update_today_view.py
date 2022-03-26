import argparse
import json
import os
from datetime import date

import requests
from dotenv import load_dotenv
from loguru import logger


def update_today_view(DATE):
    url = f'{os.environ["LS_HOST"]}/api/dm/views/{args.view_id}/'
    resp = requests.get(url, headers=headers)
    view = resp.json()

    FILTER = [{
        'filter': 'filter:tasks:created_at',
        'operator': 'greater',
        'type': 'Datetime',
        'value': f'{DATE}T04:00:00.000Z'
    }]

    view['data']['filters']['items'] = FILTER
    view['data']['ordering'] = ['tasks:id']
    resp = requests.put(url, data=json.dumps(view), headers=headers)
    logger.debug(resp.json())


def opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v',
                        '--view-id',
                        help='View to filter',
                        type=int,
                        required=True)
    return parser.parse_args()


if __name__ == '__main__':
    load_dotenv()
    args = opts()

    headers = requests.structures.CaseInsensitiveDict()
    headers['Authorization'] = f'Token {os.environ["TOKEN"]}'
    headers['Content-type'] = 'application/json'

    DATE = date.today()
    update_today_view(DATE)

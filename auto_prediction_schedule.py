#!/usr/bin/env python
# coding: utf-8

import os
import sys
import time
from pathlib import Path

import schedule
from dotenv import load_dotenv
from loguru import logger

from predict import Predict


def main():
    CONFIG = {
        'weights': '',
        'tasks_range': None,
        'predict_all': True,
        'one_task': None,
        'model_version': 'latest',
        'multithreading': True,
        'delete_if_no_predictions': False,
        'if_empty_apply_label': 'no animal',
        'debug': False
    }

    pids = os.environ['PROJECTS_ID'].split(',')

    for pid in pids:
        logger.info(f'\nCurrent project id: {pid}\n')
        CONFIG.update({'project_id': pid})
        predict = Predict(**CONFIG)
        predict.apply_predictions()


if __name__ == '__main__':
    load_dotenv()
    logger.add(f'{Path(__file__).parent}/logs.log')
    if '--once' in sys.argv:
        main()
        sys.exit(0)

    schedule.every().day.at('08:00').do(main)

    while True:
        schedule.run_pending()
        time.sleep(1)

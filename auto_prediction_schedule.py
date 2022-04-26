#!/usr/bin/env python
# coding: utf-8

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

import schedule
from dotenv import load_dotenv
from loguru import logger

from predict import Predict


def keyboard_interrupt_handler(sig: int, _) -> None:
    logger.warning(f'KeyboardInterrupt (id: {sig}) has been caught...')
    logger.info('Terminating the session gracefully...')
    sys.exit(1)


def opts() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-o',
                        '--once',
                        action='store_true',
                        help='Run once then exit')
    parser.add_argument(
        '--show-config',
        action='store_true',
        help='Show the current prediction function configuration then exit')
    return parser.parse_args()


def main() -> None:
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

    if not Path('config.json').exists():
        logger.debug('Using default configuration...')
        with open('config.json', 'w') as j:
            json.dump(CONFIG, j, indent=4)
    else:
        with open('config.json') as j:
            _CONFIG = json.load(j)
        if CONFIG == _CONFIG:
            logger.debug('Using default configuration...')
        else:
            CONFIG = _CONFIG

    if args.show_config:
        print(json.dumps(CONFIG, indent=4))
        sys.exit(0)

    pids = os.environ['PROJECTS_ID'].split(',')

    for pid in pids:
        logger.info(f'\nCurrent project id: {pid}\n')
        CONFIG.update({'project_id': pid})
        predict = Predict(**CONFIG)
        predict.apply_predictions()


if __name__ == '__main__':
    load_dotenv()
    logger.add(f'{Path(__file__).parent}/logs.log')
    args = opts()
    signal.signal(signal.SIGINT, keyboard_interrupt_handler)

    if args.once or args.show_config:
        main()
        sys.exit(0)

    logger.debug('Scheduled to run every day at 08:00...')
    schedule.every().day.at('08:00').do(main)

    while True:
        schedule.run_pending()
        time.sleep(1)

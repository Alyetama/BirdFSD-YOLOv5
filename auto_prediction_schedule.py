#!/usr/bin/env python
# coding: utf-8

import argparse
import json
import sys
import time
from pathlib import Path

import schedule
from dotenv import load_dotenv
from loguru import logger

from model_utils.download_weights import DownloadModelWeights
from model_utils.handlers import catch_keyboard_interrupt
from model_utils.utils import add_logger, upload_logs
from model_utils.utils import get_project_ids
from predict import Predict


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
    logs_file = add_logger(__file__)
    catch_keyboard_interrupt()

    CONFIG = {
        'weights': '',
        'project_ids': get_project_ids(),
        'tasks_range': None,
        'predict_all': True,
        'one_task': None,
        'model_version': 'latest',
        'multithreading': True,
        'delete_if_no_predictions': False,
        'if_empty_apply_label': 'no animal',
        'get_tasks_with_api': False
    }

    if not CONFIG['weights'] and CONFIG['model_version'] == 'latest':
        dmw = DownloadModelWeights(CONFIG['model_version'])
        weights, weights_url, weights_model_ver = dmw.get_weights(
            skip_download=False)
        logger.info(f'Downloaded weights to {weights}')
        CONFIG['weights'] = weights
        CONFIG['model_version'] = weights_model_ver
    else:
        if CONFIG['model_version'] == 'latest':
            raise sys.exit(
                'Need to specify model version if loaded from a file path!')

    if args.show_config:
        print(json.dumps(CONFIG, indent=4))
        sys.exit(0)

    pids = 
    predict = Predict(**CONFIG)
    predict.apply_predictions()

    try:
        Path('best.pt').unlink()
    except FileNotFoundError:
        pass

    upload_logs(logs_file)
    return


if __name__ == '__main__':
    load_dotenv()
    args = opts()

    if args.once or args.show_config:
        main()
        sys.exit(0)

    logger.debug('Scheduled to run every day at 08:00...')
    schedule.every().day.at('08:00').do(main)

    while True:
        schedule.run_pending()
        time.sleep(1)

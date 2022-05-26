#!/usr/bin/env python
# coding: utf-8

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv
from loguru import logger

from birdfsd_yolov5.model_utils import download_weights, handlers, utils
from birdfsd_yolov5.prediction import predict


def opts() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Namespace object containing the parsed arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('--opts-file',
                        type=str,
                        help='JSON file with predict.py options')
    parser.add_argument(
        '--show-opts',
        action='store_true',
        help='Show the current prediction function configuration then exit')
    return parser.parse_args()


def main() -> None:
    """A pipeline to run the prediction module. Intended to be used as
    systemctl service or inside a GitHub action.

    Returns:
        None
    """
    logs_file = utils.add_logger(__file__)
    handlers.catch_keyboard_interrupt()

    if args.opts_file:
        logger.debug(f'Loading options from file: `{args.opts_file}`...')
        with open(args.opts_file) as j:
            OPTS = json.load(j)
    else:
        logger.debug('Using default options...')
        OPTS = {
            'weights': '',
            'project_ids': None,  # None will return all projects
            'tasks_range': None,
            'predict_all': True,
            'one_task': None,
            'model_version': 'latest',
            'multithreading': True,
            'delete_if_no_predictions': False,
            'if_empty_apply_label': 'no animal',
            'get_tasks_with_api': False
        }

    logger.debug(f'OPTS: {OPTS}')

    if not OPTS['weights'] and OPTS['model_version'] == 'latest':
        dmw = download_weights.DownloadModelWeights(OPTS['model_version'])
        skip_download = False
        if args.show_config:
            skip_download = True
        weights, weights_url, weights_model_ver = dmw.get_weights(
            skip_download=skip_download)
        logger.info(f'Downloaded weights to {weights}')
        OPTS['weights'] = weights
        OPTS['model_version'] = weights_model_ver
    else:
        if OPTS['model_version'] == 'latest':
            raise sys.exit(
                'Need to specify model version if loaded from a file path!')

    if args.show_opts:
        print(json.dumps(OPTS, indent=4))
        sys.exit(0)

    p = predict.Predict(**OPTS)
    p.apply_predictions()

    try:
        Path('best.pt').unlink()
    except FileNotFoundError:
        pass
    utils.upload_logs(logs_file)
    return


if __name__ == '__main__':
    load_dotenv()
    args = opts()
    main()
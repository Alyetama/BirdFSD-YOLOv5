#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import sys
import time

import schedule
from dotenv import load_dotenv
from loguru import logger

from birdfsd_yolov5.label_studio_helpers._deprecated.sync_images import (
    sync_images)
from birdfsd_yolov5.label_studio_helpers._deprecated.sync_local_storage import (  # noqa: E501
    sync_local_storage)
from birdfsd_yolov5.label_studio_helpers.create_rare_classes_view import (
    CreateRareClassesView)
from birdfsd_yolov5.label_studio_helpers.sync_tasks import sync_tasks
from birdfsd_yolov5.model_utils.handlers import catch_keyboard_interrupt
from birdfsd_yolov5.model_utils.utils import (add_logger, get_project_ids_str,
                                              upload_logs)


class MissingEnvironmentVariable(Exception):
    pass


def opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o',
                        '--once',
                        help='Run once then exit',
                        action='store_true')
    return parser.parse_args()


def main():
    start = time.time()
    logs_file = add_logger(__file__)
    catch_keyboard_interrupt()

    logger.info('Running `create_rare_classes_view`...')

    if not args.project_ids:
        project_ids = get_project_ids_str().split(',')
    else:
        project_ids = args.project_ids.split(',')

    for project_id in project_ids:
        logger.debug(f'Current project id: {project_id}')
        create_rare_classes_view = CreateRareClassesView(
            project_id=project_id, model_version='latest', method='median')
        _ = create_rare_classes_view.create_view()

    logger.info('Running `sync_local_storage`...')
    sync_local_storage()

    logger.info('Running `sync_tasks`...')
    sync_tasks()

    if os.getenv('LOCAL_DB_CONNECTION_STRING'):
        logger.info('Running `sync_images`...')
        sync_images()

    logger.info(f'End. Took {round(time.time() - start, 2)}s')

    upload_logs(logs_file)


if __name__ == '__main__':
    load_dotenv()
    args = opts()

    if args.once:
        main()
        sys.exit(0)

    schedule.every().day.at('06:00').do(main)
    schedule.every().day.at('10:00').do(main)

    while True:
        ray_is_running = False
        schedule.run_pending()
        time.sleep(1)

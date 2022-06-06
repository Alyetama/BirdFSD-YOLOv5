#!/usr/bin/env python
# coding: utf-8

import argparse
import gzip
import os
import sys
import time
from pathlib import Path

import ray
import requests
import schedule
from dotenv import load_dotenv
from pymongo.errors import DuplicateKeyError
from tqdm import tqdm

from birdfsd_yolov5.model_utils.handlers import catch_keyboard_interrupt
from birdfsd_yolov5.model_utils.mongodb_helper import (mongodb_db,
                                                       get_tasks_from_mongodb)
from birdfsd_yolov5.model_utils.utils import (add_logger, get_project_ids_str,
                                              upload_logs)


@ray.remote
def img_url_to_binary(x):
    return {
        '_id': x['_id'],
        'file_name': Path(x['data']['image'].split('?')[0]).name,
        'image': gzip.compress(requests.get(x['data']['image']).content)
    }


def sync_images():

    def insert_image(d):
        try:
            db.images.insert_one(d)
        except DuplicateKeyError:
            db.images.delete_one({'_id': d['_id']})
            db.images.insert_one(d)

    logs_file = add_logger(__file__)
    catch_keyboard_interrupt()

    db = mongodb_db(os.environ['LOCAL_DB_CONNECTION_STRING'])
    main_db = mongodb_db(os.environ['DB_CONNECTION_STRING'])

    existing_ids = db.images.find().distinct('_id')

    if not args.project_ids:
        project_ids = get_project_ids_str().split(',')
    else:
        project_ids = args.project_ids.split(',')

    data = sum([
        get_tasks_from_mongodb(
            db=main_db, project_id=project_id, dump=False, json_min=False)
        for project_id in project_ids
    ], [])

    data = [x for x in data if x['_id'] not in existing_ids]

    futures = []
    for x in data:
        futures.append(img_url_to_binary.remote(x))

    for future in tqdm(futures):
        insert_image(ray.get(future))

    upload_logs(logs_file)


if __name__ == '__main__':
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument('--once',
                        help='Run once and exit',
                        action='store_true')
    args = parser.parse_args()

    if args.once:
        sync_images()
        sys.exit(0)

    schedule.every(6).hours.do(sync_images)

    while True:
        schedule.run_pending()
        time.sleep(1)

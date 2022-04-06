#!/usr/bin/env python
# coding: utf-8

import json
import os
import sys

import requests
import pymongo
from dotenv import load_dotenv


def api_request(url):
    headers = requests.structures.CaseInsensitiveDict()  # noqa
    headers['Authorization'] = f'Token {os.environ["TOKEN"]}'
    resp = requests.get(url, headers=headers)
    return resp.json()


def mongodb_db():
    client = pymongo.MongoClient(os.environ['DB_CONNECTION_STRING'])
    db = client[os.environ['DB_NAME']]
    return db


def get_tasks_from_mongodb(project_id):
    db = mongodb_db()
    col = db[f'project_{project_id}']
    tasks = list(col.find({}, {}))

    with open('tasks.json', 'w') as j:
        json.dump(tasks, j, indent=4)
    return tasks


if __name__ == '__main__':
    load_dotenv()
    if len(sys.argv) == 1:
        raise SystemExit('Missing project ID!')
    get_tasks_from_mongodb(sys.argv[1])

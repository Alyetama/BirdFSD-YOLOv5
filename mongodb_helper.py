#!/usr/bin/env python
# coding: utf-8

import json
import os
import sys

import requests
import pymongo
from dotenv import load_dotenv


def api_request(url):
    """Make a GET request to the given url.
    
    Parameters
    ----------
    url : str
        The url to make the request to.
    
    Returns
    -------
    dict
        The JSON response as a dictionary.
    """
    headers = requests.structures.CaseInsensitiveDict()  # noqa
    headers['Authorization'] = f'Token {os.environ["TOKEN"]}'
    resp = requests.get(url, headers=headers)
    return resp.json()


def mongodb_db():
    """Connects to the MongoDB database.

    Returns
    -------
    db: pymongo.database.Database
        The MongoDB database.
    """
    client = pymongo.MongoClient(os.environ['DB_CONNECTION_STRING'])
    db = client[os.environ['DB_NAME']]
    return db


def get_tasks_from_mongodb(project_id, dump=True, json_min=False):
    """Get tasks from MongoDB.

    Parameters
    ----------
    project_id : int
        The ID of the project to get tasks from.

    Returns
    -------
    tasks : list
        A list of tasks.
    """
    db = mongodb_db()
    if json_min:
        col = db[f'project_{project_id}_min']
    else:
        col = db[f'project_{project_id}']
    tasks = list(col.find({}, {}))

    if dump:
        with open('tasks.json', 'w') as j:
            json.dump(tasks, j, indent=4)
    return tasks


if __name__ == '__main__':
    load_dotenv()
    if len(sys.argv) == 1:
        raise SystemExit('Missing project ID!')
    get_tasks_from_mongodb(sys.argv[1])

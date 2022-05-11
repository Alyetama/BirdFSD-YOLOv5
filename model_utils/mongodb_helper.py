#!/usr/bin/env python
# coding: utf-8

import json
import os
import sys
from typing import Union

import pymongo
from dotenv import load_dotenv


def mongodb_db() -> pymongo.database.Database:  # noqa
    """Connects to the MongoDB database.

    Returns
    -------
    db: pymongo.database.Database
        The MongoDB database.
    """
    db_connection_string = os.getenv('DB_CONNECTION_STRING')
    if not db_connection_string:
        return
    client = pymongo.MongoClient(db_connection_string)
    db = client[os.environ['DB_NAME']]
    return db


def get_tasks_from_mongodb(project_id: Union[int, str],
                           dump: bool = True,
                           json_min: bool = False):
    """Get tasks from MongoDB.

    Parameters
    ----------
    project_id : Union[int, str]
        The id of the project to get tasks from.
    dump : bool
        Whether to dump the data to a JSON file.
    json_min : bool
        The data will be exported as JSON_MIN when set to True.

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
    tasks = list(col.find({}))

    if dump:
        with open('tasks.json', 'w') as j:
            json.dump(tasks, j, indent=4)
    return tasks


if __name__ == '__main__':
    load_dotenv()
    if len(sys.argv) == 1:
        raise SystemExit('Missing project id!')
    get_tasks_from_mongodb(sys.argv[1])

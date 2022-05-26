#!/usr/bin/env python
# coding: utf-8

import json
import os
from typing import Union

import pymongo


def mongodb_db() -> pymongo.database.Database:  # noqa
    """Create a MongoDB client. The database connection string is read from the
    environment variable `DB_CONNECTION_STRING`. The database name is read from
    the environment variable `DB_NAME`. If either of these environment
    variables are not set, None is returned.

    Returns:
        pymongo.database.Database: a MongoDB database object.
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
    """Get tasks from mongodb.

    Args:
        project_id (Union[int, str]): The project id.
        dump (bool): Dump the tasks to a json file.
        json_min (bool): Use the minified json.

    Returns:
        list: A list of tasks.
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

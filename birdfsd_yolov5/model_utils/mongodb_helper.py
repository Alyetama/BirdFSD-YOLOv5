#!/usr/bin/env python
# coding: utf-8

import json
import os
from typing import Optional

import pymongo
from pymongo.database import Database


class MissingConnectionString(Exception):
    pass


def mongodb_db(db_connection_string: Optional[str] = None) -> Database:
    """Create a MongoDB client.

    The database connection string is read from the
    environment variable `DB_CONNECTION_STRING`. The database name is read from
    the environment variable `DB_NAME`. If either of these environment
    variables are not set, None is returned.

    Args:
        db_connection_string (str): MongoDB connection string.

    Returns:
        pymongo.database.Database: a MongoDB database object.
        
    """

    if not db_connection_string:
        db_connection_string = os.getenv('DB_CONNECTION_STRING')

    if not db_connection_string:
        raise MissingConnectionString('Add `DB_CONNECTION_STRING` as an '
                                      'environment variable!')
    client = pymongo.MongoClient(db_connection_string)
    db = client[os.environ['DB_NAME']]
    return db


def get_tasks_from_mongodb(project_id: str,
                           db: Optional[Database] = None,
                           dump: bool = False,
                           json_min: bool = False,
                           get_predictions: bool = False) -> list:
    """Get tasks from mongodb.

    Args:
        project_id (Union[int, str]): The project id.
        db (pymongo.database.Database): MongoDB client instance.
        dump (bool): Dump the tasks to a json file.
        json_min (bool): Use the minified json.
        get_predictions (bool): Get predictions instead of tasks.

    Returns:
        list: A list of tasks.

    """
    if db is None:
        db = mongodb_db()

    if json_min:
        col = db[f'project_{project_id}_min']
    elif get_predictions:
        col = db[f'project_{project_id}_preds']
    else:
        col = db[f'project_{project_id}']
    tasks = list(col.find({}))

    if dump:
        with open('tasks.json', 'w') as j:
            json.dump(tasks, j, indent=4)
    return tasks

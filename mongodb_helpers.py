import os

import pymongo


def mongodb():
    client = pymongo.MongoClient(os.environ['DB_CONNECTION_STRING'])
    db = client[os.environ['DB_NAME']]
    return db


def get_mongodb_data():
    db = mongodb()
    return list(db.bbox.find())

#!/usr/bin/env python
# coding: utf-8

import os
from pathlib import Path

import torch
from minio import Minio

from birdfsd_yolov5.model_utils.mongodb_helper import mongodb_db


def create_s3_client(api_s3=False):
    prefix = ''
    if api_s3:
        prefix = 'API_'
    return Minio(os.environ[f'{prefix}S3_ENDPOINT'],
                 access_key=os.environ[f'{prefix}S3_ACCESS_KEY'],
                 secret_key=os.environ[f'{prefix}S3_SECRET_KEY'],
                 region=os.environ[f'{prefix}S3_REGION'])


def get_latest_model_weights(s3_client, skip_download=False):
    db = mongodb_db()
    col = db['model']
    latest_model_ts = max(col.distinct('added_on'))
    model_document = db.model.find_one({'added_on': latest_model_ts})
    model_version = model_document['version']
    model_name = model_document['name']
    model_object_name = f'{model_name}-v{model_version}.pt'
    if skip_download:
        return model_version, model_name, model_object_name

    _ = s3_client.fget_object('model', model_object_name, model_object_name)
    if not Path(model_object_name).exists():
        raise AssertionError
    return model_version, model_name, model_object_name


def init_model(s3):
    model_version, model_name, model_weights = get_latest_model_weights(
        s3, skip_download=True)

    if not Path(model_weights).exists():
        model_version, model_name, model_weights = get_latest_model_weights(s3)

    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_weights)
    return model_version, model_name, model_weights, model

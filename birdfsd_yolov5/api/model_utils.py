# -*- coding: utf-8 -*-
"""This module is used to get the latest model weights and information."""

from pathlib import Path
from typing import Optional, Tuple

import torch
from minio import Minio

from birdfsd_yolov5.model_utils.mongodb_helper import mongodb_db


def get_latest_model_weights(s3_client: Minio,
                             skip_download: bool = False
                             ) -> Tuple[str, str, str]:
    """Get the latest model weights from the model collection in mongodb.

    Args:
        s3_client (Minio): Minio S3 client object
        skip_download (bool): If True, skip downloading the model weights
            from S3.

    Returns:
        model_version: The version of the model weights
        model_name: The name of the model
        model_object_name: The name of the model weights file

    """
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


def init_model(
    s3: Minio,
    use_weights: Optional[str] = None
) -> Tuple[str, str, str, torch.nn.Module]:
    """This function initializes the model.

    Args:
        s3 (Minio): Minio S3 client object.
        use_weights (str): Use this weights file instead of the latest model.

    Returns:
        model_version: The model version.
        model_name: The model name.
        model_weights: The model weights file name.
        model: The model object.

    """
    if not use_weights:
        model_version, model_name, model_weights = get_latest_model_weights(
            s3, skip_download=True)
    else:
        model_version = Path(use_weights).stem
        model_name = Path(use_weights).stem
        model_weights = use_weights

    if not Path(model_weights).exists():
        model_version, model_name, model_weights = get_latest_model_weights(s3)

    model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_weights)
    return model_version, model_name, model_weights, model


def model_info(version: str) -> dict:
    """Returns the model information for the specified version.

    Args:
        version (str): The version of the model to be returned.

    Returns:
        dict: A dictionary containing the model information.

    """
    mdb = mongodb_db()
    col = mdb.model
    if version == 'latest':
        latest_model_ts = max(col.find().distinct('added_on'))
        model_obj = col.find({'added_on': latest_model_ts}).next()
    else:
        model_obj = col.find({'version': version}).next()
    model_obj.pop('_id')
    model_obj['added_on'] = str(model_obj['added_on'])
    model_obj['trained_on'] = str(model_obj['trained_on'])
    model_obj.pop('projects')
    return model_obj

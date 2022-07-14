# -*- coding: utf-8 -*-
"""This module contains utility functions for the model API."""

import io
import json
import os
import re
import tempfile
import uuid
import zipfile
from typing import Any, Dict, Optional, Union

import requests
from PIL import Image
from minio import Minio


def create_s3_client(api_s3: bool = False) -> Minio:
    """Creates a Minio client object.

    Args:
        api_s3: If True, use the API S3 credentials.

    Returns:
        A Minio client object.

    """
    prefix = ''
    if api_s3:
        prefix = 'API_'
    s3_endpoint = re.sub(r'https?:\/\/', '', os.environ[f'{prefix}S3_ENDPOINT'])
    return Minio(s3_endpoint,
                 access_key=os.environ[f'{prefix}S3_ACCESS_KEY'],
                 secret_key=os.environ[f'{prefix}S3_SECRET_KEY'],
                 region=os.environ[f'{prefix}S3_REGION'])


def species_info(species_name: str) -> Union[dict, str]:
    """Retrieves a dictionary of information about the species.

    This function takes a species name as input and returns a dictionary of
    information about the species.

    Args:
        species_name (str): The name of the species to be searched.

    Returns:
        dict: A dictionary of information about the species.

    Raises:
        ValueError: If the species name is not a string.

    """
    if '(' in species_name:
        species_name = species_name.split('(')[1].split(')')[0]
    url = f'https://api.gbif.org/v1/species/match?name={species_name}'
    response = requests.get(url)
    json_resp = response.json()
    if not json_resp.get('usageKey'):
        return 'no results'
    excl_keys = ['status', 'confidence', 'matchType']
    compact_info = {
        k: v
        for k, v in json_resp.items()
        if 'key' not in k.lower() and k not in excl_keys or k == 'usageKey '
    }
    return compact_info


def create_cropped_images_object(pred: object) -> io.BytesIO:
    """Create a zip file object with cropped predictions.

    Creates a zip file object containing cropped images and a json file with
    the labels and confidences.

    Args:
        pred (object): YOLOv5 Detections object.

    Returns:
        obj (io.BytesIO): A zip file object containing cropped images and a
        json file with the labels and confidences.

    """
    results = []
    cropped_imgs = pred.crop(save=False)
    with tempfile.NamedTemporaryFile(suffix='.zip') as f:
        with zipfile.ZipFile(f, mode='w') as zf:
            for cropped_img in cropped_imgs:
                with tempfile.NamedTemporaryFile() as cf:
                    im = Image.fromarray(cropped_img['im'])
                    im.save(cf, 'JPEG', quality=100, subsampling=0)
                    label = ' '.join(cropped_img['label'].split(' ')[:-1])
                    conf = cropped_img['label'].split(' ')[-1]
                    img_name = f'{label}/{uuid.uuid4()}.jpg'
                    zf.write(cf.name, arcname=img_name)
                    results.append({img_name: {'label': label, 'conf': conf}})
            with tempfile.NamedTemporaryFile() as rf:
                rf.write(json.dumps(results, indent=4).encode('utf-8'))
                rf.seek(0)
                zf.write(rf.name, arcname='results.json')
        f.flush()
        f.seek(0)
        obj = io.BytesIO(f.read())
    return obj


def results_dict(name: str, object_hash: str, labeled_image_url: Optional[str],
                 predictions: Dict[Union[int, str], Any], model_name: str,
                 model_version: str, page: str) -> dict:
    """Creates a dictionary with the prediction result(s) for the API call.

    Args:
        name (str): The name of the image.
        object_hash (str): The hash of the image.
        labeled_image_url (str): The url of the labeled image.
        predictions (list): A list of dictionaries with the predictions.
        model_name (str): The name of the model.
        model_version (str): The version of the model.
        page (str): The GitHub page of the model.

    Returns:
        dict: A dictionary with the results of the image classification.

    """
    return {
        'results': {
            'input_image': {
                'name': name,
                'hash': object_hash
            },
            'labeled_image_url': labeled_image_url,
            'predictions': predictions,
            'model': {
                'name': model_name,
                'version': model_version,
                'page': page
            }
        }
    }

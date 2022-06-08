#!/usr/bin/env python
# coding: utf-8

import io
import json
import tempfile
import uuid
import zipfile

import requests
from PIL import Image

from birdfsd_yolov5.model_utils.mongodb_helper import mongodb_db


def species_info(species_name):
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


def model_info(version):
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


def create_cropped_images_object(pred):
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


def _results_dict(name, _hash, labeled_image_url, predictions, _model_name,
                  _model_version, _page):
    return {
        'results': {
            'input_image': {
                'name': name,
                'hash': _hash
            },
            'labeled_image_url': labeled_image_url,
            'predictions': predictions,
            'model': {
                'name': _model_name,
                'version': _model_version,
                'page': _page
            }
        }
    }

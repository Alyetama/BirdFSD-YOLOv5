#!/usr/bin/env python
# coding: utf-8

import hashlib
import json
import mimetypes
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image
from dotenv import load_dotenv
from fastapi import FastAPI, Response, UploadFile
from fastapi.responses import StreamingResponse
from rayim import rayim

from birdfsd_yolov5._api import api_helpers, api_utils
from birdfsd_yolov5.model_utils.mongodb_helper import mongodb_db

# ----------------------------------------------------------------------------

load_dotenv()

s3 = api_utils.create_s3_client(api_s3=False)
api_s3 = api_utils.create_s3_client(api_s3=True)
db = mongodb_db()

model_version, model_name, model_weights, model = api_utils.init_model(s3)

if os.getenv('MODEL_REPO'):
    page = f'{os.getenv("MODEL_REPO")}/releases/tag/{model_version}'
else:
    page = None

app = FastAPI()

# ----------------------------------------------------------------------------


class PrettyJSONResponse(Response):
    media_type = "application/json"

    @staticmethod
    def render(content: Any, indent=4) -> bytes:
        return json.dumps(content,
                          ensure_ascii=False,
                          allow_nan=False,
                          indent=indent).encode('utf-8')


# ----------------------------------------------------------------------------


@app.get('/ping', status_code=200)
def ping():
    return 'Pong!'


# ----------------------------------------------------------------------------


@app.get("/model")
def get_model_info(version: str = 'latest'):
    model_obj = api_helpers.model_info(version)
    return PrettyJSONResponse(status_code=200, content=model_obj)


# ----------------------------------------------------------------------------


@app.post("/predict")
def predict_endpoint(file: UploadFile,
                     download: bool = False,
                     download_cropped: bool = False):
    image = Image.open(file.file)
    _hash = hashlib.md5(file.file.read()).hexdigest()

    content_type = mimetypes.guess_type(file.filename)[0]

    pred = model(image)
    pred_results = pd.concat(pred.pandas().xyxyn).T.to_dict()

    if not pred_results:
        res = api_helpers._results_dict(file.filename, _hash, None,
                                        pred_results, model_name,
                                        model_version, page)
        return PrettyJSONResponse(status_code=200, content=res)

    if download_cropped:
        obj = api_helpers.create_cropped_images_object(pred)
        return StreamingResponse(obj,
                                 status_code=200,
                                 media_type='application/zip')

    for k in pred_results:

        K = pred_results[k]
        K.pop('class')
        K.update({'confidence': round(K['confidence'], 4)})
        bbox = {}
        for _k in ['xmin', 'ymin', 'xmax', 'ymax']:
            bbox.update({_k: K[_k]})
            K.pop(_k)
        K['bbox'] = bbox
        K['species_info'] = {'gbif': api_helpers.species_info(K['name'])}

    im = Image.fromarray(pred.render()[0])

    with tempfile.NamedTemporaryFile() as f:
        im.save(f, format=content_type.split('/')[1])

        _ = rayim.compress(f.name, to_jpeg=True)
        f.seek(0)

        if download:
            return Response(f.read(), status_code=200, media_type=content_type)

        length = Path(f.name).stat().st_size
        out_file = f'{str(uuid.uuid4()).split("-")[-1]}.jpg'

        api_s3.put_object(bucket_name='api',
                          object_name=out_file,
                          data=f,
                          length=length,
                          content_type=content_type)

    url = f'https://{os.environ["API_S3_ENDPOINT"]}/api/{out_file}'

    res = api_helpers._results_dict(file.filename, _hash, url, pred_results,
                                    model_name, model_version, page)

    return PrettyJSONResponse(status_code=200, content=res)

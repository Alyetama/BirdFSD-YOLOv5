# -*- coding: utf-8 -*-
"""This module contains the API endpoints for the model."""

import hashlib
import json
import mimetypes
import os
import tempfile
import uuid
from pathlib import Path
from typing import Any, Union

import pandas as pd
import uvicorn
from PIL import Image
from dotenv import load_dotenv
from fastapi import FastAPI, Response, UploadFile
from fastapi.responses import StreamingResponse
from rayim import rayim

from birdfsd_yolov5.api import model_utils, api_utils
from birdfsd_yolov5.model_utils.mongodb_helper import mongodb_db

# ----------------------------------------------------------------------------

app = FastAPI()

# ----------------------------------------------------------------------------


class _PrettyJSONResponse(Response):
    media_type = "application/json"

    @staticmethod
    def render(content: Any, indent: int = 4) -> bytes:
        return json.dumps(content,
                          ensure_ascii=False,
                          allow_nan=False,
                          indent=indent).encode('utf-8')


# ----------------------------------------------------------------------------


@app.get('/ping', status_code=200)
def _ping() -> str:
    return 'Pong!'


# ----------------------------------------------------------------------------


@app.get('/model', status_code=200)
def get_model_info(version: str = 'latest') -> _PrettyJSONResponse:
    """Returns the info of a specific version of the model.

    Args:
        version: The model version.
    Returns:
        A string containing the model of the server.

    """
    model_obj = model_utils.model_info(version)
    return _PrettyJSONResponse(content=model_obj)


# ----------------------------------------------------------------------------


@app.post("/predict")
def predict_endpoint(file: UploadFile,
                     download: bool = False,
                     download_cropped: bool = False
                     ) -> Union[Response, StreamingResponse, str]:
    """Uses the backend model to predict the class of a given input image.

    Args:
        file (UploadFile): The UploadFile object of the image to be classified.
        download (bool): If True, the response is the labeled image bytes data.
        download_cropped (bool): If True, the response is a ZIP file with 
            subdirectories of predicted class(es) containing cropped image(s) 
            of the prediction(s).

    Returns:
        A JSON string response with predictions (class, bbox, confidence), a 
        URL to the labeled image, and information about the predicted species.

        If `download` is set to True, the response is the labeled image bytes 
        data.

        If `download_cropped` is set to True, the response is a ZIP file with 
        subdirectories of predicted class(es) containing cropped image(s) 
        of the prediction(s) and a JSON file with the information about each 
        prediction.

    """
    image = Image.open(file.file)
    obj_hash = hashlib.md5(file.file.read()).hexdigest()

    content_type = mimetypes.guess_type(file.filename)[0]

    pred = model(image)
    pred_results = pd.concat(pred.pandas().xyxyn).T.to_dict()

    if not pred_results:
        res = api_utils.results_dict(file.filename, obj_hash, None,
                                     pred_results, model_name, model_version,
                                     page)
        return _PrettyJSONResponse(content=res)

    if download_cropped:
        obj = api_utils.create_cropped_images_object(pred)
        return StreamingResponse(obj, media_type='application/zip')

    for k in pred_results:

        K = pred_results[k]
        K.pop('class')
        K.update({'confidence': round(K['confidence'], 4)})
        bbox = {}
        for _k in ['xmin', 'ymin', 'xmax', 'ymax']:
            bbox.update({_k: K[_k]})
            K.pop(_k)
        K['bbox'] = bbox
        K['species_info'] = {'gbif': api_utils.species_info(K['name'])}

    im = Image.fromarray(pred.render()[0])

    with tempfile.NamedTemporaryFile() as f:
        im.save(f, format=content_type.split('/')[1])

        _ = rayim.compress(f.name, to_jpeg=True)
        f.seek(0)

        if download:
            return Response(f.read(), media_type=content_type)

        length = Path(f.name).stat().st_size
        out_file = f'{str(uuid.uuid4()).split("-")[-1]}.jpg'

        api_s3.put_object(bucket_name='api',
                          object_name=out_file,
                          data=f,
                          length=length,
                          content_type=content_type)

    url = f'https://{os.environ["API_S3_ENDPOINT"]}/api/{out_file}'

    res = api_utils.results_dict(file.filename, obj_hash, url, pred_results,
                                 model_name, model_version, page)

    return _PrettyJSONResponse(content=res)


if __name__ == '__main__':
    load_dotenv()

    s3 = api_utils.create_s3_client()
    api_s3 = api_utils.create_s3_client(api_s3=True)
    db = mongodb_db()

    model_version, model_name, model_weights, model = \
        model_utils.init_model(s3)

    if os.getenv('MODEL_REPO'):
        page = f'{os.getenv("MODEL_REPO")}/releases/tag/{model_version}'
    else:
        page = None

    uvicorn.run(app, host='127.0.0.1', port=8000)  # noqa

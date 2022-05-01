#!/usr/bin/env python
# coding: utf-8

import argparse
import json

from dotenv import load_dotenv
from loguru import logger

try:
    from . import mongodb_helper, minio_helper, handlers
except ImportError:
    import mongodb_helper, minio_helper, handlers


class ModelVersionDoesNotExist(Exception):
    pass


class DownloadModelWeights:

    def __init__(self, model_version, output='best.pt'):
        self.model_version = model_version
        self.output = output

    def get_weights(self, skip_download=False):
        handlers.catch_keyboard_interrupt()

        db = mongodb_helper.mongodb_db()
        minio = minio_helper.MinIO()

        if self.model_version == 'latest':
            latest_model_ts = max(db.model.find().distinct('added_on'))
            model_document = db.model.find_one({'added_on': latest_model_ts})
        else:
            model_document = db.model.find_one({'version': self.model_version})
        if not model_document:
            avail_models = db.model.find().distinct('version')
            raise ModelVersionDoesNotExist(
                f'The model `{self.model_version}` does not exist! '
                f'\nAvailable models: {json.dumps(avail_models, indent=4)}')

        model_object_name = f'{model_document["version"]}.pt'
        weights_url = minio.get_presigned_download_url(
            bucket_name='model', object_name=model_object_name)

        if skip_download:
            logger.debug(f'Download URL: {weights_url}')
            return self.output, weights_url

        logger.debug(f'Downloading {model_object_name}...')
        minio.download(bucket_name='model',
                       object_name=model_object_name,
                       destination=self.output)

        logger.debug(f'\n\nModel version: {model_document["version"]}')
        logger.debug(f'Model weights file: {self.output}')
        return self.output, weights_url, model_document["version"]


if __name__ == '__main__':
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument('-v',
                        '--model-version',
                        help='Model version',
                        type=str,
                        required=True)
    parser.add_argument('-o',
                        '--output',
                        default='best.pt',
                        help='Output file name',
                        type=str)
    parser.add_argument('--skip-download',
                        action='store_true',
                        help='Return the download URL without downloading '
                        'the file')
    args = parser.parse_args()

    dmw = DownloadModelWeights(model_version=args.model_version,
                               output=args.output)
    dmw.get_weights(args.skip_download)

#!/usr/bin/env python
# coding: utf-8

import argparse
import json

from dotenv import load_dotenv
from loguru import logger

try:
    from .mongodb_helper import mongodb_db
    from .minio_helper import MinIO
except ImportError:
    from mongodb_helper import mongodb_db
    from minio_helper import MinIO


class ModelVersionDoesNotExist(Exception):
    pass


class DownloadModelWeights:

    def __init__(self, model_version, output='best.pt'):
        self.model_version = model_version
        self.output = output

    def get_weights(self, skip_download=False):
        db = mongodb_db()
        minio = MinIO()

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

        weights_url = minio.get_presigned_download_url(
            bucket_name='model', object_name=model_document['version'])

        if skip_download:
            logger.debug(f'Download URL: {weights_url}')
            return self.output, weights_url

        _ = minio.download(bucket_name='model',
                           object_name=model_document['version'],
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
                        help='Return the downloads URL without downloading '
                        'the file')
    args = parser.parse_args()

    dmw = DownloadModelWeights(model_version=args.model_version,
                               output=args.output)
    _, _ = dmw.get_weights(args.skip_download)

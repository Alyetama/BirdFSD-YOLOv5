#!/usr/bin/env python
# coding: utf-8

import argparse
import json
from datetime import timedelta
from pathlib import Path

import requests
from dotenv import load_dotenv
from loguru import logger

from birdfsd_yolov5.model_utils import (mongodb_helper, s3_helper, handlers,
                                        utils)


class ModelVersionDoesNotExist(Exception):
    """Raised when the model version does not exist in the database."""


class DownloadModelWeights:

    def __init__(self, model_version: str, output: str = 'best.pt'):
        """Method to initialize the class.

        Args:
            model_version (str): The version of the model to be saved.
            output (str): The name of the file to save the model to.

        """
        self.model_version = model_version
        self.output = output

    def get_weights(self,
                    skip_download: bool = False,
                    object_name_only: bool = False):
        """Get the weights for a given model version.

        Args:
            skip_download (bool): If True, the function will return the 
                download URL instead of downloading the weights file.
            object_name_only (bool): If True, the function will return the
                object name of the weights file instead of downloading the
                weights file.

        Returns:
            str: The path to the weights file.
            str: The download URL for the weights file.
            str: The model version.

        """
        handlers.catch_keyboard_interrupt()

        db = mongodb_helper.mongodb_db()
        s3 = s3_helper.S3()

        if self.model_version == 'latest':
            try:
                latest_model_ts = max(db.model.find().distinct('added_on'))
            except ValueError:
                default_pretrained_weights = 'md_v5a.0.0'
                logger.info('Could not find any saved model. Using the '
                            'default pretrained model: '
                            f'`{default_pretrained_weights}`...')
                weights_url = 'https://github.com/microsoft/CameraTraps/releases/download/v5.0/md_v5a.0.0.pt'  # noqa: E501
                r = requests.get(weights_url)
                with open(self.output, 'wb') as f:
                    f.write(r.content)
                logger.info(f'Saved model weights to: `{self.output}`')
                return self.output, weights_url, default_pretrained_weights

            model_document = db.model.find_one({'added_on': latest_model_ts})
        else:
            model_document = db.model.find_one({'version': self.model_version})
        if not model_document:
            avail_models = db.model.find().distinct('version')
            raise ModelVersionDoesNotExist(
                f'The model `{self.model_version}` does not exist! '
                f'\nAvailable models: {json.dumps(avail_models, indent=4)}')
        model_version = model_document["version"]
        model_object_name = f'{model_document["name"]}-v{model_version}.pt'
        if object_name_only:
            print(model_object_name)
            return model_object_name

        weights_url = s3.client.presigned_get_object(
            'model', model_object_name, expires=timedelta(hours=6))

        if skip_download:
            logger.debug(f'Download URL: {weights_url}')
            return self.output, weights_url, Path(model_object_name).stem
        logger.debug(f'Downloading {model_object_name}...')

        utils.requests_download(weights_url, self.output)

        logger.debug(f'\n\nModel version: {model_version}')
        logger.debug(f'Model weights file: {self.output}')

        return self.output, weights_url, Path(model_object_name).stem


def _opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v',
                        '--model-version',
                        help='Model version [x.y.z*]',
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
    parser.add_argument('-n',
                        '--object-name-only',
                        action='store_true',
                        help='Return the weights objects name then exit')
    return parser.parse_args()


if __name__ == '__main__':
    load_dotenv()
    args = _opts()

    dmw = DownloadModelWeights(model_version=args.model_version,
                               output=args.output)
    dmw.get_weights(skip_download=args.skip_download,
                    object_name_only=args.object_name_only)

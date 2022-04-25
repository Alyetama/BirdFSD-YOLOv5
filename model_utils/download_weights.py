#!/usr/bin/env python
# coding: utf-8

import argparse
import json
import os
import shlex
import subprocess

from dotenv import load_dotenv
from loguru import logger

try:
    from .mongodb_helper import mongodb_db
except ImportError:
    from mongodb_helper import mongodb_db


class ModelVersionDoesNotExist(Exception):
    pass


class DownloadModelWeights:
    def __init__(self, model_version, output='best.pt'):
        self.model_version = model_version
        self.output = output

    def get_weights(self, skip_download=False):
        db = mongodb_db()

        if args.model_version == 'latest':
            latest_model_ts = max(db.model.find().distinct('added_on'))
            model_document = db.model.find_one({'added_on': latest_model_ts})
        else:
            model_document = db.model.find_one({'version': self.model_version})
        if not model_document:
            avail_models = db.model.find().distinct('version')
            raise ModelVersionDoesNotExist(
                f'The model `{self.model_version}` does not exist! '
                f'\nAvailable models: {json.dumps(avail_models, indent=4)}'
            )
        weights_url = model_document['weights']
        logger.debug(weights_url)

        if not skip_download:
            download_cmd = f'curl -X GET -u ' \
            f'\"{os.environ["MINISERVE_USERNAME"]}:' \
            f'{os.environ["MINISERVE_RAW_PASSWORD"]}\" "{weights_url}" ' \
            f'--output {self.output}'

            p = subprocess.run(shlex.split(download_cmd),
                               shell=False,
                               check=True,
                               capture_output=True,
                               text=True)

            logger.debug(f'stderr: {p.stderr}')
            logger.debug(f'returncode: {p.returncode}')

        return weights_url


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

    dmw = DownloadModelWeights(
        model_version=args.model_version,
        output=args.output)
    _ = dmw.get_weights(args.skip_download)


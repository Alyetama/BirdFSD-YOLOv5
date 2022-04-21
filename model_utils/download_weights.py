#!/usr/bin/env python
# coding: utf-8

import argparse
import json
import os
import shlex
import subprocess

from dotenv import load_dotenv
from loguru import logger

from mongodb_helper import mongodb_db


class ModelVersionDoesNotExist(Exception):
    pass


def main():
    db = mongodb_db()

    model_document = db.model.find_one({'version': args.model_version})
    if not model_document:
        avail_models = db.model.find().distinct('version')
        raise ModelVersionDoesNotExist(
            f'The model `{args.model_version}` does not exist! '
            f'\nAvailable models: {json.dumps(avail_models, indent=4)}'
        )
    weights_url = model_document['weights']
    logger.debug(weights_url)

    download_cmd = f'curl -X GET -u \"{os.environ["MINISERVE_USERNAME"]}:' \
    f'{os.environ["MINISERVE_RAW_PASSWORD"]}\" "{weights_url}" ' \
    f'--output {args.output}'

    p = subprocess.run(shlex.split(download_cmd),
                       shell=False,
                       check=True,
                       capture_output=True,
                       text=True)

    logger.debug(f'stderr: {p.stderr}')
    logger.debug(f'returncode: {p.returncode}')


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
    args = parser.parse_args()

    main()

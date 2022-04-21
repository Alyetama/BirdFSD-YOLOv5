#!/usr/bin/env python
# coding: utf-8

import argparse
import os
import shlex
import subprocess

from dotenv import load_dotenv
from loguru import logger

from mongodb_helper import mongodb_db


def main():
    db = mongodb_db()

    weights_url = db.model.find_one({'version': args.model_version})['weights']
    logger.debug(weights_url)

    download_cmd = f'curl -X GET -u \"{os.environ["MINISERVE_USERNAME"]}:' \
    f'{os.environ["MINISERVE_RAW_PASSWORD"]}\" "{weights_url}" ' \
    f'--output {args.output}'

    p = subprocess.run(shlex.split(download_cmd),
                       shell=False,
                       check=True,
                       capture_output=True,
                       text=True)

    logger.debug(f'stdout: {p.stdout}')
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

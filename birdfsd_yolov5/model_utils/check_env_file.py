#!/usr/bin/env python
# coding: utf-8

import argparse
import re
from getpass import getpass
from pathlib import Path

from loguru import logger


def check_env_file(env_file):
    with open(env_file) as f:
        lines = f.readlines()

    for n, line in enumerate(lines):
        if line.startswith('LS_HOST'):
            lines[n] = line.rstrip().rstrip('/') + '\n'
        elif line.startswith('DB_CONNECTION_STRING'):
            if '<password>' in line:
                logger.error(
                    'You forgot to replace the <password> field in your '
                    'MongoDB connection string!')
                mongodb_passwd = getpass('MongoDB database password: ')
                lines[n] = line.replace('<password>', mongodb_passwd)
        elif line.startswith('S3_ENDPOINT'):
            lines[n] = re.sub(r'https?:\/\/', '',
                              line).rstrip().rstrip('/') + '\n'

    with open(env_file, 'w') as f:
        f.writelines(lines)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-e',
                        '--env-file',
                        help='Path to the `.env` file',
                        type=str,
                        required=True)
    args = parser.parse_args()

    if not Path(args.env_file).exists():
        raise FileNotFoundError(
            'The `.env` file path you provided does not exist!')

    check_env_file(args.env_file)

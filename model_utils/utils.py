#!/usr/bin/env python
# coding: utf-8

import argparse
import functools
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import gnupg
import requests
from dotenv import load_dotenv
from loguru import logger
from minio.error import S3Error
from requests.structures import CaseInsensitiveDict
from tqdm.auto import tqdm

try:
    from . import handlers, minio_helper
except ImportError:
    import handlers
    import minio_helper


def add_logger(current_file: str) -> str:
    Path('logs').mkdir(exist_ok=True)
    ts = datetime.now().strftime('%m-%d-%Y_%H.%M.%S')
    logs_file = f'logs/{Path(current_file).stem}_{ts}.log'
    logger.add(logs_file)
    return logs_file


def upload_logs(logs_file: str) -> None:
    minio = minio_helper.MinIO()
    try:
        logger.debug('Uploading logs...')
        resp = minio.upload('logs', logs_file)
        logger.debug(f'Uploaded log file: `{resp.object_name}`')
    except S3Error as e:
        logger.error('Could not upload logs file!')
        logger.error(e)
    return


def requests_download(url: str, filename: str) -> None:
    """https://stackoverflow.com/a/63831344"""
    handlers.catch_keyboard_interrupt()
    r = requests.get(url, stream=True, allow_redirects=True)
    if r.status_code != 200:
        r.raise_for_status()
        raise RuntimeError(f'Returned status code: {r.status_code}')
    file_size = int(r.headers.get('Content-Length', 0))
    r.raw.read = functools.partial(r.raw.read, decode_content=True)
    with tqdm.wrapattr(r.raw,
                       'read',
                       total=file_size,
                       desc='Download progress') as r_raw:
        with open(filename, 'wb') as f:
            shutil.copyfileobj(r_raw, f)
    return


def api_request(url: str, method: str = 'get', data: dict = None) -> dict:
    headers = CaseInsensitiveDict()
    headers['Content-type'] = 'application/json'
    headers['Authorization'] = f'Token {os.environ["TOKEN"]}'
    if method == 'get':
        resp = requests.get(url, headers=headers)
    elif method == 'post':
        resp = requests.post(url, headers=headers, data=json.dumps(data))
    elif method == 'patch':
        resp = requests.patch(url, headers=headers, data=json.dumps(data))
    return resp.json()  # noqa


def get_project_ids(exclude_ids: str = None) -> str:
    projects = api_request(
        f'{os.environ["LS_HOST"]}/api/projects?page_size=10000')
    project_ids = sorted([project['id'] for project in projects['results']])
    project_ids = [str(p) for p in project_ids]
    if exclude_ids:
        exclude_ids = [p for p in exclude_ids.split(',')]
        project_ids = [p for p in project_ids if p not in exclude_ids]
    return ','.join(project_ids)


def opts() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('--get-project-ids',
                        help='Get all project ids',
                        action='store_true')
    parser.add_argument('--exclude-ids',
                        help='Comma-separated list of project ids to exclude',
                        type=str)
    return parser.parse_args()


def encrypt_file(file: str) -> str:
    gpg = gnupg.GPG()
    key_fp = [
        k for k in gpg.list_keys() if k['keyid'] == os.environ['GPG_KEY_ID']
    ][0]['fingerprint']
    output_file = Path(file).with_suffix(f'{Path(file).suffix}.gpg')
    with open(file, 'rb') as f:
        _ = gpg.encrypt_file(f, recipients=key_fp, output=output_file)
    return output_file


if __name__ == '__main__':
    args = opts()
    load_dotenv()
    if args.get_project_ids:
        print(get_project_ids(exclude_ids=args.exclude_ids))

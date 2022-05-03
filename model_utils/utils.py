#!/usr/bin/env python
# coding: utf-8

import functools
import shutil
from datetime import datetime
from pathlib import Path

import requests
from loguru import logger
from minio.error import S3Error
from tqdm.auto import tqdm

try:
    from . import handlers, minio_helper
except ImportError:
    import handlers, minio_helper


def add_logger(current_file):
    Path('logs').mkdir(exist_ok=True)
    ts = datetime.now().strftime('%m-%d-%Y_%H.%M.%S')
    logs_file = f'logs/{Path(current_file).stem}_{ts}.log'
    logger.add(logs_file)
    return logs_file


def upload_logs(logs_file):
    minio = minio_helper.MinIO()
    try:
        logger.debug('Uploading logs...')
        resp = minio.upload('logs', logs_file)
        logger.debug(f'Uploaded log file: `{resp.object_name}`')
    except S3Error as e:
        logger.error('Could not upload logs file!')
        logger.error(e)
    return


def requests_download(url, filename):
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

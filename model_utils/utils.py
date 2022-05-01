#!/usr/bin/env python
# coding: utf-8

from datetime import datetime
from pathlib import Path

from loguru import logger
from minio.error import S3Error

try:
    from . import minio_helper
except ImportError:
    import minio_helper


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

#!/usr/bin/env python
# coding: utf-8

import os
import sys
import textwrap
from datetime import timedelta
from pathlib import Path

from dotenv import load_dotenv
from minio import Minio

try:
    from . import utils
except ImportError:
    import utils


class BucketDoesNotExist(Exception):
    pass


class MinIO:

    def __init__(self):
        self.client = Minio(os.environ['MINIO_ENDPOINT'],
                            access_key=os.environ['MINIO_ACCESS_KEY'],
                            secret_key=os.environ['MINIO_SECRET_KEY'],
                            region=os.environ['MINIO_REGION'])

    def upload(self, bucket_name, file_path):
        file = Path(file_path)
        return self.client.fput_object(bucket_name, file.name, file)

    def download(self, bucket_name, object_name, dest=None):
        if not dest:
            dest = object_name
        return self.client.fget_object(bucket_name, object_name, dest)

    def get_model_weights(self, model_version: str = 'latest') -> str:
        objects = list(self.client.list_objects('model'))
        if model_version == 'latest':
            latest_ts = max([obj.last_modified for obj in objects])
            latest_model_object = [
                obj for obj in objects if obj.last_modified == latest_ts
            ][0]
            return latest_model_object.object_name
        else:
            for obj in objects:
                if obj.object_name.endswith(model_version):
                    return obj.object_name

    def get_dataset(self, object_name=None, dest=None):
        if not object_name:
            objs = list(self.client.list_objects('dataset'))
            latest_ts = max([o.last_modified for o in objs if o.last_modified])
            latest_obj = [o for o in objs if o.last_modified == latest_ts][0]
            object_name = latest_obj.object_name

        presigned_url = self.client.presigned_get_object(
            'dataset', object_name, expires=timedelta(hours=6))

        utils.requests_download(presigned_url, object_name)
        return object_name


if __name__ == '__main__':
    if '--help' in sys.argv or '-h' in sys.argv:
        message = textwrap.dedent('''\
            - Configuration:
                $ aws configure
                    # AWS Access Key ID: ...
                    # AWS Secret Access Key: ...
                    # Default region name: us-east-1
                    # Default output format: <hit ENTER>
                $ aws configure set default.s3.signature_version s3v4''')
        print(message)
    if '--download-dataset' in sys.argv:
        load_dotenv()
        _ = MinIO().get_dataset()

#!/usr/bin/env python
# coding: utf-8

import os
import sys
import textwrap
from pathlib import Path

from minio import Minio


class MinIO:

    def __init__(self):
        self.client = Minio(os.environ['MINIO_ENDPOINT'],
                            access_key=os.environ['MINIO_ACCESS_KEY'],
                            secret_key=os.environ['MINIO_SECRET_KEY'],
                            region=os.environ['MINIO_REGION'])

    def upload(self, bucket_name, file_path):
        file = Path(file_path)
        return self.client.fput_object(bucket_name, file.name, file)

    def download(self, bucket_name, object_name, destination=None):
        if not destination:
            destination = object_name
        return self.client.fget_object(bucket_name, object_name, destination)

    def get_presigned_download_url(self, bucket_name, object_name):
        return self.client.presigned_get_object(bucket_name, object_name)

    def get_model_weights(self, model_version: str = 'latest') -> str:
        objects = list(self.client.list_objects('model'))
        if model_version == 'latest':
            latest_ts = max([obj.last_modified for obj in objects])
            latest_model_list = [
                obj for obj in objects if obj.last_modified == latest_ts
            ]
            assert len(latest_model_list) == 1
            latest_model_object = latest_model_list[0]
            return latest_model_object.object_name
        else:
            for obj in objects:
                if obj.object_name.endswith(model_version):
                    return obj.object_name


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
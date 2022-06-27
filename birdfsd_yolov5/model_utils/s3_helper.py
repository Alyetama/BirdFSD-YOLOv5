#!/usr/bin/env python
# coding: utf-8

import mimetypes
import os
import sys
import textwrap
from datetime import timedelta
from pathlib import Path
from typing import Union

from dotenv import load_dotenv
from minio import Minio
from minio.datatypes import Object
from minio.helpers import ObjectWriteResult

from birdfsd_yolov5.model_utils import utils


class BucketDoesNotExist(Exception):
    """Raised when the request bucket does not exist."""


class S3:
    """S3 client that uses MinIO as a backend."""

    def __init__(self):
        """Initialize the Minio client.

        Returns:
            None

        """
        self.client = Minio(os.environ['S3_ENDPOINT'],
                            access_key=os.environ['S3_ACCESS_KEY'],
                            secret_key=os.environ['S3_SECRET_KEY'],
                            region=os.environ['S3_REGION'])

    def upload(self,
               bucket_name: str,
               file_path: str,
               public: bool = False,
               scheme: str = 'https',
               dest: str = None) -> Union[ObjectWriteResult, str]:
        """Uploads a file to an S3 bucket.

        Args:
            bucket_name (str): The name of the bucket to upload to.
            file_path (str): The path to the file to upload.
            public (bool): True if the file is uploaded to a publicly
                accessible bucket.
            scheme (str): The scheme to use for the URL.
            dest (str): The destination path for the file.

        Returns:
            str: The URL of the uploaded file.

        """
        file = Path(file_path)
        if not dest:
            dest = file.name
        content_type = mimetypes.guess_type(file_path)
        if content_type[0]:
            content_type = content_type[0]
        else:
            content_type = 'application/octet-stream'
        res = self.client.fput_object(bucket_name=bucket_name,
                                      object_name=dest,
                                      file_path=file,
                                      content_type=content_type)
        if public:
            if 'http' not in os.environ[
                    "S3_ENDPOINT"] or 'https' not in os.environ["S3_ENDPOINT"]:
                domain = f'{scheme}://{os.environ["S3_ENDPOINT"]}'
            else:
                domain = os.environ["S3_ENDPOINT"]
            if dest:
                return f'{domain}/{bucket_name}/{dest}'
            else:
                return f'{domain}/{bucket_name}/{file.name}'
        else:
            return res

    def download(self,
                 bucket_name: str,
                 object_name: str,
                 dest: str = None) -> Object:
        """Downloads an object from the bucket.

        Args:
            bucket_name (str): The name of the bucket.
            object_name (str): The name of the object.
            dest (str): The destination path to download the object.

        Returns:
            str: The destination path where the object was downloaded.

        """
        if not dest:
            dest = object_name
        return self.client.fget_object(bucket_name, object_name, dest)

    def get_model_weights(self, model_version: str = 'latest') -> str:
        """Get the model weights from the model bucket.

        Args:
            model_version: The version of the model to get.
                If 'latest', the latest version is returned.
                If a string, the model with that version is returned.
                If None, the latest version is returned.

        Returns:
            str: The name of the model object in the model bucket.

        """
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

    def get_dataset(self, object_name: str = None) -> str:
        """Retrieves the latest dataset from the S3 bucket.

        Args:
            object_name (str): The name of the object to retrieve. If not
                provided, the latest object will be retrieved.

        Returns:
            str: The name of the object retrieved.
            
        """
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
        _ = S3().get_dataset()

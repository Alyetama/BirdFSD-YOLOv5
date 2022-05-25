#!/usr/bin/env python
# coding: utf-8

import argparse
import copy
import datetime
import json
import re
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from pymongo.errors import DuplicateKeyError

from model_utils.mongodb_helper import mongodb_db
from model_utils.s3_helper import S3
from model_utils.utils import get_project_ids_str


class ModelVersionFormatError(Exception):
    """Exception raised when a model version is not in the correct format."""


class SyncModel:

    def __init__(self, project_ids: Optional[str], model_version: str,
                 model_name: str, run_path: str, classes_file: str,
                 weights_file: str, config_file: str, train_date: str) -> None:
        self.project_ids = project_ids
        self.model_name = model_name
        self.model_version = model_version
        self.run_path = run_path
        self.classes_file = classes_file
        self.weights_file = weights_file
        self.config_file = config_file
        self.train_date = train_date

    def check_version_number_format(self) -> str:
        """Check the format of the model version.

        Notes:
            The version number should be in the format of 'vX.Y.Z' or
            'vX.Y.Z-alpha.N'.

        Returns:
            re.Match: The match object if the version is valid, None otherwise.

        Raises:
            ModelVersionFormatError: If the version is not valid.
        """
        model_version_number = 'v' + self.model_version.split(
            self.model_name)[1].split('-v')[1]
        if 'alpha' in model_version_number:
            match = re.match(r'^v\d+\.\d+\.\d+-alpha.*$', model_version_number)
        else:
            match = re.match(r'^v\d+\.\d+\.\d+$', model_version_number)

        if not match:
            raise ModelVersionFormatError(
                'Model versions is not formatted correctly!')
        return model_version_number

    def add_new_version(self) -> dict:
        """This function updates the database with the latest version of the
        dataset. It first gets the labels from the database, then adds the new
        version of the model to the `model` collection in the database.

        Returns:
            model (dict): The model that was added to the database.
        """

        model_version_number = self.check_version_number_format()

        db = mongodb_db()

        s3 = S3()
        weights_dst = Path(self.weights_file).name.replace('-best_weights', '')
        s3_resp = s3.upload(bucket_name='model',
                            file_path=self.weights_file,
                            dest=weights_dst)
        print(f'Uploaded object name: {s3_resp.object_name}')

        with open(self.classes_file) as j:
            labels_freq = json.load(j)

        if self.project_ids:
            project_ids = self.project_ids
        else:
            project_ids = get_project_ids_str()

        with open(self.config_file) as j:
            train_config = json.load(j)

        train_date = [int(x) for x in self.train_date.split('-')]

        model = {
            '_id': self.model_version,
            'name': self.model_version.split('-v')[0],
            'version': self.model_version.split('-v')[1],
            'projects': project_ids,
            'labels': labels_freq,
            'added_on': datetime.datetime.today(),
            'trained_on': datetime.datetime(*train_date),
            'wandb_run_path': self.run_path,
            'number_of_classes': len(labels_freq),
            'number_of_annotations': sum(list(labels_freq.values())),
            'train_config': train_config
        }

        try:
            db.model.insert_one(model)
        except DuplicateKeyError:
            db.model.delete_one({'_id': model['_id']})
            db.model.insert_one(model)

        serializable_model = copy.deepcopy(model)
        serializable_model.update({
            'added_on': str(model['added_on']),
            'trained_on': str(model['trained_on'])
        })
        print(json.dumps(serializable_model, indent=4))

        return model


if __name__ == '__main__':
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-p',
        '--project-ids',
        help='Comma-seperated project ids. If empty, it will select all '
        'projects',
        type=str)
    parser.add_argument('-v',
                        '--model-version',
                        help='Model version ([NAME-x.y.z])',
                        type=str,
                        required=True)
    parser.add_argument('-n',
                        '--model-name',
                        help='Model name ([NAME]-x.y.z)',
                        type=str,
                        required=True)
    parser.add_argument('-r',
                        '--run-path',
                        help='W&B run path',
                        type=str,
                        required=True)
    parser.add_argument('-c',
                        '--classes-file',
                        help='`*-classes.json` file from the release',
                        type=str,
                        required=True)
    parser.add_argument('-w',
                        '--weights-file',
                        help='`*-best_weights.pt` file from the release',
                        type=str,
                        required=True)
    parser.add_argument('-C',
                        '--config-file',
                        help='`*-config.json` file from the release',
                        type=str,
                        required=True)
    parser.add_argument('-d',
                        '--train-date',
                        help='Date on which the current model was trained ('
                        'yyyy-MM-dd)',
                        type=str,
                        required=True)
    args = parser.parse_args()

    sync_model = SyncModel(project_ids=args.project_ids,
                           model_version=args.model_version,
                           model_name=args.model_name,
                           run_path=args.run_path,
                           classes_file=args.classes_file,
                           weights_file=args.weights_file,
                           config_file=args.config_file,
                           train_date=args.train_date)
    sync_model.add_new_version()

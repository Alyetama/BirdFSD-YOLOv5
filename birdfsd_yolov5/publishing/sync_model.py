#!/usr/bin/env python
# coding: utf-8

import argparse
import copy
import datetime
import json
import re
from pathlib import Path

from dotenv import load_dotenv

from birdfsd_yolov5.model_utils import mongodb_helper, s3_helper


class ModelVersionFormatError(Exception):
    """Exception raised when a model version is not in the correct format."""
    pass


class SyncModel:

    def __init__(self, model_version: str, model_name: str, run_path: str,
                 classes_file: str, weights_file: str, weights_enc: str,
                 config_file: str, train_date: str) -> None:
        self.model_name = model_name
        self.model_version = model_version
        self.run_path = run_path
        self.classes_file = classes_file
        self.weights_file = weights_file
        self.weights_enc = weights_enc
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
        """updates the database with the latest version.

        This method updates the database with the latest version of the
        model. It first gets the labels from the database, then adds the new
        version of the model to the `model` collection in the database.

        Returns:
            dict: The model that was added to the database.
        """

        _ = self.check_version_number_format()

        db = mongodb_helper.mongodb_db()

        s3 = s3_helper.S3()
        weights_dst = Path(self.weights_file).name.replace('-best_weights', '')
        s3_resp = s3.upload(bucket_name='model',
                            file_path=self.weights_file,
                            dest=weights_dst)
        print(f'Uploaded object name: {s3_resp.object_name}')

        with open(self.classes_file) as j:
            labels_freq = json.load(j)

        with open(self.config_file) as j:
            train_config = json.load(j)

        train_date = [int(x) for x in self.train_date.split('-')]

        if '-v' in self.model_version:
            split_by = '-v'
        elif '_v' in self.model_version:
            split_by = '_v'
        else:
            split_by = input(
                'Enter the split character between the model name '
                'and its version (e.g., "MODEL-vX.Y.Z": the split character '
                'here is "-"): ') + 'v'

        model = {
            '_id': self.model_version,
            'name': self.model_version.split(split_by)[0],
            'version': self.model_version.split(split_by)[1],
            'labels': labels_freq,
            'added_on': datetime.datetime.today(),
            'trained_on': datetime.datetime(*train_date),
            'wandb_run_path': self.run_path,
            'number_of_classes': len(labels_freq),
            'number_of_annotations': sum(list(labels_freq.values())),
            'train_config': train_config,
            'weights_enc': self.weights_enc
        }

        db.model.insert_one(model)

        serializable_model = copy.deepcopy(model)
        serializable_model.update({
            'added_on': str(model['added_on']),
            'trained_on': str(model['trained_on'])
        })
        print(json.dumps(serializable_model, indent=4))
        return model


def _opts():
    parser = argparse.ArgumentParser()
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
    parser.add_argument('-W',
                        '--weights-enc',
                        help='Encrypted weights github URL',
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
    return parser.parse_args()


if __name__ == '__main__':
    load_dotenv()
    args = _opts()

    sync_model = SyncModel(model_version=args.model_version,
                           model_name=args.model_name,
                           run_path=args.run_path,
                           classes_file=args.classes_file,
                           weights_file=args.weights_file,
                           weights_enc=args.weights_enc,
                           config_file=args.config_file,
                           train_date=args.train_date)
    sync_model.add_new_version()

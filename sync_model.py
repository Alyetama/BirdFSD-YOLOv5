#!/usr/bin/env python
# coding: utf-8

import argparse
import datetime
import os
import re
import shlex
import shutil
import subprocess
from typing import Union

import numpy as np
import pymongo
import wandb
from dotenv import load_dotenv
from pymongo.errors import DuplicateKeyError
from tqdm import tqdm

from model_utils.mongodb_helper import get_tasks_from_mongodb, mongodb_db


class ModelVersionFormatError(Exception):
    """Exception raised when a model version is not in the correct format."""


class SyncModel:

    def __init__(self, projects_id: str, model_version: str,
                 run_path: str) -> None:
        self.projects_id = projects_id
        self.model_version = model_version
        self.run_path = run_path

    def check_version_format(self) -> re.Match:
        """Check the format of the model version.

        Parameters
        ----------
        self : object
            The object to check the version of.

        Returns
        -------
        re.Match
            The match object if the version is valid, None otherwise.

        Raises
        ------
        ModelVersionFormatError
            If the version is not valid.

        Notes
        -----
        The version should be in the format of 'vX.Y.Z' or 'vX.Y.Z-alpha.N'.

        Examples
        --------
        >>> check_version_format('v1.0.0')
        <re.Match object; span=(0, 6), match='v1.0.0'>
        >>> check_version_format('v1.0.0-alpha.1')
        <re.Match object; span=(0, 14), match='v1.0.0-alpha.1'>
        >>> check_version_format('1.0.0')
        None
        """
        if 'alpha' in self.model_version:
            match = re.match(r'^v[0-9]+\.[0-9]+\.[0-9]+-alpha.*$',
                             self.model_version)
        else:
            match = re.match(r'^v[0-9]+\.[0-9]+\.[0-9]+$', self.model_version)

        if not match:
            raise ModelVersionFormatError(
                'Model versions is not formatted correctly!')
        return match

    def get_labels(self, db: pymongo.database.Database) -> list:
        """Get labels from MongoDB.
    
        Parameters
        ----------
        self : object
            An instance of the class.
        db : object
            An instance of the class.
        
        Returns
        -------
        labels : list
            A list of labels.
        """
        labels = []
        for project_id in tqdm(self.projects_id.split(','), desc='Projects'):
            data = get_tasks_from_mongodb(project_id,
                                          dump=False,
                                          json_min=True)
            for x in data:
                for label in x['label']:
                    labels.append(label['rectanglelabels'])
        return labels

    def get_weights(self) -> Union[str, None]:
        """This function downloads the best weights from the run specified by the run_path.
        It returns the path to the downloaded weights.
        
        Returns
        -------
        str
            The path to the downloaded weights.

        Raises
        ------
        FileNotFoundError
            If the selected run path does not have best weights file.
        """
        api = wandb.Api()
        for m in list(api.from_path(self.run_path).logged_artifacts()):
            if 'best' in m.aliases:
                return m.download()
            else:
                raise FileNotFoundError(
                    'Could not find best weights file in this run!')

    def add_new_version(self, db: pymongo.database.Database,
                        labels: list) -> dict:
        """Add a new model version to the database.

        Parameters
        ----------
        db : pymongo.database.Database
            The database to add the model to.
        labels : list
            A list of labels.

        Returns
        -------
        model : dict
            The model that was added to the database.
        """

        unique, counts = np.unique(labels, return_counts=True)
        labels_freq = {k: int(v) for k, v in np.asarray((unique, counts)).T}

        weights_path = self.get_weights()
        renamed_weights_fname = f'BirdFSD-YOLOv5-{self.model_version}.pt'
        os.rename(f'{weights_path}/best.pt',
                  f'{weights_path}/{renamed_weights_fname}')

        upload_cmd = f'curl -u {os.environ["MINISERVE_USERNAME"]}:' \
        f'{os.environ["MINISERVE_RAW_PASSWORD"]} -F ' \
        f'"path=@{weights_path}/{renamed_weights_fname}" ' \
        rf'{os.environ["MINISERVE_HOST"]}/upload\?path\=/'

        weights_url = f'{os.environ["MINISERVE_HOST"]}/{renamed_weights_fname}'

        p = subprocess.run(shlex.split(upload_cmd),
                           shell=False,
                           check=True,
                           capture_output=True,
                           text=True)

        model = {
            '_id': self.model_version,
            'version': f'BirdFSD-YOLOv5-{self.model_version}',
            'projects': self.projects_id,
            'labels': labels_freq,
            'weights': weights_url,
            'added_on': datetime.datetime.today(),
            'wandb_run_path': self.run_path,
            'number_of_classes': len(labels_freq),
            'number_of_annotations': sum(list(labels_freq.values()))
        }

        try:
            db.model.insert_one(model)
        except DuplicateKeyError:
            db.model.delete_one({'_id': model['_id']})
            db.model.insert_one(model)

        shutil.rmtree('artifacts')
        print(model['weights'])
        return model

    def update(self) -> None:
        """This function updates the database with the latest version of the
        dataset. It first gets the labels from the database, then adds the new
        version of the model to the `model` collection in the database.
        
        Returns
        -------
        None
        """
        self.check_version_format()
        db = mongodb_db()
        labels = self.get_labels(db)
        _ = self.add_new_version(db, labels)
        return


if __name__ == '__main__':
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument('-p',
                        '--projects-id',
                        help='Comma-seperated projects ID',
                        type=str,
                        default=os.environ['PROJECTS_ID'])
    parser.add_argument('-v',
                        '--model-version',
                        help='Model version',
                        type=str,
                        required=True)
    parser.add_argument('-r',
                        '--run-path',
                        help='W&B run path',
                        type=str,
                        required=True)
    args = parser.parse_args()

    sync_model = SyncModel(projects_id=args.projects_id,
                           model_version=args.model_version,
                           run_path=args.run_path)
    sync_model.update()

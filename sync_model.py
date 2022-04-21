#!/usr/bin/env python
# coding: utf-8

import argparse
import datetime
import os
import pickle
import re
import shutil

import bson
import numpy as np
import wandb
from dotenv import load_dotenv
from pymongo.errors import DuplicateKeyError
from tqdm import tqdm

from model_utils.mongodb_helper import get_tasks_from_mongodb, mongodb_db


class ModelVersionFormatError(Exception):
    pass


class SyncModel:

    def __init__(self, projects_id, model_version, run_path):
        self.projects_id = projects_id
        self.model_version = model_version
        self.run_path = run_path

    def check_version_format(self):
        if 'alpha' in self.model_version:
            match = re.match(r'^v[0-9]+\.[0-9]+\.[0-9]+-alpha.*$',
                             self.model_version)
        else:
            match = re.match(r'^v[0-9]+\.[0-9]+\.[0-9]+$', self.model_version)
        return match

    def get_labels(self, db):
        labels = []
        for project_id in tqdm(self.projects_id.split(','), desc='Projects'):
            data = get_tasks_from_mongodb(project_id,
                                          dump=False,
                                          json_min=True)
            for x in data:
                for label in x['label']:
                    labels.append(label['rectanglelabels'])
        return labels

    def get_weights(self):
        api = wandb.Api()
        for m in list(api.from_path(self.run_path).logged_artifacts()):
            if 'best' in m.aliases:
                return m.download()

    def add_new_version(self, db, labels):
        unique, counts = np.unique(labels, return_counts=True)

        labels_freq = {k: int(v) for k, v in np.asarray((unique, counts)).T}

        if not self.check_version_format():
            raise ModelVersionFormatError(
                'Model versions is not formatted correctly!')

        weights_path = f'{self.get_weights()}/best.pt'
        with open(weights_path, 'rb') as pt:
            weights = pt.read()

        model = {
            '_id': self.model_version,
            'version': f'BirdFSD-YOLOv5-{self.model_version}',
            'projects': self.projects_id,
            'labels': labels_freq,
            'weights': weights,
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
        return model

    def update(self):
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

#!/usr/bin/env python
# coding: utf-8

import os
import sys
from typing import Optional, Union

import xml.etree.ElementTree as ETree
from loguru import logger
from tqdm import tqdm

from birdfsd_yolov5.model_utils.utils import api_request, get_project_ids_str


class LabelDoesNotExist(Exception):
    pass


class MissingArgument(Exception):
    pass


def update_project_config_label(
    project_id: Union[int, str],
    label: str,
    change_to: Optional[str] = None,
    remove: bool = False,
    backup: bool = True,
    verbose: bool = False,
    dry_run: bool = False,
) -> Union[dict, str]:
    if not remove and not change_to:
        raise MissingArgument(
            'You need to select a label to replace the input label with or '
            'set `remove` to True to remove it.')
    project = api_request(f'{os.environ["LS_HOST"]}/api/projects/{project_id}')
    if verbose:
        logger.debug(
            f'(Project {project_id}) Config before: {project["label_config"]}')

    tree = ETree.fromstring(project['label_config'])
    root = tree.findall('.//')
    for label in tree.iter('Label'):
        if label.attrib['value'] == change_to:
            logger.warning(
                f'Label is already set to `{change_to}`. Skipping...')
    transformed_xml = None

    if change_to:
        transformed_xml = ETree.tostring(tree, encoding='unicode').replace(
            f'<Label value="{label}"', f'<Label value="{change_to}"')
    elif remove:
        for elem in tree.findall('.//View/RectangleLabels/Label'):
            if elem.attrib['value'] == label:
                tag_to_remove_list = [tag for tag in root if tag == elem]
                tag_to_remove = ETree.tostring(
                    tag_to_remove_list[0]).decode('utf-8')
                if remove:
                    transformed_xml = ETree.tostring(
                        tree,
                        encoding='unicode').replace(tag_to_remove, change_to)
                break

    if not transformed_xml:
        if sys.stdin.isatty():
            c = ('\033[31m', '\033[39m')
        else:
            c = ('', '')
        raise LabelDoesNotExist(
            f'\n{c[0]}ERROR: Could not find the label `{label}` in the '
            f'project configuration of project {project_id}!{c[1]}')

    if backup:
        with open('config.xml', 'w') as f:
            f.write(transformed_xml)

    if dry_run:
        return transformed_xml

    project.pop('created_by')
    project.update({'label_config': transformed_xml})

    updated_project = api_request(
        f'{os.environ["LS_HOST"]}/api/projects/{project_id}',
        method='patch',
        data=project)

    if verbose:
        logger.debug(f'(Project {project_id}) Config After: '
                     f'{updated_project["label_config"]}')
    return updated_project


def update_all_projects_config_label(label: str, **kwargs) -> None:
    project_ids = get_project_ids_str().split(',')

    for project_id in tqdm(project_ids):
        try:
            update_project_config_label(project_id, label, **kwargs)
        except LabelDoesNotExist as e:
            print(e)
            print(f'Skipped project: {project_id}...')

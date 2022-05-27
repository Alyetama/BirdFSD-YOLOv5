#!/usr/bin/env python
# coding: utf-8

from typing import List, Optional, Union

import wandb


def upload_weights_to_finished_run(
        run_path: str,
        weights_file: str,
        aliases: Optional[Union[List[str], str]] = None) -> None:
    """Uploads a model weights file to the artifacts of a finished wandb run.

    Args:
        run_path (str): The run path.
        weights_file (str): The path to the weights file.
        aliases (Optional[Union[List[str], str]]): A list of aliases to give 
            the weights artifact.

    Returns:
        None

    """
    run_path_dict = {
        k: v
        for k, v in zip(['entity', 'project', 'id'], run_path.split('/'))
    }

    with wandb.init(**run_path_dict) as run:
        artifact = wandb.Artifact(f'run_{run_path_dict["id"]}_model', 'model')
        artifact.add_file(weights_file)
        if isinstance(aliases, str):
            aliases = aliases.split(',')
        run.log_artifact(artifact, aliases=aliases)

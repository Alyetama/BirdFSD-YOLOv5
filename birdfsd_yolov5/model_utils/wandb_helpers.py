#!/usr/bin/env python
# coding: utf-8

import argparse
from pathlib import Path
from typing import Optional

import wandb


def upload_artifact(run_path: str,
                    artifact_type: str,
                    file_path: str,
                    artifact_name: Optional[str] = None,
                    aliases: Optional[list] = None) -> None:
    """Upload an artifact to an existing wandb run.

    Args:
        run_path (str): The wandb run path.
        artifact_type (str): The artifact type (e.g., dataset, model, etc.).
        file_path (str): The path to the file on the local disk.
        artifact_name (str): A human-readable name for the artifact. If None, 
            the file's base name is used.
    """
    if not artifact_name:
        artifact_name = Path(file_path).name

    entity, project, _id = run_path.split('/')

    with wandb.init(entity=entity, project=project, id=_id,
                    resume='allow') as run:
        artifact = wandb.Artifact(artifact_name, artifact_type)
        artifact.add_file(file_path)
        if aliases:
            run.log_artifact(artifact, aliases=aliases)
        else:
            run.log_artifact(artifact)


def add_f1_score(run_path: str) -> None:
    """Adds the F1 score to the run.
    Args:
        run_path (str): The wandb run path.
    """
    api = wandb.Api()
    run = api.run(run_path)
    p = run.summary['best/precision']
    r = run.summary['best/recall']
    f1_score = 2 * ((p * r) / (p + r))
    run.summary.update({'best/f1-score': f1_score})
    run.update()


def _get_all_runs_path(entity: str, project: str) -> list:
    """Returns a list of all runs for the given entity and project.
    Args:
        entity (str): The wandb entity name.
        project (str): The wandb project name in the selected entity.
    Returns:
        list: A list of all runs for the given project.
    """
    api = wandb.Api()
    runs = api.runs(f'{entity}/{project}')
    return ['/'.join(x.path) for x in runs]


def _opts() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--add-f1-score-by-run-path',
        help='The wandb run path to update with the calculated F1-score',
        type=str)
    return parser.parse_args()


if __name__ == '__main__':
    args = _opts()
    if args.add_f1_score_by_run_path:
        add_f1_score(args.add_f1_score_by_run_path)

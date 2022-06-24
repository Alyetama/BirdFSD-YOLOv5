#!/usr/bin/env python
# coding: utf-8

from pathlib import Path

import wandb


def upload_artifact(run_path: str,
                    artifact_type: str,
                    file_path: str,
                    artifact_name: str = None) -> None:
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
        artifact.add_file("dataset-YOLO-06-13-2022_19.55.50.tar")
        run.log_artifact(artifact)


def _add_f1_score(run_path: str):
    api = wandb.Api()
    run = api.run(run_path)
    p = run.summary['best/precision']
    r = run.summary['best/recall']
    f1_score = 2 * ((p * r) / (p + r))
    run.summary.update({'best/f1-score': f1_score})
    run.update()


def _get_all_runs_path(entity: str, project: str):
    api = wandb.Api()
    runs = api.runs(f'{entity}/{project}')
    return ['/'.join(x.path) for x in runs]
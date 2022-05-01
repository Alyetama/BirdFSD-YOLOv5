#!/usr/bin/env python
# coding: utf-8

import argparse
import base64
import json
import os
import shutil
import tarfile
from glob import glob
from pathlib import Path
from typing import Union

import requests
import wandb
from dotenv import load_dotenv
from loguru import logger

from model_utils import download_weights
from model_utils.utils import add_logger, upload_logs


def upload_image(image_path: str, imgbb_token: str) -> str:
    """Uploads an image to imgbb.com and returns the URL.

    Parameters
    ----------
    image_path : str
        Path to the image to be uploaded.
    imgbb_token : str
        API token for imgbb.com.

    Returns
    -------
    str
        URL of the uploaded image.
    """
    with open(image_path, 'rb') as f:
        url = 'https://api.imgbb.com/1/upload'
        payload = {
            'key': imgbb_token,
            'image': base64.b64encode(f.read()),
        }
        res = requests.post(url, payload)
    return res


def find_file(run: wandb.wandb_run.Run, fname: str) -> Union[tuple, None]:
    """Finds a file in a run and uploads it to imgbb.

    Parameters
    ----------
    run: wandb.wandb_run.Run
        A Run object.
    fname: str
        A string, the name of the file to be found.

    Returns
    -------
    tuple
        A tuple of the file object and the url of the uploaded image.
    """
    try:
        file = [x for x in list(run.files()) if fname in x.name][0]
    except IndexError:
        return
    file.download()
    imgbb_res = upload_image(file.name, os.environ['IMGBB_API_KEY'])
    return file, imgbb_res.json()['data']['url']


def get_assets(run: wandb.wandb_run.Run,
               output_name: str) -> Union[tuple, None]:
    """Downloads the best model from the run with the given ID,
    extracts the model weights, configuration, & classes and saves them to a
    file.

    Parameters
    ----------
    run: wandb.wandb_run.Run
        The run from which to extract information.
    output_name: str
        The name of the output file.

    Returns
    -------
    tuple
        The the config dictionary and the paths to the the config and classes
        files.
    """
    logger.debug(run)
    config_fname = f'{output_name}/{output_name}-config.json'
    cfg = run.config

    for k in [('val', -3), ('train', -3), ('path', -2)]:
        relative_path = '/'.join(Path(cfg['data_dict'][k[0]]).parts[k[1]:])
        cfg['data_dict'][k[0]] = relative_path

    logger.debug(cfg)
    print()
    with open(config_fname, 'w') as j:
        json.dump(cfg, j, indent=4)

    classes_fname = f'{output_name}/{output_name}-classes.txt'
    classes = cfg['data_dict']['names']

    with open(classes_fname, 'w') as f:
        f.writelines([f'{x}\n' for x in classes])

    Path('releases').mkdir(exist_ok=True)

    with tarfile.open(f'{output_name}.tar.gz', 'w:gz') as tar:
        tar.add(output_name, output_name)

    if Path(f'releases/{output_name}').exists() and args.overwrite:
        shutil.rmtree(f'releases/{output_name}', ignore_errors=True)
    shutil.move(output_name, 'releases')
    shutil.move(f'{output_name}.tar.gz', 'releases')

    files_to_move = [x for x in glob('releases/*') if not Path(x).is_dir()]
    Path(f'releases/{output_name}').mkdir(exist_ok=True)
    _ = [shutil.move(x, f'releases/{output_name}') for x in files_to_move]
    return config_fname, classes_fname, cfg


def release_notes(run: wandb.wandb_run.Run, f1_score: float, output_name: str,
                  cfg: dict) -> str:
    """Creates a release notes file.

    Parameters
    ----------
    run: wandb.wandb_run.Run
        The current run from which to extract the model weights.
    f1_score: float
        The F1 score of the current run.
    output_name: str
        The name of the output file.
    cfg: dict
        The run's configuration.

    Returns
    -------
    str
        The content of the generated release notes file.
    """

    _run = {
        'Start time': run.created_at,
        'W&B run URL': run.url,
        'W&B run ID': run.id,
        'W&B run name': run.name,
        'W&B run path': '/'.join(run.path),
        'Number of classes': cfg['data_dict']['nc']
    }

    if cfg.get('base_ml_framework'):
        ml_framework_versions = dict(sorted(cfg['base_ml_framework'].items()))
        _run.update(ml_framework_versions)

    with open(f'releases/{output_name}/{output_name}-notes.md', 'w') as f:

        for k, v in _run.items():
            if k == 'W&B run URL' or v == '':
                f.write(f'**{k}**: {v}\n')
            else:
                f.write(f'**{k}**: `{v}`\n')

        f.write('\n<details>\n<summary>Classes</summary>\n\n```YAML'
                '\n- ' + '\n- '.join(run.config['data_dict']['names']) +
                '\n```\n</details>\n')

        try:
            _, val_imgs_example_url = find_file(run, 'Validation')

            f.write('\n<details>\n<summary>Validation predictions</summary>\n'
                    '\n' + f'<img src="{val_imgs_example_url}" alt="val"'
                    ' width="1280" height="720">'
                    '\n</details>\n')

            cm_idx = run.summary['Results']['captions'].index(
                'confusion_matrix.png')
            cm_fname = run.summary['Results']['filenames'][cm_idx]
            cm_file, cm_url = find_file(run, cm_fname)

            f.write('\n<details>\n<summary>Confusion matrix</summary>\n'
                    '\n' + f'<img src="{cm_url}" alt="cm"'
                    ' width="1280" height="720">'
                    '\n</details>\n')

            shutil.rmtree(Path(cm_file.name).parents[1], ignore_errors=True)
        except TypeError:
            logger.error('Failed to get files from the run...')

        dmw = download_weights.DownloadModelWeights(
            model_version=args.release_version)
        _, weights_url, _ = dmw.get_weights(skip_download=True)

        f.write('\n<details>\n<summary>Model weights</summary>\n'
                f'\n{weights_url} (requires authentication)'
                '\n</details>\n')

        f.write('\n**Validation results**:\n')
        val_results = {
            k: v
            for k, v in run.summary.items()
            if any(x in k for x in ['val/', 'best/precision', 'best/recall'])
        }
        val_results = dict(sorted(val_results.items()))
        val_results.update({'F1-score': f1_score})
        f.write('<table>\n<tr>\n')
        f.write('\n'.join([f'   <td>{x}'
                           for x in val_results.keys()]) + '<tr>\n')
        f.write(
            '\n'.join([f'   <td>{round(x, 4)}'
                       for x in val_results.values()]) + '</table>\n')

    with open(f'releases/{output_name}/{output_name}-notes.md') as f:
        content = f.read()

    return content


def opts() -> argparse.Namespace:
    """This is a function to parse command line arguments.

    Parameters
    ----------
    -n (--release-name): Release base name (e.g., BirdFSD-YOLOv5)
    -v (--release-version): Release version (e.g., v1.0.0-alpha.1.3)
    --repo: Link to the repository
    --overwrite: Overwrites the release folder if it already exists

    Returns
    -------
    args:
        A namespace containing the parsed command line arguments
    """
    parser = argparse.ArgumentParser(
        epilog=f'Basic usage: python {Path(__file__).name} '
        '-v <release_version_name> -r <run_path> --repo <repo_link>')
    parser.add_argument('-v',
                        '--release-version',
                        help='Release version (e.g., v1.0.0-alpha.1.3)',
                        type=str,
                        required=True)
    parser.add_argument('-r',
                        '--run-path',
                        help='The W&B run path to use',
                        type=str,
                        required=True)
    parser.add_argument('--repo', help='Link to the repository', type=str)
    parser.add_argument('--overwrite',
                        help='Overwrite if the release already exists on the ' \
                        'local disk',
                        action='store_true')

    return parser.parse_args()


def main() -> None:
    """This function is the main function of the program.
    It does the following:
        1. Creates a directory for the release
        2. Gets the f1-score of the run
        3. Gets some data assets from the run
        4. Creates the release notes
        5. Generates a command to create the release

    Returns
    -------
    None
    """
    logs_file = add_logger(__file__)
    output_name = args.release_version

    api = wandb.Api()
    run = api.from_path(args.run_path)

    Path(output_name).mkdir(exist_ok=True)

    p = run.summary['best/precision']
    r = run.summary['best/recall']
    f1_score = 2 * ((p * r) / (p + r))
    logger.debug(f'{run.name}: {f1_score}')

    config, classes, cfg = get_assets(run, output_name)

    _ = release_notes(run, f1_score=f1_score, output_name=output_name, cfg=cfg)

    files = [f'releases/{output_name}/*{x}' for x in ['.json', '.gz', '.txt']]
    logger.info(f'gh release create {args.release_version} -d -F '
                f'"releases/{output_name}/{output_name}-notes.md" --title '
                f'"{args.release_version}" --repo '
                f'{args.repo} {" ".join(files)}')
    upload_logs(logs_file)
    return


if __name__ == '__main__':
    load_dotenv()
    args = opts()
    main()

#!/usr/bin/env python
# coding: utf-8

import argparse
import json
import shutil
import tarfile
import textwrap
import time
from glob import glob
from pathlib import Path
from platform import platform
from typing import Union, Optional

import wandb
from dotenv import load_dotenv
from loguru import logger

from model_utils.minio_helper import MinIO
from model_utils.utils import (
    add_logger, upload_logs, compress_data, get_labels_count)


class GenerateRelease:
    def __init__(self, run_path: str, version: str, repo: Optional[str],
                 overwrite: bool = False, dataset: str = None) -> \
            None:
        self.run_path = run_path
        self.version = version
        self.repo = repo
        self.overwrite = overwrite
        self.dataset = dataset

    @staticmethod
    def find_file(run, fname: str) -> \
            Union[tuple, list, None]:
        # noqa
        """Finds a file in a run and uploads it to imgbb.

        Parameters
        ----------
        run:
            The W&B run to use for the release.
        fname: str
            A string, the name of the file to be found.

        Returns
        -------
        tuple
            A tuple of the file object and the url of the uploaded image.
        """
        try:
            files = [x for x in list(run.files()) if fname in x.name]  # noqa
        except IndexError:
            return
        urls = []
        for file in files:
            file.download()
            url = MinIO().upload('public', file.name, public=True)
            if fname != 'Validation':
                return file, url
            else:
                urls.append(url)
        return urls

    def get_assets(self, run) -> Union[tuple, None]:
        """Downloads the best model from the run with the given ID,
        extracts the model's configuration, & classes and saves them
        to a file.

        Returns
        -------
        run:
            The W&B run to use for the release.
        tuple
            The config dictionary and the paths to the config and classes
            files.
        """
        logger.debug(run)
        config_fname = f'{self.version}/{self.version}-config.json'

        for k in [('val', -3), ('train', -3), ('path', -2)]:
            relative_path = '/'.join(
                Path(run.config['data_dict'][k[0]]).parts[k[1]:])
            run.config['data_dict'][k[0]] = relative_path

        logger.debug(run.config)
        print()
        with open(config_fname, 'w') as j:
            json.dump(run.config, j, indent=4)

        classes_fname = f'{self.version}/{self.version}-classes.txt'
        classes = run.config['data_dict']['names']

        with open(classes_fname, 'w') as f:
            f.writelines([f'{x}\n' for x in classes])

        Path('releases').mkdir(exist_ok=True)

        with tarfile.open(f'{self.version}.tar.gz', 'w:gz') as tar:
            tar.add(self.version, self.version)

        if Path(f'releases/{self.version}').exists() and self.overwrite:
            shutil.rmtree(f'releases/{self.version}', ignore_errors=True)
        shutil.move(self.version, 'releases')
        shutil.move(f'{self.version}.tar.gz', 'releases')

        files_to_move = [x for x in glob('releases/*') if not Path(x).is_dir()]
        Path(f'releases/{self.version}').mkdir(exist_ok=True)
        _ = [shutil.move(x, f'releases/{self.version}') for x in files_to_move]
        return config_fname, classes_fname

    def release_notes(self, run, f1_score: float) -> str:
        """Creates a release notes file.

        Parameters
        ----------
        run:
            The W&B run to use for the release.
        f1_score: float
            The F1 score of the current run.

        Returns
        -------
        str
            The content of the generated release notes file.
        """

        _run = {
            'Training start time':
            run.created_at,
            'Training duration':
            time.strftime('%H:%M:%S', time.gmtime(run.summary['_runtime'])),
            'W&B run URL':
            run.url,
            'W&B run ID':
            run.id,
            'W&B run name':
            run.name,
            'W&B run path':
            '/'.join(run.path),
            'Number of classes':
            run.config['data_dict']['nc']
        }

        if run.config.get('dataset_name'):
            _run.update({'Dataset name': run.config['dataset_name']})

        with open(f'releases/{self.version}/{self.version}-notes.md',
                  'w') as f:

            for k, v in _run.items():
                if k == 'W&B run URL' or v == '':
                    f.write(f'**{k}**: {v}\n')
                else:
                    f.write(f'**{k}**: `{v}`\n')

            if run.config.get('system_hardware'):
                sys_cfg = run.config.get('system_hardware')
                system_hardware_section = textwrap.dedent(f'''\
                <details>
                    <summary>System hardware</summary>
                    <table>
                        <tr>
                            <td>CPU count
                            <td>GPU count
                            <td>GPU type
                            <td>NVIDIA driver version
                        <tr>
                            <td>{sys_cfg["cpu_count"]}
                            <td>{sys_cfg["gpu_count"]}
                            <td>{sys_cfg["gpu_type"]}
                            <td>{sys_cfg["nvidia_driver_version"]}
                    </table>
                </details>''')
                f.write(system_hardware_section)

            if run.config.get('base_ml_framework'):
                ml = run.config.get('base_ml_framework')
                base_ml_framework_section = textwrap.dedent(f'''\n
                <details>
                    <summary>Base ML framework</summary>
                    <table>
                        <tr>
                            <td>Python
                            <td>CUDA
                            <td>Torch
                            <td>Torchvision
                        <tr>
                            <td>{ml["Python"]}
                            <td>{ml["CUDA"]}
                            <td>{ml["Torch"]}
                            <td>{ml["Torchvision"]}
                    </table>
                </details>''')
                f.write(base_ml_framework_section)

            f.write('\n<details>\n<summary>Classes</summary>\n\n```YAML'
                    '\n- ' + '\n- '.join(run.config['data_dict']['names']) +
                    '\n```\n')

            hist_file, hist_url = self.find_file(run, 'hist.jpg')
            f.write(f'\n<img src="{hist_url}" alt="classes-hist">'
                    '\n</details>\n')
            Path(hist_file.name).unlink()

            try:
                urls = self.find_file(run, 'Validation')
                f.write(
                    '\n<details>\n<summary>Validation predictions</summary>\n')
                for n, val_imgs_example_url in enumerate(urls):
                    f.write('\n' + f'<img src="{val_imgs_example_url}" '
                            f'alt="val-{n}">')
                f.write('\n</details>\n')

                cm_idx = run.summary['Results']['captions'].index(
                    'confusion_matrix.png')
                cm_fname = run.summary['Results']['filenames'][cm_idx]
                cm_file, cm_url = self.find_file(run, cm_fname)

                f.write('\n<details>\n<summary>Confusion matrix</summary>\n'
                        '\n' + f'<img src="{cm_url}" alt="cm"'
                        ' width="1280" height="720">'
                        '\n</details>\n')

                shutil.rmtree(Path(cm_file.name).parents[1],
                              ignore_errors=True)
            except TypeError:
                logger.error('Failed to get files from the run...')

            f.write('\n**Model performance**:\n')
            val_results = {
                k: v
                for k, v in run.summary.items() if any(
                    x in k for x in ['val/', 'best/precision', 'best/recall'])
            }
            val_results = dict(sorted(val_results.items()))
            val_results.update({'F1-score': f1_score})
            f.write('<table>\n<tr>\n')
            f.write('\n'.join([f'   <td>{x}'
                               for x in val_results.keys()]) + '<tr>\n')
            f.write('\n'.join(
                [f'   <td>{round(x, 4)}'
                 for x in val_results.values()]) + '</table>\n')

        with open(f'releases/{self.version}/{self.version}-notes.md') as f:
            content = f.read()
        return content

    def generate(self) -> None:
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

        api = wandb.Api()
        run = api.from_path(self.run_path)

        Path(self.version).mkdir(exist_ok=True)

        p = run.summary['best/precision']
        r = run.summary['best/recall']
        f1_score = 2 * ((p * r) / (p + r))
        logger.debug(f'{run.name}: {f1_score}')

        _ = self.get_assets(run)
        _ = self.release_notes(run=run, f1_score=f1_score)

        artifact = api.artifact(f'{Path(self.run_path).parent}/'
                                f'run_{Path(self.run_path).name}_model:best')
        _ = artifact.download('artifacts')
        shutil.move(f'artifacts/best.pt', f'releases/{self.version}')
        shutil.rmtree('artifacts')

        upload_logs(logs_file)

        if self.repo:
            files = [
                f'releases/{self.version}/*{x}'
                for x in ['.json', '.gz', '.txt']
            ]

            if self.dataset:
                files = files + [self.dataset]

            gh_cli_cmd = f'gh release create {self.version} -d -F ' \
            f'"releases/{self.version}/{self.version}-notes.md" --title ' \
            f'"{self.version}" --repo ' \
            f'{self.repo} {" ".join(files)}'
            if 'macOS' in platform():
                gh_cli_cmd = gh_cli_cmd + ' | xargs open'
            print(gh_cli_cmd)
        return


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
        '-r <run_path> -v <version> -R <repo_URL>')
    parser.add_argument('-r',
                        '--run-path',
                        help='The W&B run path to use',
                        type=str,
                        required=True)
    parser.add_argument('-v',
                        '--version',
                        help='Release version (e.g., MODEL-v1.0.0-alpha.1.3)',
                        type=str,
                        required=True)
    parser.add_argument('-R',
                        '--repo',
                        help='URL to the repository (i.e., [...].git)',
                        type=str)
    parser.add_argument('-d',
                        '--dataset',
                        help='Path to the dataset used in the run '
                        '(.tar.gpg file)',
                        type=str)
    parser.add_argument('--overwrite',
                        help='Overwrite if the release already exists on '
                        'the local disk',
                        action='store_true')

    return parser.parse_args()


if __name__ == '__main__':
    load_dotenv()
    args = opts()
    gr = GenerateRelease(run_path=args.run_path,
                         version=args.version,
                         repo=args.repo,
                         overwrite=args.overwrite,
                         dataset=args.dataset)
    gr.generate()

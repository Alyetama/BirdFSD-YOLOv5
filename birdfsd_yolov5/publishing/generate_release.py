#!/usr/bin/env python
# coding: utf-8

import argparse
import json
import shutil
import sys
import tarfile
import textwrap
import time
from glob import glob
from pathlib import Path
from typing import Union, Optional

import wandb
from dotenv import load_dotenv
from loguru import logger

from birdfsd_yolov5.model_utils import s3_helper, utils


class GenerateRelease:
    """Generate a release for a model."""

    def __init__(self,
                 run_path: str,
                 version: str,
                 train_folder: str,
                 val_folder: str,
                 dataset_folder: str,
                 repo: Optional[str] = None,
                 overwrite: bool = False) -> None:
        """Initialize the class.

        Args:
            run_path: The path to the run to use for the release.
            version: The version of the release.
            train_folder: The path to the folder containing the training
                results.
            val_folder: The path to the folder containing the validation
                results.
            dataset_folder: The path to the folder containing the dataset.
            repo: The path to the repository to use for the release.
            overwrite: Whether to overwrite an existing release.
        """

        self.run_path = run_path
        self.version = version
        self.repo = repo
        self.overwrite = overwrite
        self.dataset_folder = dataset_folder
        self.train_folder = train_folder
        self.val_folder = val_folder
        self.release_folder = f'releases/{self.version}'

    def find_file(self, run, fname: str) -> \
            Union[tuple, list, None]:
        """Finds a file in a run and uploads it to imgbb.

        Args:
            run: The W&B run to use for the release.
            fname(str): A string, the name of the file to be found.
        """

        s3 = s3_helper.S3()

        try:
            files = [x for x in list(run.files()) if fname in x.name]
        except IndexError:
            logger.error(f'Could not find `{fname}`!')
            return
        if not files:
            return None, None
        urls = []
        for file in files:
            file.download(replace=True)
            if fname == 'hist.jpg':
                dest = f'{Path(fname).stem}-{self.version}{Path(fname).suffix}'
                url = s3.upload('public', file.name, public=True, dest=dest)
            else:
                url = s3.upload('public', file.name, public=True)
            if fname != 'Validation':
                return file, url
            else:
                urls.append(url)
        return urls

    def get_assets(self, run) -> Union[tuple, None]:
        """Accesses the run files to get relevant assets.

        Downloads the best model from the run with the given ID,
        extracts the model's configuration, & classes and saves them
        to a file.

        Args:
            run: The W&B run to use for the release.
        """
        logger.debug(run)

        for k in [('val', -3), ('train', -3), ('path', -2)]:
            try:
                relative_path = '/'.join(
                    Path(run.config['data_dict'][k[0]]).parts[k[1]:])
                run.config['data_dict'][k[0]] = relative_path
            except KeyError:
                pass

        logger.debug(run.config)
        print()

        with open(f'{self.release_folder}/{self.version}-config.json',
                  'w') as j:
            json.dump(run.config, j, indent=4)

        classes = run.config['data_dict']['names']

        with open(f'{self.release_folder}/{self.version}-classes.txt',
                  'w') as f:
            f.writelines([f'{x}\n' for x in classes])

        summary = {
            k: v
            for k, v in run.summary.__dict__.items()
            if k not in ['_client', '_run', '_root', '_locked_keys']
        }

        with open(f'{self.release_folder}/{self.version}-summary.json',
                  'w') as j:
            json.dump(summary, j, indent=4)

        shutil.copy2(f'{self.dataset_folder}/classes.json',
                     f'{self.release_folder}/{self.version}-classes.json')

        train_files = glob(f'{self.train_folder}/*')
        train_files = [
            x for x in train_files if Path(x).name in
            ['results.png', 'results.csv', 'opt.yaml', 'hyp.yaml']
        ]

        for file in train_files:
            shutil.copy2(
                file,
                f'{self.release_folder}/{self.version}-{Path(file).name}')

        with tarfile.open(f'{self.release_folder}/{self.version}-meta.tgz',
                          'w:gz') as tar:
            tar.add(f'{self.release_folder}', arcname=f'{self.version}-meta')

        for file in glob(f'{self.release_folder}/*'):
            if Path(file).suffix != '.tgz':
                Path(file).unlink()

        shutil.copy2(f'{self.dataset_folder}/tasks.json',
                     f'{self.release_folder}/{self.version}-tasks.json')

        shutil.copy2(f'{self.train_folder}/weights/best.pt',
                     f'{self.release_folder}/{self.version}-best_weights.pt')

        with tarfile.open(
                f'{self.release_folder}/{self.version}-val_results.tgz',
                'w:gz') as tar:
            tar.add(f'{self.val_folder}',
                    arcname=f'{self.version}-val_results')
        return

    def release_notes(self, run, f1_score: float) -> str:
        """Creates a release notes file.

        Args:
            run: The W&B run to use for the release.
            f1_score(float): The F1 score of the current run.

        Returns:
            str: The content of the release notes file.
        """

        s3 = s3_helper.S3()

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

        with open(f'{self.release_folder}/{self.version}-notes.md', 'w') as f:

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

            try:
                with open(f'{self.dataset_folder}/classes.json') as j:
                    f.write(
                        '\n<details>\n<summary>Classes</summary>\n\n```JSON\n'
                        f'{json.dumps(json.load(j), indent=4)}\n```\n')
            except FileNotFoundError:
                f.write('\n<details>\n<summary>Classes</summary>\n\n```YAML'
                        '\n- ' +
                        '\n- '.join(run.config['data_dict']['names']) +
                        '\n```\n')

            hist_file, hist_url = self.find_file(run, 'hist.jpg')
            if (hist_file, hist_url) != (None, None):
                f.write(f'\n<img src="{hist_url}" alt="classes-hist">')
                Path(hist_file.name).unlink()

            f.write('\n</details>\n')

            val_files = glob(f'{self.val_folder}/val_*')
            val_urls = []
            for val_file in val_files:
                val_dest = f'{Path(val_file).stem}-{self.version}.jpg'
                val_url = s3.upload('public',
                                    val_file,
                                    public=True,
                                    dest=val_dest)
                val_urls.append((val_dest, val_url))

            f.write('\n<details>\n<summary>Validation predictions</summary>\n')

            for val_file, val_url in val_urls:
                f.write('\n' + f'<img src="{val_url}" '
                        f'alt="{val_file}">')
            f.write('\n</details>\n')

            cm_file = f'{self.val_folder}/confusion_matrix.png'
            cm_dest = f'cm-{self.version}.png'
            cm_url = s3.upload('public', cm_file, public=True, dest=cm_dest)

            f.write('\n<details>\n<summary>Confusion matrix</summary>\n'
                    '\n' + f'<img src="{cm_url}" alt="cm"'
                    ' width="1280" height="720">'
                    '\n</details>\n')

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

        with open(f'{self.release_folder}/{self.version}-notes.md') as f:
            content = f.read()
        return content

    def generate(self) -> None:
        """This method generates the release.

        The method does the following:
        1. Creates a directory for the release.
        2. Gets the f1-score of the run.
        3. Gets some data assets from the run.
        4. Creates the release notes.
        5. Generates a command to create the release on GitHub.
        """
        logs_file = utils.add_logger(__file__)

        api = wandb.Api()
        run = api.from_path(self.run_path)

        if Path(f'{self.release_folder}').exists() and self.overwrite:
            shutil.rmtree(f'{self.release_folder}', ignore_errors=True)

        Path(f'{self.release_folder}').mkdir(exist_ok=True, parents=True)

        p = run.summary['best/precision']
        r = run.summary['best/recall']
        f1_score = 2 * ((p * r) / (p + r))
        logger.debug(f'{run.name}: {f1_score}')

        self.get_assets(run)
        _ = self.release_notes(run=run, f1_score=f1_score)

        dataset_file = glob(f'{self.dataset_folder}/*.tar')
        if dataset_file:
            shutil.copy2(dataset_file[0], f'{self.release_folder}')
            dataset_file_name = Path(dataset_file[0]).name
        else:
            dataset_file_name = None

        print(f'{"-" * 40}\n')
        gpg_cmds = []
        for file in [
                f'{self.version}-best_weights.pt',
                f'{self.version}-tasks.json', dataset_file_name
        ]:
            if file:
                gpg_cmds.append(
                    f'FILE="{self.release_folder}/{file}"; gpg '
                    '--pinentry-mode loopback -c "$FILE" && rm "$FILE"')

        full_gpg_cmd = ' && '.join(gpg_cmds)
        if sys.stdout.isatty():
            print(f'\033[35m{full_gpg_cmd}\033[39m')
        else:
            print(full_gpg_cmd)
        print(f'\n{"-" * 40}')

        utils.upload_logs(logs_file)

        if self.repo:
            print(f'{"-" * 40}\n')
            gh_cli_cmd = f'gh release create {self.version} -d -F ' \
                f'"{self.release_folder}/{self.version}-notes.md" --title ' \
                f'"{self.version}" --repo ' \
                f'{self.repo} {self.release_folder}/*.gpg ' \
                f'{self.release_folder}/*.tgz'

            if sys.stdout.isatty():
                print(f'\033[35m{gh_cli_cmd}\033[39m')
            else:
                print(gh_cli_cmd)
            print(f'\n{"-" * 40}\n')
        return


def _opts() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Namespace object containing the parsed arguments.

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
    parser.add_argument('-t',
                        '--train-folder',
                        help='Path to the train folder',
                        type=str,
                        required=True)
    parser.add_argument('-V',
                        '--val-folder',
                        help='Path to the validation folder',
                        type=str,
                        required=True)
    parser.add_argument('-D',
                        '--dataset-folder',
                        help='Path to the artifacts folder that contains the '
                        'dataset TAR file',
                        type=str,
                        required=True)
    parser.add_argument('-R',
                        '--repo',
                        help='URL to the repository (i.e., [...].git)',
                        type=str)
    parser.add_argument('--overwrite',
                        help='Overwrite if the release already exists on '
                        'the local disk',
                        action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    load_dotenv()
    args = _opts()
    gr = GenerateRelease(run_path=args.run_path,
                         version=args.version,
                         train_folder=args.train_folder,
                         val_folder=args.val_folder,
                         dataset_folder=args.dataset_folder,
                         repo=args.repo,
                         overwrite=args.overwrite)
    gr.generate()

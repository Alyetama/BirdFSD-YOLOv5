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

from model_utils.s3_helper import S3
from model_utils.utils import add_logger, upload_logs


class GenerateRelease:
    def __init__(self, run_path: str, version: str, repo: Optional[str],
                 overwrite: bool = False, dataset_folder: str = None) -> \
            None:
        self.run_path = run_path
        self.version = version
        self.repo = repo
        self.overwrite = overwrite
        self.dataset_folder = dataset_folder
        self.release_folder = f'releases/{self.version}'

    def find_file(self, run, fname: str) -> \
            Union[tuple, list, None]:
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

        s3 = S3()

        try:
            files = [x for x in list(run.files()) if fname in x.name]  # noqa
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

        for k in [('val', -3), ('train', -3), ('path', -2)]:
            relative_path = '/'.join(
                Path(run.config['data_dict'][k[0]]).parts[k[1]:])
            run.config['data_dict'][k[0]] = relative_path

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

        if self.dataset_folder:
            shutil.copy2(f'{self.dataset_folder}/classes.json',
                         f'{self.release_folder}/{self.version}-classes.json')

        with tarfile.open(f'{self.release_folder}/{self.version}-meta.tgz',
                          'w:gz') as tar:
            tar.add(f'{self.release_folder}', arcname=self.version)

        for file in glob(f'{self.release_folder}/*'):
            if Path(file).suffix != '.tgz':
                Path(file).unlink()

        if self.dataset_folder:
            shutil.copy2(f'{self.dataset_folder}/tasks.json',
                         f'{self.release_folder}/{self.version}-tasks.json')
        return

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

            if self.dataset_folder:
                with open(f'{self.dataset_folder}/classes.json') as j:
                    f.write(
                        '\n<details>\n<summary>Classes</summary>\n\n```JSON\n'
                        f'{json.dumps(json.load(j), indent=4)}\n```\n')
            else:
                f.write('\n<details>\n<summary>Classes</summary>\n\n```YAML'
                        '\n- ' +
                        '\n- '.join(run.config['data_dict']['names']) +
                        '\n```\n')

            hist_file, hist_url = self.find_file(run, 'hist.jpg')
            if (hist_file, hist_url) != (None, None):
                f.write(f'\n<img src="{hist_url}" alt="classes-hist">')
                Path(hist_file.name).unlink()

            f.write('\n</details>\n')

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

        with open(f'{self.release_folder}/{self.version}-notes.md') as f:
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

        if Path(f'{self.release_folder}').exists() and self.overwrite:
            shutil.rmtree(f'{self.release_folder}', ignore_errors=True)

        Path(f'{self.release_folder}').mkdir(exist_ok=True, parents=True)

        p = run.summary['best/precision']
        r = run.summary['best/recall']
        f1_score = 2 * ((p * r) / (p + r))
        logger.debug(f'{run.name}: {f1_score}')

        self.get_assets(run)
        _ = self.release_notes(run=run, f1_score=f1_score)

        artifact = api.artifact(f'{Path(self.run_path).parent}/'
                                f'run_{Path(self.run_path).name}_model:best')
        _ = artifact.download('artifacts')
        shutil.move(f'artifacts/best.pt',
                    f'{self.release_folder}/{self.version}-best_weights.pt')
        shutil.rmtree('artifacts')

        if self.dataset_folder:
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
                print(f'\033[31m{full_gpg_cmd}\033[39m')
            else:
                print(full_gpg_cmd)

        upload_logs(logs_file)

        if self.repo:
            print(f'{"-" * 40}\n')
            gh_cli_cmd = f'gh release create {self.version} -d -F ' \
                f'"{self.release_folder}/{self.version}-notes.md" --title ' \
                f'"{self.version}" --repo ' \
                f'{self.repo} {self.release_folder}/*.gpg ' \
                f'{self.release_folder}/*.tgz'

            if sys.stdout.isatty():
                print(f'\033[31m{gh_cli_cmd}\033[39m')
            else:
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
    parser.add_argument('--overwrite',
                        help='Overwrite if the release already exists on '
                        'the local disk',
                        action='store_true')
    parser.add_argument('-D',
                        '--dataset-folder',
                        help='Path to the artifacts folder that contains the '
                        'dataset TAR file',
                        type=str)

    return parser.parse_args()


if __name__ == '__main__':
    load_dotenv()
    args = opts()
    gr = GenerateRelease(run_path=args.run_path,
                         version=args.version,
                         repo=args.repo,
                         overwrite=args.overwrite,
                         dataset_folder=args.dataset_folder)
    gr.generate()

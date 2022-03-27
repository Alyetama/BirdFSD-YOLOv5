import argparse
import base64
import json
import os
import shutil
import tarfile
from datetime import datetime
from pathlib import Path

import requests
import wandb
from dotenv import load_dotenv
from loguru import logger


class DummyModule:
    __version__ = ''


try:
    import torch, torchvision
except ModuleNotFoundError:
    torch, torchvision = DummyModule, DummyModule


def get_artifacts(run, download=False):
    artifacts = list(run.logged_artifacts())
    if len(artifacts) > 1:
        logger.warning('There is more than one artifact in this run!')
    if download:
        return artifacts[-1].download()
    return artifacts


def upload_image(image_path, imgbb_token):
    with open(image_path, 'rb') as f:
        url = 'https://api.imgbb.com/1/upload'
        payload = {
            'key': imgbb_token,
            'image': base64.b64encode(f.read()),
        }
        res = requests.post(url, payload)
    return res


def find_file(run, fname):
    file = [x for x in list(run.files()) if fname in x.name][0]
    file.download()
    imgbb_res = upload_image(file.name, os.environ['IMGBB_API_KEY'])
    return file, imgbb_res.json()['data']['url']


def get_assets(run, output_name, move_to='releases'):
    weights_fname = f'{output_name}/{output_name}-best.pt'
    artifact_local = get_artifacts(run, download=True)
    shutil.move(f'{artifact_local}/best.pt', weights_fname)
    shutil.rmtree('artifacts', ignore_errors=True)

    config_fname = f'{output_name}/{output_name}-config.json'
    cfg = run.config
    for k in [('val', -3), ('train', -3), ('path', -2)]:
        relative_path = '/'.join(Path(cfg['data_dict']['val']).parts[k[1]:])
        cfg['data_dict'][k[0]] = relative_path

    with open(config_fname, 'w') as j:
        json.dump(cfg, j, indent=4)

    classes_fname = f'{output_name}/{output_name}-classes.txt'
    bbd_url = [
        x for x in list(run.files())
        if 'BoundingBoxDebugger' in x.name and '.json' in x.name
    ][0].direct_url
    r = requests.get(bbd_url)
    class_labels = json.loads(r.content)['class_labels']

    with open(classes_fname, 'w') as f:
        f.writelines([f'{x}\n' for x in class_labels.values()])

    with tarfile.open(f'{output_name}.tar', 'w') as tar:
        tar.add(output_name, output_name)

    if move_to:
        Path(move_to).mkdir(exist_ok=True)
        _ = [shutil.move(x, move_to) for x in [output_name, f'{output_name}.tar']]

    return (weights_fname, config_fname, classes_fname)


def opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('-e',
                        '--entity',
                        help='Name of W&B\'s entity',
                        type=str)
    parser.add_argument('-p',
                        '--project-name',
                        help='Name of W&B\'s project',
                        type=str)
    parser.add_argument('-n',
                        '--release-name',
                        help='Release base name',
                        type=str)
    parser.add_argument('-v',
                        '--release-version',
                        help='Release version (vx.y.z)',
                        type=str)
    return parser.parse_args()


def release_notes(run, f1_score, output_name):
    run_timestamp = datetime.fromtimestamp(run.summary['_timestamp'])
    run_start_time = run_timestamp.strftime('%d-%m-%Y %H:%M:%S')
    run_local_tag = run_timestamp.strftime('%d%m%Y')

    RUN = {
        'Start time': run.created_at,
        'Local tag': run_timestamp.strftime('%d%m%Y'),
        'W&B run URL': run.url,
        'W&B run ID': run.id,
        'W&B run name': run.name,
        'W&B run path': '/'.join(run.path),
        'W&B artifact path':
        f'{"/".join(run.path[:-1])}/{get_artifacts(run)[0].name}',
        'Torch version': torch.__version__,
        'Torchvision version': torchvision.__version__,
    }

    with open(f'{output_name}/{output_name}-notes.md', 'w') as f:

        for k, v in RUN.items():
            if k == 'W&B run URL' or v == '':
                f.write(f'**{k}**: {v}\n')
            else:
                f.write(f'**{k}**: `{v}`\n')


        f.write(
            '\n<details>\n<summary>Classes</summary>\n\n```YAML' + \
         '\n- ' + '\n- '.join(run.config['data_dict']['names']) + \
         '\n```\n</details>\n'
        )

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

    with open(f'{output_name}/{output_name}-notes.md') as f:
        content = f.read()

    return content


def main():
    api = wandb.Api({'entity': args.entity, 'project': args.project_name})
    runs = list(api.runs())

    scores = []
    for _run in runs:
        p = _run.summary['best/precision']
        r = _run.summary['best/recall']
        _f1_score = 2 * ((p * r) / (p + r))
        scores.append({_run: _f1_score})

    vals = [list(x.values())[0] for x in scores]
    best_score = max(vals)
    best_run = list([x for x in scores
                     if list(x.values())[0] == best_score][0].keys())[0]

    Path(output_name).mkdir(exist_ok=True)
    
    release_notes(best_run, f1_score=best_score, output_name=output_name)

    weights, config, classes = get_assets(best_run, output_name)


if __name__ == '__main__':
    load_dotenv()
    args = opts()
    output_name = f'{args.release_name}-{args.release_version}'
    main()

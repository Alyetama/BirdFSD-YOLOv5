import argparse
import base64
import copy
import json
import os
import shutil
import sys
import tarfile
from datetime import datetime
from glob import glob
from pathlib import Path

import requests
import wandb
from dotenv import load_dotenv
from loguru import logger


def flush_artifacts(runs):
    for cur_run in runs:
        artifacts = list(cur_run.logged_artifacts())
        _tagged_artifacts = [(x, x.aliases) for x in artifacts]
        for artif in _tagged_artifacts:
            if not any(x for x in artif[1] if x in ['best', 'last', 'latest']):
                try:
                    artif[0].delete(delete_aliases=True)
                    logger.debug(f'Deleted {artif}...')
                except wandb.errors.CommError as e:
                    logger.error(e)


def get_artifacts(run, download=False):
    artifacts = list(run.logged_artifacts())
    _tagged_artifacts = [(x, x.aliases) for x in artifacts]
    for artif in _tagged_artifacts:
        if not any(x for x in artif[1] if x in ['best', 'last', 'latest']):
            try:
                artif[0].delete(delete_aliases=True)
                logger.debug(f'Deleted {artif}...')
            except wandb.errors.CommError:
                continue

    tagged_artifacts = [x for x in _tagged_artifacts if 'best' in x[1]]
    artifact = [x for x in tagged_artifacts if 'best' in x[1]][0][0]
    if download:
        assert 'best' in artifact.aliases
        return artifact.download()
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
    try:
        file = [x for x in list(run.files()) if fname in x.name][0]
    except IndexError:
        return
    file.download()
    imgbb_res = upload_image(file.name, os.environ['IMGBB_API_KEY'])
    return file, imgbb_res.json()['data']['url']


def get_assets(best_run, output_name, move_to='releases'):
    logger.debug(best_run)
    weights_fname = f'{output_name}/{output_name}-best.pt'

    if args.supply:
        artifact = api.artifact(args.supply)
        artifact_local = artifact.download()
        logger.debug(artifact_local)
    else:
        artifact_local = get_artifacts(best_run, download=True)
    try:
        shutil.move(f'{artifact_local}/best.pt', weights_fname)
    except FileNotFoundError:
        assert 'best' in artifact.aliases
        shutil.move(f'{artifact_local}/last.pt', weights_fname)
    shutil.rmtree('artifacts', ignore_errors=True)

    config_fname = f'{output_name}/{output_name}-config.json'
    cfg = best_run.config
    
    for k in [('val', -3), ('train', -3), ('path', -2), ('weights', -2)]:
        if k[0] == 'weights':
            relative_path = '/'.join(Path(cfg[k[0]]).parts[k[1]:])
        else:
            relative_path = '/'.join(Path(cfg['data_dict'][k[0]]).parts[k[1]:])
        cfg['data_dict'][k[0]] = relative_path

    logger.debug(cfg)
    print()
    with open(config_fname, 'w') as j:
        json.dump(cfg, j, indent=4)

    classes_fname = f'{output_name}/{output_name}-classes.txt'
    bbd_url = [
        x for x in list(best_run.files())
        if 'BoundingBoxDebugger' in x.name and '.json' in x.name
    ][0].direct_url
    r = requests.get(bbd_url)
    class_labels = json.loads(r.content)['class_labels']

    with open(classes_fname, 'w') as f:
        f.writelines([f'{x}\n' for x in class_labels.values()])


    if move_to:
        Path(move_to).mkdir(exist_ok=True)

    with tarfile.open(f'{output_name}.tar.gz', 'w:gz') as tar:
        tar.add(output_name, output_name)

    if move_to:
        if Path(move_to).exists():
            # shutil.rmtree(move_to, ignore_errors=True)
            shutil.move(output_name, move_to)
            shutil.move(f'{output_name}.tar.gz', move_to)

    files_to_move = [x for x in glob('releases/*') if not Path(x).is_dir()]
    Path(f'{move_to}/{output_name}').mkdir(exist_ok=True)
    _ = [shutil.move(x, f'{move_to}/{output_name}') for x in files_to_move]
    return (weights_fname, config_fname, classes_fname, cfg)


def opts():
    parser = argparse.ArgumentParser(epilog=f'Basic usage: python {Path(__file__).name} -e "entity" -p "project" -n "release_base_name" -v "v1.0.0-alpha"')
    parser.add_argument('-e',
                        '--entity',
                        help='Name of W&B\'s entity',
                        type=str,
                        required=True)
    parser.add_argument('-p',
                        '--project-name',
                        help='Name of W&B\'s project',
                        type=str,
                        required=True)
    parser.add_argument('-n',
                        '--release-name',
                        help='Release base name (e.g., BirdFSD-YOLOv5)',
                        type=str,
                        required=True)
    parser.add_argument('-v',
                        '--release-version',
                        help='Release version (e.g., v1.0.0-alpha.1.3)',
                        type=str,
                        required=True)
    parser.add_argument('-f',
                        '--flush',
                        help='Flush and exit',
                        action='store_true')
    parser.add_argument('--pick',
                        help='Pick the best model manually',
                        action='store_true')
    parser.add_argument('--supply',
                        help='Supply the model path manually',
                        type=str)
    parser.add_argument('--repo',
                        help='Link to the repository',
                        type=str)

    return parser.parse_args()


def release_notes(run, f1_score, output_name, cfg, move_to='releases'):
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
        f'{"/".join(run.path[:-1])}/{get_artifacts(run)[0].name}'
    }

    if cfg.get('base_ml_framework'):
        ml_framework_versions = dict(sorted(cfg['base_ml_framework'].items()))
        RUN.update(ml_framework_versions)


    with open(f'{move_to}/{output_name}/{output_name}-notes.md', 'w') as f:

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

    with open(f'{move_to}/{output_name}/{output_name}-notes.md') as f:
        content = f.read()

    return content


def main():
    output_name = f'{args.release_name}-{args.release_version}'

    runs = list(api.runs())

    Path(output_name).mkdir(exist_ok=True)

    flush_artifacts(runs)

    if args.flush:
        sys.exit(0)

    scores = []
    for _run in runs:
        p = _run.summary['best/precision']
        r = _run.summary['best/recall']
        _f1_score = 2 * ((p * r) / (p + r))
        logger.debug(f'{_run.name}: {_f1_score}')
        scores.append({_run: _f1_score})

    vals = [list(x.values())[0] for x in scores]
    best_score = max(vals)
    best_run = list([x for x in scores
                     if list(x.values())[0] == best_score][0].keys())[0]

    if args.supply:
        run_name = args.supply.split('/')[-1].split(':')[0].split('_')[1]
        logger.debug(run_name)
        best_run = [x for x  in runs if run_name in x.id][0]
        logger.debug(best_run)

    if args.pick:
        del best_run
        best_run = copy.deepcopy(picked_run)
        _input = input('Run name: ')
        best_run = [x for x in runs if x.name == _input][0]
        p = best_run.summary['best/precision']
        r = best_run.summary['best/recall']
        best_score = 2 * ((p * r) / (p + r))

    move_to = 'releases'
    weights, config, classes, cfg = get_assets(best_run, output_name, move_to)

    release_notes(best_run, f1_score=best_score, output_name=output_name, cfg=cfg, move_to=move_to)

    logger.info(f'gh release create {args.release_version} -d -F "{move_to}/{output_name}/{output_name}-notes.md" --title "{args.release_version}" --repo {args.repo} {move_to}/{output_name}/*')


if __name__ == '__main__':
    load_dotenv()
    args = opts()
    api = wandb.Api({'entity': args.entity, 'project': args.project_name})
    main()
    

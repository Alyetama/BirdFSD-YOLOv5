import argparse
from pathlib import Path

import wandb
from dotenv import load_dotenv
from loguru import logger


def opts():
    parser = argparse.ArgumentParser()
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
    parser.add_argument(
        '-d',
        '--destination',
        help='Output file destination (directory path). Default: current '
             'working directory.',
        type=str,
        default='.')
    return parser.parse_args()


def get_best_weights(entity, project, dest='.'):
    api = wandb.Api({'entity': entity, 'project': project})
    runs = list(api.runs())

    scores = []

    for _run in runs:
        p = _run.summary['best/precision']
        r = _run.summary['best/recall']
        _f1_score = 2 * ((p * r) / (p + r))
        logger.debug(f'{_run.name}: {_f1_score}')
        scores.append((_run, _f1_score))

    best_run = [x for x in scores if x[1] == max([y[1] for y in scores])][0][0]
    logger.info(f'Best run: {best_run.name}')

    artifacts = list(best_run.logged_artifacts())
    best = [x for x in artifacts if 'best' in x.aliases][0]
    to_ = best.download(root=dest)
    logger.debug(f'Downloaded to {Path(to_).absolute()}/best.pt')


if __name__ == '__main__':
    load_dotenv('../.env')
    args = opts()
    get_best_weights(args.entity, args.project_name, args.destination)

#!/usr/bin/env python
# coding: utf-8

import argparse
import copy
import imghdr
import json
import logging
import os
import signal
import shutil
import sys
import uuid
import warnings
from itertools import groupby
from operator import itemgetter
from pathlib import Path

import numpy as np
import ray
import requests
from PIL import Image
from rich.console import Console
from rich.logging import RichHandler
import tensorflow as tf
from tqdm import tqdm

from mongodb_helpers import get_mongodb_data


def keyboard_interrupt_handler(sig, frame):
    logger.info(f'KeyboardInterrupt (ID: {sig}) has been caught...')
    sys.exit(0)


def bot_message(message):
  pipedream_url = os.environ['PIPEDREAM_URL']
  headers = requests.structures.CaseInsensitiveDict()
  headers['Content-type'] = 'text/plain'
  res = requests.post(pipedream_url, message, headers=headers)
  return res.text


def mkdirs():
    Path('tmp').mkdir(exist_ok=True)
    Path('tmp/downloaded').mkdir(exist_ok=True)
    Path('tmp/cropped').mkdir(exist_ok=True)


def make_headers():
    TOKEN = os.environ['TOKEN']
    headers = requests.structures.CaseInsensitiveDict()
    headers['Content-type'] = 'application/json'
    headers['Authorization'] = f'Token {TOKEN}'
    return headers


def get_all_tasks(headers, project_id):
    logger.debug('Getting tasks data... This might take few minutes...')
    url = f'{os.environ["LS_HOST"]}/api/projects/{project_id}/'
    resp = requests.get(url, headers=headers)
    tasks_len = resp.json()['task_number']

    res_lists = []
    progress_bar = tqdm(total=tasks_len)
    step = 100
    page = 1
    steps = 0

    while True:
        url = f'{os.environ["LS_HOST"]}/api/projects/{project_id}/tasks?page_size={step}&page={page}'
        resp = requests.get(url, headers=headers)
        try:
            res_lists.append(resp.json())
        except requests.exceptions.JSONDecodeError:
            break

        page += 1
        progress_bar.update(step)
        steps += step
        if steps >= tasks_len:
            break
        if page == 3:
            break

    res_lists = sum(res_lists, [])
    with open('latest_tasks.json', 'w') as j:
        json.dump(res_lists, j)
    return res_lists


def find_image(img_name):
    for im in md_data:
        if Path(im['file']).name == img_name:
            return im


def load_local_image(img_path, as_numpy=False):
    '''https://github.com/microsoft/CameraTraps/blob/main/classification/crop_detections.py'''
    try:
        with Image.open(img_path) as img:
            img.load()
        if as_numpy:
            return np.array(img)
        return img
    except OSError as e:
        exception_type = type(e).__name__
        logger.error(f'Unable to load {img_path}. {exception_type}: {e}.')
    return None


def save_crop(img, bbox_norm, square_crop, save):
    '''https://github.com/microsoft/CameraTraps/blob/main/classification/crop_detections.py'''
    img_w, img_h = img.size
    xmin = int(bbox_norm[0] * img_w)
    ymin = int(bbox_norm[1] * img_h)
    box_w = int(bbox_norm[2] * img_w)
    box_h = int(bbox_norm[3] * img_h)

    if square_crop:
        box_size = max(box_w, box_h)
        xmin = max(0, min(xmin - int((box_size - box_w) / 2), img_w - box_w))
        ymin = max(0, min(ymin - int((box_size - box_h) / 2), img_h - box_h))
        box_w = min(img_w, box_size)
        box_h = min(img_h, box_size)

    if box_w == 0 or box_h == 0:
        logger.debug(f'Skipping size-0 crop (w={box_w}, h={box_h}) at {save}')
        return False

    crop = img.crop(box=[xmin, ymin, xmin + box_w,
                         ymin + box_h])  # [left, upper, right, lower]

    if square_crop and (box_w != box_h):
        crop = ImageOps.pad(crop, size=(box_size, box_size), color=0)
    crop.save(save)
    return os.path.dirname(save)


@ray.remote
def download_and_crop(task_id, as_numpy=False):
    url = f'{os.environ["LS_HOST"]}/api/tasks/{task_id}'

    resp = requests.get(url, headers=headers)
    task_ = resp.json()
    if task_['predictions']:
        return
    img_in_task = task_['data']['image']

    LS_domain_name = os.environ['LS_HOST'].split('//')[1]
    SRV_domain_name = os.environ['SRV_HOST'].split('//')[1]

    url = task_['data']['image'].replace(
        f'{LS_domain_name}/data/local-files/?d=', f'{SRV_domain_name}/')

    img_name = Path(img_in_task).name
    img_relative_path = f'tmp/downloaded/{img_name}'

    bbox_res = find_image(img_name)

    r = requests.get(url)
    with open(img_relative_path, 'wb') as f:
        f.write(r.content)

    if not imghdr.what(img_relative_path):
        logger.error(f'Not a valid image file: {img_relative_path}')
        return

    img = load_local_image(img_relative_path, as_numpy=as_numpy)
    return bbox_res, img


@ray.remote
def process_input(result):
    input_ = []
    for detection in result['bbox_res']['detections']:
        if detection['category'] != '1':
            continue
        item = copy.deepcopy(result)
        item['bbox_res']['detections'] = detection
        input_.append(item)
    return input_


def preprocess(image_path):
    image_raw = tf.io.read_file(image_path)
    image = tf.image.decode_image(image_raw)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (224, 224))
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    image = image[None, ...]
    return image


# @ray.remote(num_gpus=1)
@ray.remote
def crop_input(_input_item):
    out_cropped = f'tmp/cropped/{uuid.uuid4().hex}.jpg'
    bbox_ = _input_item['bbox_res']['detections']['bbox']
    save_crop(_input_item['img'], bbox_, False, out_cropped)
    preprocessed_image = preprocess(out_cropped)
    return {
        'task_id': _input_item['task_id'],
        'bbox': bbox_,
        'out_cropped': out_cropped,
        'cropped_tensor': preprocessed_image
    }


def divide_list(list_, chunk_size):
    return [list_[i:i + chunk_size] for i in range(0, len(list_), chunk_size)]


def make_prediction(im_batch):
    with tf.device('/gpu:0'):
        reloaded_result_batch = model.predict(im_batch)
        reloaded_predicted_id = tf.math.argmax(reloaded_result_batch, axis=-1)
        class_names = np.load('class_names.npy')
        reloaded_predicted_label_batch = class_names[reloaded_predicted_id]
        return reloaded_result_batch, reloaded_predicted_id, reloaded_predicted_label_batch


def predict_batch(BATCH):
    logger.info('Downloading and cropping the batch...')
    futures = []
    for task_id in BATCH:
        future = download_and_crop.remote(task_id, as_numpy=False)
        futures.append([task_id, future])

    results = []
    for future in tqdm(futures):
        result = ray.get(future[1])  # bbox_res, img
        if not result:
            continue
        else:
            result = {
                'task_id': future[0],
                'bbox_res': result[0],
                'img': result[1]
            }
            results.append(result)

    logger.info('Sorting the batch...')
    input_futures = []
    inputs_ = []

    for res in results:
        input_futures.append(process_input.remote(res))

    for future_ in tqdm(input_futures):
        out = ray.get(future_)
        if out:
            inputs_.append(out)

    input_expanded = sum(inputs_, [])

    logger.info('Preprocessing the raw images in the batch...')
    preproc_futures = []
    batch_dicts = []

    for _i in input_expanded:
        preproc_futures.append(crop_input.remote(_i))

    for preproc_future_ in tqdm(preproc_futures):
        out_ = ray.get(preproc_future_)
        batch_dicts.append(out_)

    imgs_batch = tf.concat([x['cropped_tensor'] for x in batch_dicts], axis=0)
    imgs_batch.shape

    class_names = np.load('class_names.npy')

    logger.info('Using the batch to make predictions...')
    _preds = make_prediction(imgs_batch)
    _Dicts = []

    logger.info('Sorting predictions in dictionaries...')
    for x, y, z in zip(batch_dicts, _preds[0], _preds[1]):
        _D = copy.deepcopy(x)
        _D['prediction'] = class_names[z]
        _D['score'] = y.flatten()[z]
        del _D['cropped_tensor']
        _Dicts.append(_D)

    POST_DICTS = []
    for key, value in groupby(_Dicts, key=itemgetter('task_id')):
        POST_DICTS.append({key: list(value)})

    return POST_DICTS


def opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('-p',
                        '--project-id',
                        help='Project id number',
                        type=int,
                        required=True)
    parser.add_argument('-m',
                        '--model-path',
                        help='Path to the model in the SavedModel format',
                        type=str)
    parser.add_argument('-b',
                        '--batch-size',
                        help='Batch size to process per iteration',
                        type=int)
    parser.add_argument(
        '-s',
        '--min-score',
        help=
        'Minimum prediction score to accept as valid prediction. Accept all if left empty',
        type=float)
    parser.add_argument('-c',
                        '--class-names',
                        help='Path to class names in .npy format',
                        type=str)

    parser.add_argument('-e',
                    '--exported-tasks',
                    help='Project exported tasks JSON file path',
                    type=str)
    return parser.parse_args()


@ray.remote
def main(_DCT):
    for K, _item in _DCT.items():
        results = []
        scores = []

        for DETECTION in _item:
            scores.append(float(DETECTION['score']))
            x, y, width, height = [x * 100 for x in DETECTION['bbox']]

            results.append({
                'from_name': 'label',
                'to_name': 'image',
                'type': 'rectanglelabels',
                'value': {
                    'rectanglelabels': [DETECTION['prediction']],
                    'x': x,
                    'y': y,
                    'width': width,
                    'height': height
                },
                'score': float(DETECTION['score'])
            })

        post_ = {
            'model_version': 'picam-detector_1647175692',
            'result': results,
            'score': np.mean(scores),
            'cluster': 0,
            'neighbors': {},
            'mislabeling': 0,
            'task': K
        }

        logger.debug({'post': post_})

        url = F'{os.environ["LS_HOST"]}/api/predictions/'
        resp = requests.post(url, headers=headers, data=json.dumps(post_))
        logger.debug({'response': resp.json()})


if __name__ == '__main__':
    args = opts()

    IDs_BATCH_SIZE = args.batch_size
    MODEL_PATH = args.model_path
    class_names = args.class_names

    logging.basicConfig(level='NOTSET',
                        format='%(message)s',
                        datefmt='[%X]',
                        handlers=[RichHandler()])

    tf.get_logger().setLevel('INFO')
    tf.autograph.set_verbosity(1)
    logger = logging.getLogger('rich')
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '1'
    os.environ['RAY_IGNORE_UNHANDLED_ERRORS'] = '1'
    warnings.simplefilter(action='ignore', category=FutureWarning)

    sess = ray.init()
    logger.info(sess)

    signal.signal(signal.SIGINT, keyboard_interrupt_handler)

    md_data = get_mongodb_data()
    headers = make_headers()
    mkdirs()

    if not Path(class_names).exists():
        raise FileNotFoundError(
            'No class names detected. You need to train the model at least once!'
        )

    try:
        project_tasks = get_all_tasks(headers, args.project_id)
    except requests.exceptions.JSONDecodeError as e:
        logger.error(e)
        bot_message(f'TIMEOUT ERROR! Check if label-studio is online, then try again...\n{e}')
        logger.error('TIMEOUT ERROR! Failed to connect to label-studio!')
        logger.error('Check if label-studio is online then try again...')
        ray.shutdown()
        sys.exit(1)

    # tasks_id = [t_['id'] for t_ in project_tasks]
    tasks_id = [
        t_['id'] for t_ in project_tasks if not t_['predictions']
    ]  # !!! will ignore split predictions within a task of detections of one task are in different batches (intentional bug)

    logger.info(f'Number of tasks to predict: {len(tasks_id)}')
    tasks_id_batches = divide_list(tasks_id, IDs_BATCH_SIZE)

    model = tf.keras.models.load_model(MODEL_PATH)

    for N, BATCH in enumerate(tasks_id_batches):
        Console().rule(f'[#50fa7b]BATCH: {N + 1}/{len(tasks_id_batches)}')
        try:
            POST_DICTS = predict_batch(BATCH)
        except IndexError as e:
            logger.debug(e)
            continue

        logger.info('Posting the batch results to label-studio...')

        applied_futures = []
        for _task in POST_DICTS:
            applied_futures.append(main.remote(_task))

        ready_tasks = []
        for task_future in tqdm(applied_futures):
            ready_tasks.append(ray.get(task_future))

    ray.shutdown()
    shutil.rmtree('tmp', ignore_errors=True)
    bot_message('Task completed successfully! You can safely stop the compute instance in few minutes...')

#!/usr/bin/env python
# coding: utf-8

import argparse
import random
import shutil
import uuid
from glob import glob
from pathlib import Path

import numpy as np
import ray
import requests
from PIL import Image
from python_color_transfer.color_transfer import ColorTransfer, Regrain
from python_color_transfer.utils import Rotations
from tqdm import tqdm

random.seed(1)


class Transfer(ColorTransfer, Regrain, Rotations):

    def __init__(self,
                 n=300,
                 eps=1e-6,
                 m=6,
                 c=3,
                 file_path=None,
                 ref_path=None):
        super(ColorTransfer, self).__init__(n)
        super(Regrain, self).__init__()
        super(Rotations, self).__init__()
        if c == 3:
            self.rotation_matrices = self.optimal_rotations()
        else:
            self.rotation_matrices = self.random_rotations(m, c=c)
        self.eps = eps
        self.m = m
        self.c = c
        self.n = n
        self.file_path = file_path
        self.ref_path = ref_path

    @staticmethod
    def img2arr(file_path):
        return np.asarray(Image.open(file_path), dtype=np.uint8)

    @staticmethod
    def save(arr, output):
        im = Image.fromarray(arr.astype('uint8')).resize((640, 640), resample=2)
        im.save(output, 'JPEG', quality=100, subsampling=0)
        return output

    def insert(self, file_path, ref_path):
        self.file_path = file_path
        self.ref_path = ref_path
        return

    def pdf(self):
        img_in = self.img2arr(self.file_path)
        _result = self.pdf_tranfer(img_in, self.img2arr(self.ref_path))
        result = self.regrain(img_in, _result)
        return result

    def mstd(self):
        result = self.mean_std_transfer(self.img2arr(self.file_path),
                                        self.img2arr(self.ref_path))
        return result

    def lt(self):
        result = self.lab_transfer(self.img2arr(self.file_path),
                                   self.img2arr(self.ref_path))
        return result


@ray.remote
def process(img, label, ref_img):
    method = random.choice(['pdf', 'mstd', 'lt'])
    t = Transfer()
    t.insert(img, ref_img)
    r = getattr(t, method).__call__()

    _uuid = uuid.uuid4()
    ds_name = f'color_aug_{int(time.time())}'

    out_imgs_dir = f'{ds_name}/images/{Path(Path(img).parent).name}'
    out_labels_dir = f'{ds_name}/labels/{Path(Path(label).parent).name}'
    Path(out_imgs_dir).mkdir(exist_ok=True, parents=True)
    Path(out_labels_dir).mkdir(exist_ok=True, parents=True)

    t.save(r, f'{out_imgs_dir}/{_uuid}.jpg')
    shutil.copy2(label, f'{out_labels_dir}/{_uuid}.txt')
    return


def opts() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('-d',
                        '--dataset-dir',
                        help='Path to the preprocessed dataset',
                        type=str,
                        required=True)
    parser.add_argument(
        '-r',
        '--ref-images-dir',
        help='Path to the folder with images to use as reference',
        type=str,
        required=True)
    return parser.parse_args()


def run(dataset_dir, ref_images_dir):
    ref_imgs = glob(f'{ref_images_dir}/*')

    images = sorted(glob(f'{dataset_dir}/images/**/*'))
    labels = sorted(glob(f'{dataset_dir}/labels/**/*'))

    futures = [
        process.remote(img, label, random.choice(ref_imgs))
        for img, label in zip(images, labels)
    ]
    _ = [ray.get(future) for future in tqdm(futures)]
    return


if __name__ == '__main__':
    args = opts()
    run(dataset_dir=args.dataset_dir, ref_images_dir=args.ref_images_dir)

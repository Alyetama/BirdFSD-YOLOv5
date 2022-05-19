#!/usr/bin/env python
# coding: utf-8

import uuid
from glob import glob
from pathlib import Path

import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np
from PIL import Image
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage


def xywh_to_xyxy(x, y, w, h, image_width, image_height):
    x1 = (x - w / 2) * image_width
    y1 = (y - h / 2) * image_height
    x2 = x1 + (w * image_width)
    y2 = y1 + (h * image_height)
    return (x1, y1, x2, y2)


def create_batch(images_dir, labels_dir):
    _images = sorted(glob(f'{images_dir}/*'))
    _labels = sorted(glob(f'{labels_dir}/*'))

    batch = [np.array(Image.open(img), dtype=np.uint8) for img in _images]
    images = np.asarray(batch, dtype=np.uint8)
    bbs = []

    for label, img in zip(_labels, images):
        with open(label) as f:
            lines = f.readlines()

        bboxes_per_img = []
        for line in lines:
            nxywh = list(map(float, line.rstrip().split(' ')))
            xywh = nxywh[1:]
            bbox = xywh_to_xyxy(*xywh, img.shape[1], img.shape[0])
            bboxes_per_img.append(BoundingBox(*bbox))
        bbs.append(bboxes_per_img)
    return images, bbs


def export_augs_as_files(images_aug, output_folder):
    Path(output_folder).mkdir(exist_ok=True)
    for im, bbs in zip(*images_aug):
        bbs = BoundingBoxesOnImage(bbs, im.shape[:-1])
        image_w_bbs = bbs.draw_on_image(im, size=2)
        im = Image.fromarray(image_w_bbs)
        im.save(f'{output_folder}/aug-{str(uuid.uuid4()).split("-")[-1]}.jpg',
                'JPEG')


def aug_pipelines():
    """https://imgaug.readthedocs.io/en/latest/source/examples_basics.html"""
    ia.seed(1)

    seq = iaa.Sequential(
        [
            iaa.Fliplr(0.5),  # horizontal flips
            iaa.Crop(percent=(0, 0.1)),  # random crops
            # Small gaussian blur with random sigma between 0 and 0.5.
            # But we only blur about 50% of all images.
            iaa.Sometimes(0.5, iaa.GaussianBlur(sigma=(0, 0.5))),
            # Strengthen or weaken the contrast in each image.
            iaa.LinearContrast((0.75, 1.5)),
            # Add gaussian noise.
            # For 50% of all images, we sample the noise once per pixel.
            # For the other 50% of all images, we sample the noise per pixel AND
            # channel. This can change the color (not only brightness) of the
            # pixels.
            iaa.AdditiveGaussianNoise(
                loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
            # Make some images brighter and some darker.
            # In 20% of all cases, we sample the multiplier once per channel,
            # which can end up changing the color of the images.
            iaa.Multiply((0.8, 1.2), per_channel=0.2),
            # Apply affine transformations to each image.
            # Scale/zoom them, translate/move them, rotate them and shear them.
            iaa.Affine(scale={
                "x": (0.8, 1.2),
                "y": (0.8, 1.2)
            },
                       translate_percent={
                           "x": (-0.2, 0.2),
                           "y": (-0.2, 0.2)
                       },
                       rotate=(-25, 25),
                       shear=(-8, 8))
        ],
        random_order=True)  # apply augmenters in random order
    return seq

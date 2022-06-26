#!/usr/bin/env python
# coding: utf-8

import argparse
import random
import shutil
import tarfile
import uuid
from datetime import datetime
from glob import glob
from pathlib import Path

import imgaug as ia
import imgaug.augmenters as iaa
import numpy as np
import pybboxes as pbx
from PIL import Image
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage
from tqdm import tqdm


def xywh_to_xyxy(x: float, y: float, w: float, h: float, image_width: float,
                 image_height: float) -> tuple:
    """Converts a bounding box from xywh format to xyxy format.

    Args:
        x (float): The x coordinate of the center of the bounding box.
        y (float): The y coordinate of the center of the bounding box.
        w (float): The width of the bounding box.
        h (float): The height of the bounding box.
        image_width (float): The width of the image.
        image_height (float): The height of the image.

    Returns:
        tuple: A tuple containing the x1, y1, x2, and y2 coordinates of the
        bounding box.

    """
    x1 = (x - w / 2) * image_width
    y1 = (y - h / 2) * image_height
    x2 = x1 + (w * image_width)
    y2 = y1 + (h * image_height)

    return x1, y1, x2, y2


def export_augs_as_files(image_aug: list, bbs_aug: list, output_dir: str,
                         _skipped: list) -> None:
    """Exports augmented images and bounding boxes to a directory.

    Args:
        image_aug (list): List of augmented images.
        bbs_aug (list): List of augmented bounding boxes.
        output_dir (str): Path to the output directory.
        _skipped (list): List of skipped images.

    Returns:
        None

    """
    Path(f'{output_dir}/_labels').mkdir(exist_ok=True, parents=True)
    Path(f'{output_dir}/_images').mkdir(exist_ok=True, parents=True)

    for im, bbs in zip(image_aug, bbs_aug):
        SKIP = False
        if bbs[0].is_out_of_image(im):
            _skipped.append(1)
            continue

        bbs = BoundingBoxesOnImage(bbs, im.shape[:-1])
        im_shape = (im.shape[1], im.shape[0])

        im = Image.fromarray(im)

        fname = f'aug-{uuid.uuid4()}'

        with open(f'{output_dir}/_labels/{fname}.txt', 'w') as f:
            for bbox in bbs:
                voc_bbox = (bbox.x1, bbox.y1, bbox.x2, bbox.y2)
                try:
                    yolo_bbox = pbx.convert_bbox(voc_bbox,
                                                 from_type='voc',
                                                 to_type='yolo',
                                                 image_width=im_shape[0],
                                                 image_height=im_shape[1])
                except ValueError:
                    SKIP = True
                    continue

                line = ' '.join(
                    [str(_x) for _x in [int(bbox.label), *yolo_bbox]])
                f.write(line + '\n')
        if SKIP:
            Path(f'{output_dir}/_labels/{fname}.txt').unlink()
            _skipped.append(1)
            continue

        im.save(f'{output_dir}/_images/{fname}.jpg', 'JPEG')


def aug_pipelines() -> iaa.Sequential:
    # noinspection PyPep8
    """A simple and common augmentation sequence

        This function returns a sequence of augmentations to be applied to the
        images. The augmentations are:

        1. Flipping the image horizontally with a probability of 0.5
        2. Random cropping of the image with a probability of 0.1
        3. Blurring the image with a probability of 0.5
        4. Changing the contrast of the image with a probability of 1
        5. Adding gaussian noise to the image with a probability of 0.5
        6. Changing the brightness of the image with a probability of 0.2
        7. Applying affine transformations to the image with a probability of 1

        Notes:
            Ref: https://imgaug.readthedocs.io/en/latest/source/examples_basics.html  # noqa

        Returns
            iaa.Sequential: A sequence of augmentations to be applied to the
            images.

        """
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
            # For the other 50% of all images, we sample the noise per pixel
            # AND channel. This can change the color (not only brightness)
            # of the pixels.
            iaa.AdditiveGaussianNoise(
                loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
            # Make some images brighter and some darker.
            # In 20% of all cases, we sample the multiplier once per channel,
            # which can end up changing the color of the images.
            iaa.Multiply((0.8, 1.2), per_channel=0.2),
            # Apply affine transformations to each image.
            # Scale/zoom them, translate/move them, rotate them and shear them.
            # iaa.Affine(scale={
            #     "x": (0.8, 1.2),
            #     "y": (0.8, 1.2)
            # },
            #            translate_percent={
            #                "x": (-0.2, 0.2),
            #                "y": (-0.2, 0.2)
            #            },
            #            rotate=(-25, 25),
            #            shear=(-8, 8))
        ],
        random_order=True)  # apply augmenters in random order
    return seq


def create_batch(images_dir: str,
                 labels_dir: str,
                 output_dir: str,
                 batch_size: int = 128) -> list:
    """Creates batches of images and labels from the given directories.

    Args:
        images_dir (str): The directory containing the images.
        labels_dir (str): The directory containing the labels.
        output_dir (str): The directory to save the augmented images and
            labels.
        batch_size (int): The number of images to be augmented in a single
            batch.
    Returns:
        list: A list of images that were skipped due to errors.

    """
    SKIPPED = []
    img_files = sorted(glob(f'{images_dir}/**/*'))
    label_files = sorted(glob(f'{labels_dir}/**/*'))

    image_batches = [
        img_files[i:i + batch_size]
        for i in range(0, len(img_files), batch_size)
    ]

    label_batches = [
        label_files[i:i + batch_size]
        for i in range(0, len(label_files), batch_size)
    ]

    for labels_batch, images_batch in zip(label_batches, image_batches):

        images_batch_resized_arrs = [
            np.array(
                Image.open(img).resize((640, 640), resample=2),  # noqa
                dtype=np.uint8) for img in tqdm(images_batch)
        ]

        images_batch_arrs = np.asarray(images_batch_resized_arrs,
                                       dtype=np.uint8)

        bbs = []

        for label, img in zip(labels_batch, images_batch_arrs):
            with open(label) as f:
                lines = f.readlines()

            bboxes_per_img = []
            for line in lines:
                nxywh = list(map(float, line.rstrip().split(' ')))
                xywh = nxywh[1:]
                bbox = xywh_to_xyxy(*xywh, img.shape[1], img.shape[0])
                bboxes_per_img.append(BoundingBox(*bbox, label=nxywh[0]))
            bbs.append(bboxes_per_img)

        Path(output_dir).mkdir(exist_ok=True, parents=True)

        seq = aug_pipelines()
        image_aug, bbs_aug = seq(images=images_batch_arrs, bounding_boxes=bbs)
        export_augs_as_files(image_aug, bbs_aug, output_dir, SKIPPED)
    return SKIPPED


def split_data(images_dir: str, labels_dir: str, _output_dir: str) -> None:
    """Split the data into train and validation sets.

    Args:
        images_dir (str): The path to the directory containing the images.
        labels_dir (str): The path to the directory containing the labels.
        _output_dir (str): The path to the output directory.

    Returns:
        None

    """
    random.seed(1)

    for subdir in ['images/train', 'labels/train', 'images/val', 'labels/val']:
        Path(f'{_output_dir}/{subdir}').mkdir(parents=True, exist_ok=True)

    images = sorted(glob(f'{images_dir}/*'))
    labels = sorted(glob(f'{labels_dir}/*'))
    pairs = list(zip(images, labels))

    len_ = len(images)
    train_len = round(len_ * 0.8)
    random.shuffle(pairs)

    train, val = pairs[:train_len], pairs[train_len:]

    for split, split_str in zip([train, val], ['train', 'val']):
        for n, dtype in zip([0, 1], ['images', 'labels']):
            _ = [
                shutil.move(
                    x[n],
                    f'{_output_dir}/{dtype}/{split_str}/{Path(x[n]).name}')
                for x in split
            ]
    return


def check_classes_preserved(classes_file: str, output_dir: str) -> None:
    """Checks if the dataset classes were preserved after augmentation.

    Args:
        classes_file: The path to the classes file.
        output_dir: The path to the output directory.

    Returns:
        None

    """
    with open(classes_file) as f:
        classes = f.read().splitlines()
        classes_len = len(classes)
    aug_labels = glob(f'{output_dir}/labels/**/*')
    existing_aug_labels = []
    for file in aug_labels:
        with open(file) as f:
            lines = f.read().splitlines()
            existing_aug_labels.append([line.split(' ')[0] for line in lines])
    existing_aug_labels = list(set(sum(existing_aug_labels, [])))
    if len(existing_aug_labels) != classes_len:
        raise AssertionError
    shutil.copy2(classes_file, output_dir)
    print('Classes were preserved.')


def run_aug_pipeline(dataset_path: str, batch_size: int = 128) -> None:
    """Runs the images' augmentation pipeline

    This function takes a dataset path and augments the dataset by applying
    random transformations to the images and labels.

    Args:
        dataset_path (str): The path to the dataset.
        batch_size (int): The number of images to process at once.

    Returns:
        None

    """
    output_dir = f'{dataset_path}-aug'
    imgs_source = f'{dataset_path}/images'
    labels_source = f'{dataset_path}/labels'
    classes_file = f'{dataset_path}/classes.txt'
    dataset_config = f'{dataset_path}/dataset_config.yml'

    SKIPPED = create_batch(imgs_source, labels_source, output_dir, batch_size)
    print('Skipped', len(SKIPPED), 'images.')

    split_data(f'{output_dir}/_images', f'{output_dir}/_labels', output_dir)

    shutil.rmtree(f'{output_dir}/_images')
    shutil.rmtree(f'{output_dir}/_labels')

    check_classes_preserved(classes_file, output_dir)

    with open(dataset_config) as f1:
        lines = f1.read()
        with open(f'{output_dir}/dataset_config.yml', 'w') as f2:
            dataset_name = Path(dataset_path).name
            aug_dataset_name = Path(output_dir).name
            f2.write(
                lines.replace(f'path: {dataset_name}',
                              f'path: {aug_dataset_name}'))

    ts = datetime.now().strftime('%m-%d-%Y_%H.%M.%S')
    dataset_name = f'{output_dir}-{ts}.tar'

    with tarfile.open(dataset_name, 'w') as tar:
        tar.add(output_dir, Path(output_dir).name)
    return


def _opts() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Namespace object containing the parsed arguments.

    """
    parser = argparse.ArgumentParser()
    parser.add_argument('-d',
                        '--dataset-path',
                        help='Path to the preprocessed dataset directory',
                        type=str,
                        required=True)
    return parser.parse_args()


if __name__ == '__main__':
    args = _opts()
    run_aug_pipeline(dataset_path=args.dataset_path)

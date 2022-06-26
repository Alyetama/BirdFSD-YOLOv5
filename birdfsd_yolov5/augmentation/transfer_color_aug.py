#!/usr/bin/env python
# coding: utf-8

import argparse
import random
import shutil
import time
import uuid
from glob import glob
from pathlib import Path
from typing import Optional

import numpy as np
import ray
from PIL import Image
from python_color_transfer.color_transfer import ColorTransfer, Regrain
from python_color_transfer.utils import Rotations
from tqdm import tqdm

random.seed(1)


class Transfer(ColorTransfer, Regrain, Rotations):
    """Uses three color transfer algorithms to augment the dataset.

    This class is used to transfer the color distribution of the reference
    image to the source image while preserving the details and structure of the
    source image.

    """

    def __init__(self,
                 n: int = 300,
                 eps: float = 1e-6,
                 m: int = 6,
                 c: int = 3,
                 file_path: Optional[str] = None,
                 ref_path: Optional[str] = None) -> None:
        """Initializes the class.

        Args:
            n (int): The number of bins for the histogram.
            eps (float): A small value to avoid division by zero.
            m (int): The number of rotations to perform.
            c (int): The number of channels in the image.
            file_path (str): The path to the source image.
            ref_path (str): The path to the reference image.

        Returns:
            None

        """
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
    def img2arr(file_path: str) -> np.ndarray:
        """Converts an image file to a numpy array.
        Args:
            file_path (str): The path to the image file.

        Returns:
            np.ndarray: The image as a numpy array.

        """
        return np.asarray(Image.open(file_path), dtype=np.uint8)  # noqa: PyTypeChecker

    @staticmethod
    def save(arr: np.ndarray, output: str) -> None:
        """Saves an image array to disk.

        Args:
            arr (np.ndarray): The image to save.
            output (str): The path to save the image to.

        Returns:
            str: The path to the saved image.

        """
        im = Image.fromarray(arr.astype('uint8')).resize((640, 640),
                                                         resample=2)
        im.save(output, 'JPEG', quality=100, subsampling=0)
        return

    def insert(self, file_path: str, ref_path: str) -> None:
        """Inserts data into the `Transfer` class instance.

        Inserts a source and target image to the class to be used by any of the 
        transformer methods. 

        Args:
            file_path (str): The path to the source image.
            ref_path (str): The path to the reference image.

        Returns:
            None

        """
        self.file_path = file_path
        self.ref_path = ref_path
        return

    def pdf_algo(self) -> np.ndarray:
        """Method proposed by Pitié, F., Kokaram, A. C., & Dahyot, R. (2007).

        Note:
            Original paper: https://doi.org/10.1016/j.cviu.2006.11.011

        Returns:
            np.ndarray: Transformed image as a numpy array.

        """
        img_in = self.img2arr(self.file_path)
        result = self.pdf_tranfer(img_in, self.img2arr(self.ref_path))
        return self.regrain(img_in, result)

    def mstd_algo(self) -> np.ndarray:
        """This method uses algorithms proposed by Reinhard, E., Adhikhmin, 
        M., Gooch, B., & Shirley, P. (2001).

        Note:
            Original paper: https://doi.org/10.1109/38.946629

        Returns:
            np.ndarray: Transformed image as a numpy array.
            
        """
        return self.mean_std_transfer(self.img2arr(self.file_path),
                                      self.img2arr(self.ref_path))

    def lm_algo(self) -> np.ndarray:
        """This method adapts the source mean and std to the reference image's

        Returns:
            np.ndarray: Transformed image as a numpy array.

        """


@ray.remote
def transfer_color(img_file: str, label_file: str, ref_img: str) -> None:
    """Transfers color from a reference image to a target image

    A ray remote function that randomly pick a method and use it to transfer
    the color from the reference to the source image, then saves the
    image and label and saves them to `color_aug_<input_dir>/x/y` in yolov5 
    format.

    Args:
        img_file (str): The path to the source image.
        label_file (str): The path to the source label.
        ref_img (str): The path to the reference image.
    """
    method = random.choice(['pdf_algo', 'mstd_algo', 'lm_algo'])
    t = Transfer()
    t.insert(img_file, ref_img)
    r = getattr(t, method).__call__()

    _uuid = uuid.uuid4()
    ds_name = f'color_aug_{int(time.time())}'

    out_imgs_dir = f'{ds_name}/images/{Path(Path(img_file).parent).name}'
    out_labels_dir = f'{ds_name}/labels/{Path(Path(label_file).parent).name}'
    Path(out_imgs_dir).mkdir(exist_ok=True, parents=True)
    Path(out_labels_dir).mkdir(exist_ok=True, parents=True)

    t.save(r, f'{out_imgs_dir}/{_uuid}.jpg')
    shutil.copy2(label_file, f'{out_labels_dir}/{_uuid}.txt')
    return


def _opts() -> argparse.Namespace:
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


def run(dataset_dir: str, ref_images_dir: str) -> None:
    """Runs the color transfer pipeline

    Runs the color tansfer augmentation from a dataset with a structure that 
    conforms to yolov5 dataset structure. Reference images are randomly 
    selected from a pool of images that are placed in `ref_images_dir`.

    Args:
        dataset_dir (str): The directory containing the dataset.
        ref_images_dir (str): The directory containing the reference images.

    Returns:
        None

    """
    ref_imgs = glob(f'{ref_images_dir}/*')

    images = sorted(glob(f'{dataset_dir}/images/**/*'))
    labels = sorted(glob(f'{dataset_dir}/labels/**/*'))

    futures = [
        transfer_color.remote(img, label, random.choice(ref_imgs))
        for img, label in zip(images, labels)
    ]
    _ = [ray.get(future) for future in tqdm(futures)]
    return


if __name__ == '__main__':
    args = _opts()
    run(dataset_dir=args.dataset_dir, ref_images_dir=args.ref_images_dir)

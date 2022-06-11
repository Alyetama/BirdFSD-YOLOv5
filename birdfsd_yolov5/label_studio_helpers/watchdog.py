#!/usr/bin/env python
# coding: utf-8

import argparse
import copy
import imghdr
import shutil
import time
import uuid
from glob import glob
from pathlib import Path

from PIL import Image, UnidentifiedImageError
from dotenv import load_dotenv
from loguru import logger

from birdfsd_yolov5.label_studio_helpers.add_and_sync_new_project import (
    add_new_project, add_and_sync_data_storage)
from birdfsd_yolov5.model_utils.handlers import catch_keyboard_interrupt


class _MissingArgument(Exception):
    pass


class WatchDog:
    """Creates new projects when a change is detected in the source folder.

    This class is used to create a new folder for the images to be stored
    in. The folder name is based on the current time and date. The folder is
    created in the root data folder. The folder name is returned.
    """

    def __init__(self,
                 root_data_folder: str,
                 images_per_folder: int = 1000,
                 debug: bool = False):
        """
        Args:
            root_data_folder (str): The Root folder where the images are
                stored.
            images_per_folder (int): Number of Images per folder.
            debug (bool): Debug mode.

        Returns:
            str: The name of the folder that was created.

        """
        self.root_data_folder = root_data_folder
        self.images_per_folder = images_per_folder
        self.debug = debug

    @staticmethod
    def _create_dummy_data() -> None:
        dummy_projects = ['project-1000', 'project-1001', 'MOVE_ME']

        for project in dummy_projects:
            num_dummies = 1000
            proj_folder = Path(f'dummy/{project}')
            if project == dummy_projects[-2]:
                num_dummies = 200
            elif project == 'MOVE_ME':
                num_dummies = 3600
                proj_folder = Path(project)

            proj_folder.mkdir(exist_ok=True, parents=True)

            for _ in range(num_dummies):
                fname = str(uuid.uuid4())
                Path(proj_folder /
                     Path(f'fake_{str(fname).zfill(4)}.jpg')).touch()

        logger.debug('Created dummy data')
        return

    def validate_image_file(self, file: str) -> str:
        """Validates an image file.

        Args:
            file (str): The path to the image file.

        Returns:
            str: The path to the image file.

        Raises:
            UnidentifiedImageError: If the image file is corrupted.

        """
        if self.debug:
            return file
        try:
            if imghdr.what(file) and Image.open(file):
                return file
        except UnidentifiedImageError:
            time.sleep(1)
            if imghdr.what(file) and Image.open(file):
                return file
            else:
                logger.warning(f'`{file}` is corrupted!')
                shutil.move(
                    file,
                    f'{Path(self.root_data_folder).parent}/data_corrupted')

    def _generate_next_folder_name(self) -> str:
        project_folders = sorted(glob(f'{self.root_data_folder}/project-*'))
        num = str(int(project_folders[-1].split('project-')[-1]) + 1).zfill(4)
        Path(f'{self.root_data_folder}/project-{num}').mkdir()
        return f'{self.root_data_folder}/project-{num}'

    def _refresh_src(self) -> tuple:
        folders = glob(f'{self.root_data_folder}/*')
        project_folders = glob(f'{self.root_data_folder}/project-*')
        new_folders = list(set(folders).difference(project_folders))
        if new_folders:
            logger.debug(f'New folder(s) detected: {new_folders}')
        new_files = []
        for new_folder in new_folders:
            cur_files = [
                x for x in glob(f'{new_folder}/**/*', recursive=True)
                if not Path(x).is_dir()
            ]
            if cur_files:
                new_files.append(cur_files)
        new_files = sum(new_files, [])
        return folders, project_folders, new_folders, new_files

    def _arrange_new_data_files(self) -> None:
        Path(f'{Path(self.root_data_folder).parent}/data_corrupted').mkdir(
            exist_ok=True)

        folders, project_folders, new_folders, new_files = self._refresh_src()

        not_filled_folders = []
        project_folders = sorted(glob(f'{self.root_data_folder}/project-*'))

        for folder in project_folders:
            folder_size = len(glob(f'{folder}/*'))
            if self.images_per_folder > folder_size:
                not_filled_folders.append((folder, folder_size))
                logger.debug(f'Not filled: {folder}, size: {folder_size}')

        if not_filled_folders:
            for folder, folder_size in not_filled_folders:
                for file in new_files:
                    if self.validate_image_file(file):
                        shutil.move(file, folder)
                        if len(glob(f'{folder}/*')) == 1000:
                            break

        folders, project_folders, new_folders, new_files = self._refresh_src()

        chunks = [
            new_files[i:i + self.images_per_folder]
            for i in range(0, len(new_files), self.images_per_folder)
        ]

        for chunk in chunks:
            dst = self._generate_next_folder_name()
            folder_name = Path(dst).name
            for file in chunk:
                if self.validate_image_file(file):
                    shutil.move(file, dst)
            if not self.debug:
                new_project = add_new_project(folder_name)
                _ = add_and_sync_data_storage(new_project['id'],
                                              new_project['title'])

        for empty_folder in new_folders:
            contains_any_file = [
                x for x in glob(f'{empty_folder}/**/*', recursive=True)
                if not Path(x).is_dir()
            ]
            if not contains_any_file:
                shutil.rmtree(empty_folder)

    def watch(self):
        """Monitor changes in the source and organize the data accordingly.

        Returns:
            None

        """
        catch_keyboard_interrupt()
        if self.debug:
            self.root_data_folder = 'dummy'
            self._create_dummy_data()
        else:
            Path(f'{self.root_data_folder}/project-0001').mkdir(exist_ok=True)

        logger.debug('Started watchdog...')

        global_state = glob(f'{self.root_data_folder}/**/*', recursive=True)
        while True:
            local_state = glob(f'{self.root_data_folder}/**/*', recursive=True)
            if global_state != local_state:
                logger.debug('Detected change!')
                global_state = copy.deepcopy(local_state)
                self._arrange_new_data_files()
            time.sleep(60)


if __name__ == '__main__':
    load_dotenv()

    parser = argparse.ArgumentParser()
    parser.add_argument('--root-data-folder',
                        help='Path to the folder where all the data is kept',
                        type=str)
    parser.add_argument('--debug', help='Debug mode', action='store_true')
    parser.add_argument('--images-per-folder',
                        help='Number of images per folder',
                        type=int,
                        default=1000)
    args = parser.parse_args()

    if not args.root_data_folder and not args.debug:
        raise _MissingArgument(
            '`--root_data_folder` is required when not in debug mode!')

    watch_dog = WatchDog(root_data_folder=args.root_data_folder,
                         images_per_folder=args.images_per_folder,
                         debug=args.debug)
    watch_dog.watch()

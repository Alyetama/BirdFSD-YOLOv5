#!/usr/bin/env python
# coding: utf-8

from pathlib import Path


def relative_to_abs() -> None:
    """Updates paths in the dataset_config.yaml file from relative to absolute.

    This function is used to replace the dataset path in the
    `dataset_config.yml` file from a relative to absolute path.
    """
    with open('dataset_config.yml') as f:
        lines = f.readlines()

    existing_path = lines[0].split('path: ')[1].strip()
    replace_with = str(Path('dataset-YOLO').absolute())
    lines[0] = lines[0].replace(existing_path, replace_with)

    with open('dataset_config.yml', 'w') as f:
        f.writelines(lines)

    print(''.join(lines))


if __name__ == '__main__':
    relative_to_abs()

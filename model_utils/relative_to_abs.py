#!/usr/bin/env python
# coding: utf-8

from pathlib import Path


def main():
    """Changes the dataset path from relative to absolute"""
    with open('dataset_config.yml') as f:
        lines = f.readlines()

    if '/' not in lines[0]:
        lines[0] = lines[0].replace('dataset-YOLO',
                                    f'{Path().cwd()}/dataset-YOLO')

    with open('dataset_config.yml', 'w') as f:
        f.writelines(lines)
    print(lines)


if __name__ == '__main__':
    main()

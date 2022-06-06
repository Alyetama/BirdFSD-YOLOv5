# BirdFSD-YOLOv5

Build and train a custom model to identify birds visiting bird feeders.

📖 **[Documentation](https://birdfsd-yolov5.readthedocs.io/en/latest/)**

[![Poetry Build](https://github.com/bird-feeder/BirdFSD-YOLOv5/actions/workflows/poetry-build.yml/badge.svg)](https://github.com/bird-feeder/BirdFSD-YOLOv5/actions/workflows/poetry-build.yml) [![Docker Build](https://github.com/bird-feeder/BirdFSD-YOLOv5/actions/workflows/docker-build.yml/badge.svg)](https://github.com/bird-feeder/BirdFSD-YOLOv5/actions/workflows/docker-build.yml) [![Documentation Status](https://readthedocs.org/projects/birdfsd-yolov5/badge/?version=latest)](https://birdfsd-yolov5.readthedocs.io/en/latest/?badge=latest) [![Supported Python versions](https://img.shields.io/badge/Python-%3E=3.8-blue.svg)](https://www.python.org/downloads/) [![PEP8](https://img.shields.io/badge/Code%20style-PEP%208-orange.svg)](https://www.python.org/dev/peps/pep-0008/) [![Codacy Badge](https://app.codacy.com/project/badge/Grade/8810d995e593497d9bd04afcfdc366ce)](https://www.codacy.com/gh/bird-feeder/BirdFSD-YOLOv5/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=bird-feeder/BirdFSD-YOLOv5&amp;utm_campaign=Badge_Grade)

## Requirements
- 🐍 [python>=3.8](https://www.python.org/downloads/)

## :rocket: Getting started

- First, [fork the repository](https://github.com/bird-feeder/BirdFSD-YOLOv5/fork).
- Then, run:

```shell
git clone https://github.com/bird-feeder/BirdFSD-YOLOv5.git
cd BirdFSD-YOLOv5

# Recommended: create a conda/pyenv environment
pip install -r requirements.txt

poetry build
pip install dist/*.whl

git clone https://github.com/ultralytics/yolov5.git

mv .env.example .env
nano .env  # or with any other editor
```

## :card_file_box: Setup

- To use the GitHub Actions workflows (recommended!), you will need to add every environment variable and its value from `.env` to the `Secrets` of your fork (you can find `Secrets` under `Settings`).

![secrets](https://i.imgur.com/xlVfoxX.png)

<details>
  <summary>Click here to show an example of a new secret</summary>

  ![secrets_ex](https://i.imgur.com/fOKMgHy.png)

</details>

## :wrench: Dataset preparation

- **Option 1:** Run the `JSON to YOLOv5 (data preprocessing)` workflow under github `Actions`.

- **Option 2:** Run it locally with:

  ```shell
  python birdfsd_yolov5/preprocessing/json2yolov5.py
  mv dataset-YOLO/dataset_config.yml .
  python birdfsd_yolov5/model_utils/relative_to_abs.py
  ```

## :zap: Training[^1]

Use the *Colab* notebook: 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bird-feeder/BirdFSD-YOLOv5/blob/main/notebooks/BirdFSD_YOLOv5_train.ipynb)

## :memo: Prediction

- **Option 1:** Run the `Predict` workflow under github `Actions`.
- **Option 2:** Use the *Colab* notebook:

  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bird-feeder/BirdFSD-YOLOv5/blob/main/notebooks/BirdFSDV1_YOLOv5_LS_Predict.ipynb)

## :bookmark: Related

- [BirdFSD-YOLOv5-API](https://github.com/bird-feeder/BirdFSD-YOLOv5-API)
- [webapp-beta](https://github.com/bird-feeder/webapp-beta)


[^1]: [yolov5/wiki/Train-Custom-Data](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)

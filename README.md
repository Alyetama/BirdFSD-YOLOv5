# BirdFSD-YOLOv5

Build and train a custom model to identify birds visiting bird feeders.

📖 **[Documentation](https://birdfsd-yolov5.readthedocs.io/en/latest/)**

[![Poetry Build](https://github.com/bird-feeder/BirdFSD-YOLOv5/actions/workflows/poetry-build.yml/badge.svg)](https://github.com/bird-feeder/BirdFSD-YOLOv5/actions/workflows/poetry-build.yml) [![Docker Build](https://github.com/bird-feeder/BirdFSD-YOLOv5/actions/workflows/docker-build.yml/badge.svg)](https://github.com/bird-feeder/BirdFSD-YOLOv5/actions/workflows/docker-build.yml) [![Documentation Status](https://readthedocs.org/projects/birdfsd-yolov5/badge/?version=latest)](https://birdfsd-yolov5.readthedocs.io/en/latest/?badge=latest) [![Supported Python versions](https://img.shields.io/badge/Python-%3E=3.8-blue.svg)](https://www.python.org/downloads/) [![PEP8](https://img.shields.io/badge/Code%20style-PEP%208-orange.svg)](https://www.python.org/dev/peps/pep-0008/) [![Codacy Badge](https://app.codacy.com/project/badge/Grade/8810d995e593497d9bd04afcfdc366ce)](https://www.codacy.com/gh/bird-feeder/BirdFSD-YOLOv5/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=bird-feeder/BirdFSD-YOLOv5&amp;utm_campaign=Badge_Grade) [![GitHub latest release](https://badgen.net/github/release/bird-feeder/BirdFSD-YOLOv5)](https://github.com/bird-feeder/BirdFSD-YOLOv5/releases) [![Docker Hub](https://badgen.net/badge/icon/Docker%20Hub?icon=docker&label)](https://hub.docker.com/r/alyetama/birdfsd-yolov5) [![GitHub License](https://img.shields.io/github/license/Naereen/StrapDown.js.svg)](https://github.com/bird-feeder/BirdFSD-YOLOv5/blob/main/LICENSE)

## Requirements
- 🐍 [python>=3.8](https://www.python.org/downloads/)

## 🚀 Getting started

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

## 🗃️ Setup

- To use the GitHub Actions workflows (recommended!), you will need to add every environment variable and its value from `.env` to the `Secrets` of your fork (you can find `Secrets` under `Settings`).

<img src="https://i.imgur.com/xlVfoxX.png"  width="720"> 

<details>
  <summary>Click here to show an example of a new secret</summary>

  <img src="https://i.imgur.com/fOKMgHy.png"  width="720"> 

</details>

## 🔧 Dataset preparation

- **Option 1:** Run the `JSON to YOLOv5 (data preprocessing)` workflow under github `Actions`.

- **Option 2:** Run it locally with:

  ```shell
  python birdfsd_yolov5/preprocessing/json2yolov5.py
  mv dataset-YOLO/dataset_config.yml .
  python birdfsd_yolov5/model_utils/relative_to_abs.py
  ```

## ⚡ Training[^1]

Use the *Colab* notebook: 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bird-feeder/BirdFSD-YOLOv5/blob/main/notebooks/BirdFSD_YOLOv5_train.ipynb)

## 📝 Prediction

- **Option 1:** Run the `Predict` workflow under github `Actions`.
- **Option 2:** Use the *Colab* notebook:

  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bird-feeder/BirdFSD-YOLOv5/blob/main/notebooks/BirdFSDV1_YOLOv5_LS_Predict.ipynb)
  
  
## 🐳 Using Docker
```sh
docker pull alyetama/birdfsd-yolov5:latest
```

### Example Usage
```sh
docker run -it --env-file .env alyetama/birdfsd-yolov5 python birdfsd_yolov5/preprocessing/json2yolov5.py
```


## 🔖 Related

- [BirdFSD-YOLOv5-API](https://github.com/bird-feeder/BirdFSD-YOLOv5-API)
- [BirdFSD-YOLOv5-APP](https://github.com/bird-feeder/BirdFSD-YOLOv5-App)


[^1]: [yolov5/wiki/Train-Custom-Data](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)

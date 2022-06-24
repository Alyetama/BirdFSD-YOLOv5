# BirdFSD-YOLOv5

Build and train a custom model to identify birds visiting bird feeders.

📖 **[Documentation](https://birdfsd-yolov5.readthedocs.io/en/latest/)**

[![Poetry Build](https://github.com/bird-feeder/BirdFSD-YOLOv5/actions/workflows/poetry-build.yml/badge.svg)](https://github.com/bird-feeder/BirdFSD-YOLOv5/actions/workflows/poetry-build.yml) [![Docker Build](https://github.com/bird-feeder/BirdFSD-YOLOv5/actions/workflows/docker-build.yml/badge.svg)](https://github.com/bird-feeder/BirdFSD-YOLOv5/actions/workflows/docker-build.yml) [![Documentation Status](https://readthedocs.org/projects/birdfsd-yolov5/badge/?version=latest)](https://birdfsd-yolov5.readthedocs.io/en/latest/?badge=latest) [![Supported Python versions](https://img.shields.io/badge/Python-%3E=3.8-blue.svg)](https://www.python.org/downloads/) [![PEP8](https://img.shields.io/badge/Code%20style-PEP%208-orange.svg)](https://www.python.org/dev/peps/pep-0008/) [![Codacy Badge](https://app.codacy.com/project/badge/Grade/8810d995e593497d9bd04afcfdc366ce)](https://www.codacy.com/gh/bird-feeder/BirdFSD-YOLOv5/dashboard?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=bird-feeder/BirdFSD-YOLOv5&amp;utm_campaign=Badge_Grade) [![GitHub latest release](https://badgen.net/github/release/bird-feeder/BirdFSD-YOLOv5)](https://github.com/bird-feeder/BirdFSD-YOLOv5/releases) [![Docker Hub](https://badgen.net/badge/icon/Docker%20Hub?icon=docker&label)](https://hub.docker.com/r/alyetama/birdfsd-yolov5) [![GitHub License](https://img.shields.io/github/license/Naereen/StrapDown.js.svg)](https://github.com/bird-feeder/BirdFSD-YOLOv5/blob/main/LICENSE)

## Requirements
- 🐍 [python>=3.8](https://www.python.org/downloads/)

## 🚀 Getting started

- First, [fork the repository](https://github.com/bird-feeder/BirdFSD-YOLOv5/fork).
- Enable workflows in your fork:
<img src="https://i.imgur.com/aF5U6J0.png"  width="720"> 

- Then, click on and enable all the workflows that are highlighted wuth a red square in the image below.
<img src="https://i.imgur.com/pj0Fe9e.png"  width="720"> 


- On your machine, run:

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
# See `🌱 Environment Variables` section for details about the environment variables.
```

## 🌱 Environment Variables


| Name                 | Value                                                                                                                   |
|----------------------|-------------------------------------------------------------------------------------------------------------------------|
| TOKEN                | Label-Studio `Access Token`.                                                                                            |
| LS_HOST              | The URL of the label-studio app (e.g., https://label-studio.example.com) – make sure you include `https://` in the URL. |
| DB_CONNECTION_STRING | MongoDB connection string (e.g., `mongodb://mongodb0.example.com:27017`). See [this article](https://www.mongodb.com/docs/manual/reference/connection-string/) for details.                                                                                                |
| DB_NAME              | Name of the main MongoDB database (default: `label_studio`).                                                            |
| S3_ACCESS_KEY        | (Optional) The S3 bucket's `Access Key ID`.                                                                             |
| S3_SECRET_KEY        | (Optional) The S3 bucket's `Secret Key`.                                                                                |
| S3_REGION            | (Optional) The S3 bucket's region (default: `us-east-1`).                                                               |
| S3_ENDPOINT          | (Optional) The S3 bucket's endpoint/URL server.                                                                         |
| EXCLUDE_LABELS       | (Optional) Comma-separated list of labels to exclude from processing (e.g., label a,label b).                           |


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

- [BirdFSD-YOLOv5-APP](https://github.com/bird-feeder/BirdFSD-YOLOv5-App)


[^1]: [yolov5/wiki/Train-Custom-Data](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)

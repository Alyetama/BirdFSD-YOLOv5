# BirdFSD-YOLOv5


## Getting started

- First, [fork the repository](https://github.com/bird-feeder/BirdFSD-YOLOv5/fork).
- Then, run:

```shell
git clone https://github.com/bird-feeder/BirdFSD-YOLOv5.git
cd BirdFSD-YOLOv5

conda create --name yolov5 python=3.8.13 --yes
pip install -r requirements.txt

git clone https://github.com/ultralytics/yolov5.git

mv .env.example .env
nano .env  # or with any other editor
```

## Setup

- To use the GitHub Actions workflows (recommended!), you will need to add every environment variable and its value from `.env` to the `Secrets` of your fork (you can find `Secrets` under `Settings`).

![secrets](https://i.imgur.com/xlVfoxX.png)

<details>
  <summary>Click here to show an example of a new secret</summary>

  ![secrets_ex](https://i.imgur.com/fOKMgHy.png)

</details>


## Dataset preparation

- **Option 1:** Run the `JSON to YOLOv5 (data preprocessing)` workflow under github `Actions`.
- **Option 2:** Run it locally with:

  ```shell
  python json2yolov5.py
  mv dataset-YOLO/dataset_config.yml .
  python utils/relative_to_abs.py
  ```

## Training[^1]

Use the *Colab* notebook: 

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bird-feeder/BirdFSD-YOLOv5/blob/main/notebooks/BirdFSD_YOLOv5_train.ipynb)


## Prediction

- **Option 1:** Run the `Predict` workflow under github `Actions`.
- **Option 2:** Use the *Colab* notebook:

  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/bird-feeder/BirdFSD-YOLOv5/blob/main/notebooks/BirdFSDV1_YOLOv5_LS_Predict.ipynb)


## Related

- [BirdFSD-YOLOv5-API](https://github.com/bird-feeder/BirdFSD-YOLOv5-API)
- [label-studio-workers](https://github.com/bird-feeder/label-studio-workers)
- [webapp-beta](https://github.com/bird-feeder/webapp-beta)


[^1]: [yolov5/wiki/Train-Custom-Data](https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data)

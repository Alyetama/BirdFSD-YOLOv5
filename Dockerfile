FROM python:3.8.12-bullseye

USER root
RUN chown $USER:$USER /
USER $USER
RUN mkdir -p app
WORKDIR /app

RUN apt-get update

RUN git clone https://github.com/ultralytics/yolov5.git

ADD . /app

RUN pip install -r requirements.txt
RUN poetry build
RUN pip install dist/*.whl

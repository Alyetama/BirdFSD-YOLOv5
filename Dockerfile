FROM python:3.8.12-bullseye

RUN mkdir -p app
WORKDIR /app

RUN apt-get update \
	&& apt-get clean \
	&& rm -rf /var/lib/apt/lists/*

RUN git clone https://github.com/ultralytics/yolov5.git

COPY . /app

RUN python -m pip install --upgrade pip
RUN pip install -r requirements.txt
RUN poetry build
RUN pip install dist/*.whl

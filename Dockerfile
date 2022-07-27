FROM python:3.8

COPY requirements.txt /home/requirements.txt

WORKDIR /home

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6  -y

RUN pip install --upgrade pip
RUN pip install -r requirements.txt
RUN mkdir data data/raw data/intermediate data/final

COPY /src /home/app
COPY /configs /home/configs

CMD python app/main.py


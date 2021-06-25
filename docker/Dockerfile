FROM python:3.8.2-slim

RUN apt-get update && \
  apt-get install -y --no-install-recommends \
  apt-utils \
  gcc \
  zsh \
  build-essential && \
  apt-get autoremove -y && \
  apt-get clean -y && \ 
  rm -rf /var/lib/apt/lists/* && \
  mkdir /code
WORKDIR /code
COPY requirements.txt /code/
RUN pip --no-cache-dir install --upgrade pip setuptools && \
  pip --no-cache-dir install -r requirements.txt
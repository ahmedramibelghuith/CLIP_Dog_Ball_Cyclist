# syntax=docker/dockerfile:1
FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    KERAS_HOME=/tmp/keras \
    CUDA_VISIBLE_DEVICES=-1 \
    TF_CPP_MIN_LOG_LEVEL=2

WORKDIR /app

# Installation mise en cache indépendamment du code applicatif.
COPY requirements.txt ./
RUN python -m pip install --no-cache-dir -r requirements.txt \
    && python -m pip check \
    && python -c "import tensorflow as tf; print(tf.__version__)"

RUN groupadd --gid 1000 clip \
    && useradd --uid 1000 --gid clip --create-home clip \
    && mkdir /data /artifacts \
    && chown clip:clip /data /artifacts

COPY main.py ./
COPY src/ ./src/
COPY tests/ ./tests/

USER clip
ENTRYPOINT ["python", "/app/main.py"]
CMD ["--help"]

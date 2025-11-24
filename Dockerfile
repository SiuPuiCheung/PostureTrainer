# syntax=docker/dockerfile:1

FROM python:3.10-slim AS base

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_SERVER_PORT=8501 \
    POSE_TRAINER_WEIGHTS_DIR=/app/models

WORKDIR /app

# System dependencies for OpenCV and media processing
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libgl1 \
        libglib2.0-0 \
        ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt

COPY . .
RUN mkdir -p ${POSE_TRAINER_WEIGHTS_DIR}

EXPOSE 8501

ENTRYPOINT ["streamlit", "run", "app.py"]

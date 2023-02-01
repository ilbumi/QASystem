ARG BASE_IMAGE=pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime
FROM ${BASE_IMAGE}
USER root
RUN apt update && apt install -y --no-install-recommends \
	build-essential \
	git \
	&& rm -rf /var/lib/apt/lists/*
COPY requirements.txt requirements.txt
RUN --mount=type=cache,target=/root/.cache \
	pip3 install -r requirements.txt
COPY *.py /app
WORKDIR /app

EXPOSE 8000
CMD python app.py

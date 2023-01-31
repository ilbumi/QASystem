ARG BASE_IMAGE=pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime
FROM ${BASE_IMAGE}
RUN apt-get update && apt-get install -y --no-install-recommends \
	build-essential \
	git \
	&& rm -rf /var/lib/apt/lists/*
COPY requirements.txt requirements.txt
RUN --mount=type=cache,target=/root/.cache \
	pip3 install -r requirements.txt

RUN mkdir /app
COPY *.py /app
WORKDIR /app

EXPOSE 8000
CMD python app.py
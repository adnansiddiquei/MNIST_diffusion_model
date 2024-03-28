#FROM continuumio/miniconda3
FROM python:3.12

WORKDIR usr/src/app

# Copy everything into the container, into WORKDIR
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

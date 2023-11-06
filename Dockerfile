# syntax=docker/dockerfile:1

FROM python:3.9

WORKDIR /CarBrandClass

RUN apt-get update
RUN apt-get install ffmpeg libsm6 libxext6 -y

# COPY folder หรือ file งาน
COPY ./requirements.txt /CarBrandClass/requirements.txt
RUN pip install --upgrade pip
RUN pip install --no-cache-dir --upgrade -r /CarBrandClass/requirements.txt

COPY ./app /CarBrandClass/app
COPY ./model /CarBrandClass/model

ENV PYTHONPATH "${PYTHONPATH}:/carbrandclass"

# ต้องการใช้ poth 80 สำหรับให้ Containers สื่อสารกันเองใน Docker
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "80"]
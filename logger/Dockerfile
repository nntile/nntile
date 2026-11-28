# @copyright (c) 2022-present Skolkovo Institute of Science and Technology
#                              (Skoltech), Russia. All rights reserved.
#                 2023-present Artificial Intelligence Research Institute
#                              (AIRI), Russia. All rights reserved.
#
# NNTile is software framework for fast training of big neural networks on
# distributed-memory heterogeneous systems with GPU acceleration.
#
# @file logger/Dockerfile
# NNTile 2.0 file.
#
# @version 2.0.0

FROM python:3.11-slim

WORKDIR /workspace

COPY requirements.txt .

RUN pip install -r requirements.txt

COPY . /app

WORKDIR /app

EXPOSE 5001 6006

ENV LOG_DIR=/workspace/logs

ENV SPLIT_HOURS=1

CMD ["python", "-u", "server.py"]

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

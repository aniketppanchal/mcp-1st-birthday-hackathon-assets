FROM python:3.12-slim

COPY --from=ghcr.io/blaxel-ai/sandbox:latest /sandbox-api /usr/local/bin/sandbox-api

RUN apt-get update && apt-get install -y git \
&& rm -rf /var/lib/apt/lists/*

WORKDIR /blaxel

ENV HOME=/blaxel

COPY requirements.txt /blaxel/requirements.txt

RUN pip install --no-cache-dir -r /blaxel/requirements.txt

COPY model_server.py /blaxel/model_server.py

COPY --chmod=755 entrypoint.sh /entrypoint.sh

EXPOSE 7860

ENTRYPOINT ["/entrypoint.sh"]
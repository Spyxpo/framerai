# FramerAI backend image.
#
# Includes Python + torch so the inference bridge can run the model worker
# (model/serve.py) for real inference when a checkpoint is mounted. With no
# checkpoint, the backend returns placeholder responses.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NODE_VERSION=20

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl ca-certificates libsndfile1 \
    && curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Python model dependencies (for the inference worker)
COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

# Model package used by the worker
COPY model/ ./model/

# Backend Node dependencies
COPY backend/package*.json ./backend/
RUN cd backend && npm ci --omit=dev

COPY backend/ ./backend/

WORKDIR /app/backend

ENV MODEL_ENABLED=true \
    PYTHON_BIN=python3 \
    MODEL_PATH=/app/checkpoints/export/framerai_model.pt \
    TOKENIZER_PATH=/app/checkpoints/export/tokenizer

EXPOSE 3001
CMD ["node", "src/index.js"]

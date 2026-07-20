# FramerAI model image: builds, trains, exports, and serves the model.
#
# Build:   docker build -t framerai-model .
# Build a tiny model into a mounted volume:
#   docker run --rm -v framerai_checkpoints:/app/checkpoints framerai-model \
#     --mode all --size tiny
# Train on your own data:
#   docker run --rm -v $(pwd)/data:/app/data -v framerai_checkpoints:/app/checkpoints \
#     framerai-model --mode all --size small --data-dir data
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# libsndfile is required by soundfile for reading and writing audio.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY model/ ./model/
COPY build.py ./

# Default: build, train, and export a tiny model. Override args at run time.
ENTRYPOINT ["python", "build.py"]
CMD ["--mode", "all", "--size", "tiny"]

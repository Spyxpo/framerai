# FramerAI Guide

This guide is a hands-on walkthrough of the FramerAI codebase: how the pieces fit
together, how to run each part, and how to extend them. For a short overview read
the [README](README.md). For contribution mechanics read [CONTRIBUTING.md](CONTRIBUTING.md).

## Table of contents

- [System overview](#system-overview)
- [Repository layout](#repository-layout)
- [Prerequisites](#prerequisites)
- [End-to-end setup](#end-to-end-setup)
- [The model](#the-model)
- [Modalities](#modalities)
- [The training pipeline](#the-training-pipeline)
- [Training on your own data](#training-on-your-own-data)
- [The backend and the inference bridge](#the-backend-and-the-inference-bridge)
- [The website](#the-website)
- [Running with Docker](#running-with-docker)
- [Configuration and environment](#configuration-and-environment)
- [Common workflows](#common-workflows)
- [Troubleshooting](#troubleshooting)
- [Where to go next](#where-to-go-next)

## System overview

FramerAI is a single multimodal model served through a small stack:

```txt
+-----------+     HTTP / WebSocket     +-----------+     JSON over stdio      +-----------+
|  Website  | <--------------------->  |  Backend  | <--------------------->  |  Worker   |
| (React)   |                          | (Express) |                          | (Python)  |
+-----------+                          +-----------+                          +-----------+
                                                                                    |
                                                                              FramerModel
```

- The model defines the architecture, tokenizer, and training and inference code.
- The backend exposes REST endpoints and a WebSocket stream, and drives the model
  through a Python inference worker.
- The website is the chat and generation interface.

Each layer can be developed independently. During development you typically run
all three at once.

## Repository layout

| Path | Purpose |
|------|---------|
| `model/framer.py` | The unified `FramerModel` that ties the modules together. |
| `model/modules/` | Transformer, vision/audio encoders, and image/video/audio generators. |
| `model/tokenizer/` | Byte-level BPE tokenizer with multimodal special tokens. |
| `model/configs/` | Model size and hyperparameter configuration. |
| `model/data.py` | Local-corpus datasets (text, image-caption, audio-caption). |
| `model/generate.py` | Inference and sampling utilities. |
| `model/serve.py` | Inference worker driven by the backend over stdin/stdout. |
| `build.py` | Command line entry point to build, train, and export models. |
| `train.sh` | Convenience wrapper around the full pipeline. |
| `backend/src/` | Express server, routes, services, and the Python bridge. |
| `website/src/` | React components, hooks, services, and styles. |
| `data/` | Your local training data. |
| `Dockerfile`, `docker-compose.yml` | Container build for the model and stack. |

## Prerequisites

- Python 3.10 or newer.
- Node.js 18 or newer and npm.
- A CUDA-capable GPU is recommended for training. CPU works for small models and inference but is slow.
- Optional: `soundfile` or `torchaudio` for reading and writing audio files (installed by `requirements.txt`).

## End-to-end setup

```bash
git clone https://github.com/Spyxpo/framerai.git
cd framerai
```

### 1. Model environment

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python build.py --mode all --size tiny
```

### 2. Backend

```bash
cd backend
npm install
cp .env.example .env
npm run dev
```

The API server listens on `http://localhost:3001`.

### 3. Website

```bash
cd website
npm install
npm run dev
```

The web app opens at `http://localhost:5173`.

## The model

`FramerModel` in `model/framer.py` composes several modules:

- Transformer backbone: an autoregressive decoder using RoPE, SwiGLU, and RMSNorm.
- Vision encoder: a ViT-style encoder that turns images into patch embeddings.
- Audio encoder: a mel-spectrogram front-end and transformer that turns audio into embeddings.
- Diffusion module: a U-Net with cross-attention for text-conditioned images.
- Video generator: spatial-temporal diffusion with a 3D U-Net.
- Audio generator: text-conditioned mel diffusion with Griffin-Lim reconstruction.
- Multimodal projector: aligns vision and audio embeddings with the language model space.

Model sizes are named presets in `model/configs/presets.py` (run
`python build.py --list-presets` to print total/active parameter counts). LLM-core
counts (total vs. active per token):

| Preset | d_model | Layers | Heads (Q/KV) | Total | Active |
|--------|---------|--------|--------------|-------|--------|
| `framer-tiny` | 256 | 6 | 8 / 4 | ~19M | ~19M |
| `framer-small` | 768 | 12 | 12 / 4 | ~142M | ~142M |
| `framer-medium` | 1024 | 24 | 16 / 8 | ~429M | ~429M |
| `framer-large` | 2048 | 24 | 16 / 8 | ~1.2B | ~1.2B |
| `framer-8b` | 4096 | 32 | 32 / 8 | ~7.2B | ~7.2B |
| `framer-30b-a3b` | 2048 | 28 | 16 / 4 | ~34B | ~3.0B |
| `framer-1t-a32b` | 8192 | 64 | 64 / 8 | ~999B | ~32B |

Dense presets train on a single consumer GPU or CPU; the MoE presets (`*-a*b`)
scale total parameters via sparse experts and need multi-GPU hardware to train.
`--size tiny|small|medium|large` remains as a legacy alias.

To experiment with a custom shape, pass overrides:

```bash
python build.py --mode build --d-model 512 --n-layers 12 --n-heads 8
```

## Modalities

Modality routing is implicit. In `FramerModel.forward`, an argument being present
selects the path:

- `images` / `audio` (inputs) are encoded and prepended to the token sequence.
- `input_ids` runs the transformer and, with `labels`, computes the text loss.
- `target_images` / `target_video` / `target_audio` (outputs) run the matching
  text-conditioned diffusion decoder and add a loss.

At inference time, `FramerGenerator` (`model/generate.py`) exposes explicit
methods: `generate_text` (optionally conditioned on an image or audio),
`generate_image`, `generate_video`, `generate_audio`, `generate_code`, and
`transcribe` (audio to text).

The tokenizer adds `<audio>` and `<audio_end>` alongside the existing image,
video, and code special tokens.

## The training pipeline

`build.py` is the single entry point with four modes:

| Mode | Action |
|------|--------|
| `build` | Initialize weights and train the tokenizer. |
| `train` | Train from scratch on the local corpus. |
| `export` | Export the model for serving. |
| `all` | Run build, train, and export in sequence. |

Example:

```bash
python build.py --mode all --size tiny --data-dir data --max-steps 10000
```

`train.sh` wraps install, build/train/export, and serving. Run `./train.sh --help`
for options.

## Training on your own data

FramerAI trains from scratch on a local corpus - there are no teacher models.
Put files under `data/` (scanned recursively):

- `*.txt` - plain text, split on blank lines.
- `*.jsonl` - `{"text": ...}`, `{"prompt": ..., "response": ...}`, or `{"instruction": ..., "output": ...}`.
- image-caption `*.jsonl` - `{"image": path, "caption": ...}`.
- audio-caption `*.jsonl` - `{"audio": path, "text": ...}`.

The tokenizer and language model train on all text sources. To also train the
image and audio generators on caption pairs:

```bash
python build.py --mode all --size small --data-dir data --train-modalities
```

The loaders live in `model/data.py`; the data layout is documented in
[data/README.md](data/README.md).

## The backend and the inference bridge

The backend is an Express server under `backend/src/`:

- `routes/` defines REST endpoints for chat, generation, and health.
- `services/model.js` routes each request to the model or a placeholder.
- `services/pythonBridge.js` spawns and talks to the Python inference worker.
- `services/websocket.js` handles real-time token streaming.

The bridge lazily spawns `python -m model.serve` when `MODEL_ENABLED=true` and a
checkpoint exists at `MODEL_PATH`. Requests are sent as JSON lines and answered by
`FramerGenerator`. If the worker is unavailable, the backend returns placeholder
responses, so the stack always runs. Generated media is written to
`backend/uploads/generated` and served under `/uploads/generated`.

Key endpoints:

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check and capabilities. |
| POST | `/api/chat/conversations/:id/messages` | Send a message. |
| POST | `/api/generate/image` | Generate an image from text. |
| POST | `/api/generate/video` | Generate a video from text. |
| POST | `/api/generate/audio` | Generate audio/speech from text. |
| POST | `/api/generate/code` | Generate code. |
| POST | `/api/generate/understand` | Analyze an uploaded image. |
| POST | `/api/generate/transcribe` | Transcribe an uploaded audio file. |
| WS | `/ws` | Real-time streaming. |

## The website

The website is a React app built with Vite under `website/src/`:

- `components/Chat/` holds the chat view, message bubbles, and the modality mode
  buttons (text, code, image, video, audio) plus an audio-upload control that
  transcribes to text.
- `components/Chat/MessageBubble.jsx` renders text, code blocks, and image, video,
  and audio players for generated media.
- `services/api.js` and `services/websocket.js` are the REST and WebSocket clients.
- `hooks/useChat.js` manages conversation state and streaming.

Development commands:

```bash
npm run dev      # start the dev server
npm run build    # produce a production build in website/dist
npm run preview  # preview the production build locally
```

## Running with Docker

```bash
# Build the model into the shared checkpoints volume
docker compose run --rm trainer

# Start the backend and website
docker compose up backend website
```

- Website: `http://localhost:8080`
- Backend: `http://localhost:3001`

The `trainer` service (root `Dockerfile`) builds the model. The `backend` image
includes Python and torch so the inference worker runs inside the container; it
reads the checkpoint from the shared volume. The `website` image builds the React
app and serves it with nginx, proxying `/api`, `/uploads`, and `/ws` to the backend.

Build the model image directly:

```bash
docker build -t framerai-model .
docker run --rm -v $(pwd)/data:/app/data -v framerai_checkpoints:/app/checkpoints \
  framerai-model --mode all --size tiny --data-dir data
```

## Configuration and environment

- The backend reads configuration from `backend/.env` (copy `backend/.env.example`).
  Notable keys: `MODEL_ENABLED`, `MODEL_PATH`, `TOKENIZER_PATH`, `PYTHON_BIN`, `DEVICE`.
- The root `.env` is git-ignored. Never commit secrets.

## Common workflows

Build, train, and serve a small model end to end:

```bash
python build.py --mode all --size small --data-dir data --max-steps 20000
cd backend && npm run dev
cd website && npm run dev
```

Run the same checks CI runs before opening a pull request:

```bash
ruff check model build.py
python -m compileall -q model build.py
cd backend && npm ci && for f in $(find src -name '*.js'); do node --check "$f"; done
cd website && npm ci && npm run build
```

## Troubleshooting

- Out of memory during training: lower `--batch-size`, use a smaller `--size`, or train on CPU with `--device cpu`.
- Audio file loading fails: install `soundfile` or `torchaudio` (both are in `requirements.txt`).
- Backend returns placeholders: confirm a checkpoint exists at `MODEL_PATH` and `MODEL_ENABLED=true`.
- Website shows connection errors: confirm the backend is running on the expected port.

## Where to go next

- Pick up an item from [TODOS.md](TODOS.md) or the open issues.
- Read [CONTRIBUTING.md](CONTRIBUTING.md) for the branching model and pull request flow.
- Review [SECURITY.md](SECURITY.md) before reporting a vulnerability.

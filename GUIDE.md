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
- [The training pipeline](#the-training-pipeline)
- [Knowledge distillation](#knowledge-distillation)
- [The backend](#the-backend)
- [The website](#the-website)
- [Configuration and environment](#configuration-and-environment)
- [Common workflows](#common-workflows)
- [Troubleshooting](#troubleshooting)
- [Where to go next](#where-to-go-next)

## System overview

FramerAI is a single multimodal model served through a small stack:

```txt
+-----------+        HTTP / WebSocket        +-----------+        in-process        +-----------+
|  Website  |  <------------------------->   |  Backend  |  <------------------->   |   Model   |
| (React)   |                                | (Express) |                          | (Python)  |
+-----------+                                +-----------+                          +-----------+
```

- The model defines the architecture, tokenizer, and training and inference code.
- The backend exposes REST endpoints and a WebSocket stream for token streaming.
- The website is the chat and generation interface.

Each layer can be developed independently. During development you typically run
all three at once.

## Repository layout

| Path | Purpose |
|------|---------|
| `model/framer.py` | The unified `FramerModel` that ties the modules together. |
| `model/modules/` | Transformer, vision encoder, diffusion, video generator, and projector. |
| `model/tokenizer/` | Byte-level BPE tokenizer with multimodal special tokens. |
| `model/configs/` | Model size and training configuration definitions. |
| `model/distillation/` | Provider clients, data generation, and the distillation trainer. |
| `model/generate.py` | Inference and sampling utilities. |
| `build.py` | Command line entry point for build, train, distill, and export. |
| `train.sh` | Convenience wrapper around common training runs. |
| `backend/src/` | Express server, routes, services, and middleware. |
| `website/src/` | React components, hooks, services, and styles. |
| `.github/` | CI workflows and issue and pull request templates. |

## Prerequisites

- Python 3.10 or newer.
- Node.js 18 or newer and npm.
- A CUDA-capable GPU is recommended for training. CPU works for small models and inference but is slow.
- For gated teacher models such as Llama, a HuggingFace account and token.

## End-to-end setup

Clone the repository and set up each layer.

```bash
git clone https://github.com/Spyxpo/framerai.git
cd framerai
```

### 1. Model environment

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
python build.py --mode build --size tiny
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

The web app opens at `http://localhost:5173` and talks to the backend.

## The model

`FramerModel` in `model/framer.py` composes several modules:

- Transformer backbone: an autoregressive decoder using RoPE, SwiGLU, and RMSNorm.
- Vision encoder: a ViT-style encoder that turns images into patch embeddings.
- Diffusion module: a U-Net with cross-attention for text-conditioned images.
- Video generator: spatial-temporal diffusion with a 3D U-Net.
- Multimodal projector: aligns vision embeddings with the language model space.

Model sizes are defined in `model/configs/`:

| Size | d_model | Layers | Heads | Parameters |
|------|---------|--------|-------|------------|
| Tiny | 256 | 6 | 4 | ~25M |
| Small | 512 | 12 | 8 | ~150M |
| Medium | 1024 | 24 | 16 | ~600M |
| Large | 2048 | 32 | 32 | ~2.5B |

To experiment with a custom shape, pass overrides to `build.py`:

```bash
python build.py --mode build \
  --d-model 512 --n-layers 12 --n-heads 8
```

## The training pipeline

`build.py` is the single entry point. It has four modes:

| Mode | Action |
|------|--------|
| `build` | Initialize weights and save a fresh model. |
| `train` | Train from scratch on generated or provided data. |
| `distill` | Train the student using teacher logits. |
| `export` | Export the model for serving. |
| `all` | Run build, train, and export in sequence. |

Example full run for a tiny model:

```bash
python build.py --mode all --size tiny --max-steps 10000
```

`train.sh` wraps the most common invocations so you do not have to remember every
flag. Read it to see the recommended defaults.

## Knowledge distillation

Distillation trains FramerAI using open-source teacher models pulled from
HuggingFace. Everything runs locally.

The workflow is two steps:

1. Generate training data from one or more teachers.
2. Distill that knowledge into the student using soft labels.

```bash
# Generate data from every recommended teacher, loaded one at a time
python build.py --mode generate-data \
  --teacher-model all --quantize 4bit --num-samples 10000 --conversations

# Distill into a small student
python build.py --mode build --size small
python build.py --mode distill --size small \
  --teacher-model all --quantize 4bit --max-steps 100000
```

Teacher shorthands, VRAM requirements, multi-teacher grouping, and RoPE context
extension are documented in the [README](README.md#knowledge-distillation). Use
`--quantize 4bit` to fit large teachers on consumer GPUs, or `--teacher-device cpu`
to trade speed for memory.

## The backend

The backend is an Express server under `backend/src/`:

- `routes/` defines the REST endpoints for chat, generation, and health.
- `services/model.js` is the interface to the model.
- `services/websocket.js` handles real-time token streaming.
- `index.js` wires everything together and starts the server.

Key endpoints:

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check and capabilities. |
| POST | `/api/chat/conversations` | Create a conversation. |
| GET | `/api/chat/conversations` | List conversations. |
| POST | `/api/chat/conversations/:id/messages` | Send a message. |
| POST | `/api/generate/image` | Generate an image from text. |
| POST | `/api/generate/video` | Generate a video from text. |
| POST | `/api/generate/code` | Generate code. |
| POST | `/api/generate/understand` | Analyze an uploaded image. |
| WS | `/ws` | Real-time streaming. |

Run the server in watch mode with `npm run dev`, or in production mode with
`npm start`.

## The website

The website is a React app built with Vite under `website/src/`:

- `components/` holds the chat, sidebar, and code block views.
- `hooks/useChat.js` manages conversation state.
- `services/` contains the REST and WebSocket clients.
- `styles/` holds the CSS.

Development commands:

```bash
npm run dev      # start the dev server
npm run build    # produce a production build in website/dist
npm run preview  # preview the production build locally
```

## Configuration and environment

- The backend reads configuration from `backend/.env`. Copy `backend/.env.example`
  and adjust values such as the port and model settings.
- Gated teacher downloads need a token: `export HF_TOKEN=hf_...`.
- The root `.env` is ignored by git. Never commit secrets.

## Common workflows

Build, distill, and serve a small model end to end:

```bash
# 1. Build the student
python build.py --mode build --size small

# 2. Generate data and distill
python build.py --mode generate-data --teacher-model all --quantize 4bit --num-samples 20000 --conversations
python build.py --mode distill --size small --teacher-model all --quantize 4bit --max-steps 100000

# 3. Export for serving
python build.py --mode export --size small

# 4. Start backend and website
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

- Out of memory during distillation: lower `--num-samples`, add `--quantize 4bit`,
  or move the teacher to CPU with `--teacher-device cpu`.
- Gated model download fails: confirm `HF_TOKEN` is set and the model license is accepted.
- Backend cannot reach the model: confirm a built model exists and `backend/.env` is configured.
- Website shows connection errors: confirm the backend is running on the expected port.

## Where to go next

- Pick up an item from [TODOS.md](TODOS.md) or the open issues.
- Read [CONTRIBUTING.md](CONTRIBUTING.md) for the branching model and pull request flow.
- Review [SECURITY.md](SECURITY.md) before reporting a vulnerability.

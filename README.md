<p align="center">
  <img src="images/logo.png" alt="FramerAI" width="200" />
</p>

<h1 align="center">FramerAI</h1>

<p align="center">
  <strong>An open-source, self-contained multimodal model.</strong><br/>
  Understands and generates text, code, images, video, and audio - trained from scratch, no external teacher models.
</p>

<p align="center">
  <a href="https://github.com/Spyxpo/framerai/actions/workflows/ci.yml"><img src="https://github.com/Spyxpo/framerai/actions/workflows/ci.yml/badge.svg" alt="CI" /></a>
  <a href="https://github.com/Spyxpo/framerai/actions/workflows/codeql.yml"><img src="https://github.com/Spyxpo/framerai/actions/workflows/codeql.yml/badge.svg" alt="CodeQL" /></a>
  <a href="https://codecov.io/gh/Spyxpo/framerai"><img src="https://codecov.io/gh/Spyxpo/framerai/branch/stable/graph/badge.svg" alt="Coverage" /></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-MIT-blue.svg" alt="License: MIT" /></a>
  <a href="CONTRIBUTING.md"><img src="https://img.shields.io/badge/PRs-welcome-brightgreen.svg" alt="PRs Welcome" /></a>
  <img src="https://img.shields.io/badge/python-3.10%2B-blue.svg" alt="Python 3.10+" />
  <img src="https://img.shields.io/badge/node-20%2B-green.svg" alt="Node 20+" />
</p>

---

## Overview

FramerAI is a single model that combines a transformer backbone, vision and audio
encoders, and diffusion decoders for images, video, and audio. It is trained from
scratch on your own local data - there is no dependence on external teacher models.

The repository ships the full stack: the model and training pipeline (Python), a
REST and WebSocket API (Node/Express), and a chat interface (React).

## Contents

- [Features](#features)
- [Architecture](#architecture)
- [Quick start](#quick-start)
- [Run with Docker](#run-with-docker)
- [Training on your own data](#training-on-your-own-data)
- [Model sizes](#model-sizes)
- [API endpoints](#api-endpoints)
- [Inference bridge](#inference-bridge)
- [Cognition layer](#cognition-layer)
- [build.py usage](#buildpy-usage)
- [Project structure](#project-structure)
- [Tests](#tests)
- [Documentation](#documentation)
- [Contributing](#contributing)
- [License](#license)

## Features

- **Text generation** - chat and Q&A with autoregressive decoding (top-k, top-p, temperature).
- **Code generation** - lower-temperature sampling for more deterministic output.
- **Image generation** - text-to-image via a latent diffusion transformer with rectified flow
  and classifier-free guidance, at 512x512 by default and any aspect ratio on request
  (the small presets keep a pixel-space U-Net).
- **Video generation** - text-to-video via a causal 3D VAE and a spacetime diffusion
  transformer, with variable duration, resolution, and frame rate.
- **Audio generation** - text-to-audio and speech via discrete acoustic tokens from a residual
  vector-quantized codec, decoded through a neural vocoder, with speaker conditioning.
- **Image understanding** - a vision encoder with dynamic high-resolution tiling, placed
  inline with the text rather than prepended to it.
- **Audio understanding** - an audio encoder transcribes and describes uploaded audio.
- **Streaming** - WebSocket-based real-time token streaming.
- **BPE tokenizer** - byte-level tokenizer with multimodal special tokens.
- **From-scratch training** - trains on a local corpus you provide; no API keys, no teacher models.
- **Cognition layer** *(optional)* - a persistent mind around the model: episodic and semantic
  memory that decays and consolidates, curiosity from novelty and learning progress, an
  affective state that changes how it decodes, a self-model, and sleep. See
  [Cognition layer](#cognition-layer).
- **Live senses** *(optional)* - camera and microphone streamed into the same loop, gated by
  change so a static scene does not flood memory.
- **Language-aware** - script detection across the world's writing systems, per-language
  memory and competence, and replies asked for in the language it was addressed in.

## Architecture

FramerAI combines several neural architectures into one unified model:

- **Transformer backbone** - autoregressive decoder with **grouped-query attention (GQA)**,
  fused scaled-dot-product (flash / memory-efficient) attention, an **incremental KV cache**
  for O(n) decoding, RoPE with **context-extension scaling** (linear, NTK, or YaRN), SwiGLU,
  RMSNorm, and optional QK-normalization for stability at scale.
- **Mixture-of-Experts (MoE)** - sparse top-k routed experts (with optional always-on shared
  experts and a load-balancing auxiliary loss) let *total* parameters reach a trillion while
  *active* per-token compute stays small. See [Model sizes](#model-sizes).
- **Vision encoder** - ViT-style encoder with patch embeddings for image understanding.
- **Audio encoder** - mel-spectrogram front-end and transformer for audio understanding.
- **Image decoder** - a KL-VAE compressing 8x into a latent grid plus a **diffusion
  transformer** with adaLN-zero timestep and text conditioning, trained on a **rectified-flow**
  objective and sampled with a 20-50 step ODE solver and **classifier-free guidance** against a
  learned null-context embedding. Selected by `image_gen_arch`; `unet` keeps the original
  pixel-space U-Net for the laptop-scale presets.
- **Video decoder** - a **causal 3D VAE** compressing 4x in time and 8x in space, plus a
  **spacetime diffusion transformer** with factorised spatial and temporal attention and
  frame-rate conditioning. Causal convolutions mean frame *t* never sees frame *t+1*, which is
  what makes variable duration and streaming decode possible. Selected by `video_gen_arch`.
- **Audio decoder** - a **residual vector-quantized codec** at 24 kHz turning audio into
  discrete tokens the language model predicts directly, and an **inverse-STFT neural vocoder**
  that reconstructs phase instead of guessing it back. Speaker conditioning from a reference
  clip, and an optional CTC head so transcription is trained rather than hoped for. Selected by
  `audio_gen_arch`; `mel_diffusion` keeps the original spectrogram path.
- **Multimodal projector** - aligns vision and audio embeddings with the language model space.

Input modalities are encoded and, under `mm_token_placement="interleaved"`, written into
placeholder positions so an image sits exactly where it was mentioned rather than ahead of the
whole prompt. High-resolution images are split into encoder-sized tiles plus a global
thumbnail, so effective resolution grows with the tile count while attention cost stays
per-tile. Output modalities are generated by text-conditioned decoders sharing the language
model's hidden dimension.

Every part scales together. The large presets grow the perception and generation towers
alongside the backbone, so `framer-1t-a32b` is a trillion parameters of text, code, image,
video, and audio in **one** model rather than a large language model with small attachments.
`--text-only` builds just the LLM core, which is a build-scope option for cheap text
experiments, not what the flagship is.

## Quick start

Requirements: Python 3.10 or newer for the model, Node 20.19 or newer for the website
(the build tooling requires it), and Node 18 or newer for the backend.

### 1. Build the model

```bash
# Install Python dependencies
pip install -r requirements.txt

# Build, train, and export a tiny model
python build.py --mode all --size tiny
```

### 2. Start the backend

```bash
cd backend
npm install
cp .env.example .env
npm run dev
```

The API server starts at `http://localhost:3001`. It runs real inference when a
trained checkpoint exists (see [Inference bridge](#inference-bridge)), and returns
placeholder responses otherwise.

### 3. Start the frontend

```bash
cd website
npm install
npm run dev
```

The web app opens at `http://localhost:5173`.

## Run with Docker

Docker builds and runs the whole stack. The model is trained into a shared volume,
then the backend and website serve it.

```bash
# 1. Build the model into the shared checkpoints volume
docker compose run --rm trainer

# 2. Start the backend and website
docker compose up backend website
```

- Website: `http://localhost:8080`
- Backend: `http://localhost:3001`

Build a specific size or train on your own data:

```bash
docker compose run --rm trainer --mode all --size small --data-dir data
```

You can also build and run the model image directly:

```bash
docker build -t framerai-model .
docker run --rm -v $(pwd)/data:/app/data -v framerai_checkpoints:/app/checkpoints \
  framerai-model --mode all --size tiny --data-dir data
```

## Training on your own data

FramerAI learns from a local corpus. Drop files under `data/` and train:

```bash
python build.py --mode all --size small --data-dir data --max-steps 10000
```

Supported formats:

- `*.txt` - plain text, split on blank lines into samples.
- `*.jsonl` - `{"text": ...}`, `{"prompt": ..., "response": ...}`, or `{"instruction": ..., "output": ...}`.
- image-caption `*.jsonl` - `{"image": path, "caption": ...}` (with `--train-modalities`).
- audio-caption `*.jsonl` - `{"audio": path, "text": ...}` (with `--train-modalities`).

To also train the image and audio generators on caption pairs:

```bash
python build.py --mode all --size small --data-dir data --train-modalities
```

See [data/README.md](data/README.md) for the full data layout and examples.

## Model sizes

Presets live in a named registry (`model/configs/presets.py`) and scale from
laptop-trainable dense models up to a trillion-parameter multimodal MoE.

**Text** is the transformer backbone: total parameters versus the parameters one text token
actually routes through (they differ only for MoE). **Multimodal** is the vision and audio
encoders plus the image, video, and audio diffusion decoders. **Model total** is the whole
model — the number that describes FramerAI as a multimodal system.

| Preset | d_model | Layers | Heads (Q/KV) | Text | Active/token | Multimodal | Model total |
|--------|---------|--------|--------------|------|--------------|------------|-------------|
| `framer-tiny`      | 256  | 6  | 8 / 4  | ~19M   | ~19M   | ~20M   | ~39M   |
| `framer-small`     | 768  | 12 | 12 / 4 | ~142M  | ~142M  | ~94M   | ~236M  |
| `framer-medium`    | 1024 | 24 | 16 / 8 | ~429M  | ~429M  | ~512M  | ~941M  |
| `framer-large`     | 2048 | 24 | 16 / 8 | ~1.2B  | ~1.2B  | ~2.5B  | ~3.7B  |
| `framer-3b`        | 2560 | 32 | 20 / 4 | ~2.3B  | ~2.3B  | ~549M  | ~2.9B  |
| `framer-8b`        | 4096 | 32 | 32 / 8 | ~7.2B  | ~7.2B  | ~596M  | ~7.8B  |
| `framer-tiny-moe`  | 256  | 6  | 8 / 4  | ~33M   | ~21M   | ~20M   | ~53M   |
| `framer-30b-a3b`   | 2048 | 28 | 16 / 4 | ~34B   | ~3.0B  | ~536M  | ~34B   |
| `framer-160b-a16b` | 4096 | 48 | 32 / 8 | ~152B  | ~15B   | ~5.4B  | ~157B  |
| `framer-200b-a20b` | 5120 | 48 | 40 / 8 | ~193B  | ~20B   | ~11.5B | ~205B  |
| `framer-1t-a32b`   | 8192  | 64 | 64 / 8 | ~983B  | ~32B   | ~21.1B | ~1.00T |
| `framer-2t-a49b`   | 10240 | 84 | 80 / 8 | ~1.96T | ~49B   | ~39.3B | **~2.00T** |

`framer-2t-a49b` is the all-modality flagship: 2.00T parameters covering text, code, image,
video, and audio, of which ~49B are active for any given text token. The four largest
presets scale their perception and generation towers with the backbone; the smaller ones
keep laptop-sized towers.

Its 384 experts are deliberately fine-grained rather than fewer and wider: total parameters
scale with the expert count while active parameters scale with top-k, and 384 (3 x 128)
divides evenly across 8, 16, 32, 64, and 128-way expert-parallel meshes.

Legacy `--size tiny|small|medium|large` still works as an alias for the `framer-*` presets.

### Sizing a model yourself

Every figure above comes from the built-in estimator, which allocates nothing — the text core
is computed analytically and the multimodal towers are constructed on the meta device. A
trillion-parameter definition can therefore be sized on a laptop in under a second:

```console
$ python build.py --preset framer-2t-a49b --estimate
Preset: framer-2t-a49b
  d_model=10240 layers=84 heads=80/8 seq=16384
  MoE: 384 experts, top-4, 1 shared, expert_d_ff=2048, 4 dense layers first

  Text backbone          1.96T total      49.40B active/token
    vision_encoder        12.89B
    vision_projector     146.86M
    audio_encoder          5.45B
    audio_projector      136.38M
    image_vae            126.03M
    image_diffusion       10.23B
    video_vae            236.11M
    video_diffusion        7.33B
    audio_codec            2.69B
    audio_vocoder         78.75M
    audio_diffusion            0
  Multimodal total      39.32B

  MODEL TOTAL            2.00T parameters
  Weights (bf16)        3727.8 GiB
  Training state       29822.0 GiB (weights + fp32 master + AdamW moments)
```

`python build.py --list-presets` prints the whole ladder, and `estimate_params(config)` returns
the same numbers programmatically.

> **Reality check.** The small and mid presets train on a single consumer GPU or CPU. The large
> MoE presets are *correct, estimable definitions*: `framer-2t-a49b` needs roughly 3.6 TiB just
> to hold its weights in bf16, and around 29 TiB with the optimizer state, so it is a
> multi-node preset that no single machine can even *construct*, let alone train. Materialising
> it needs deferred initialization, sharded checkpointing, and expert-parallel placement, none
> of which single-device code paths provide. The code path — MoE routing, GQA, the KV cache,
> FSDP2 sharding — is what makes that scale reachable, and the whole architecture can be
> validated locally at `framer-tiny-moe`.
>
> Parameters are the ceiling, not the quality. Frontier-level output additionally requires
> training compute on the order of 10^25–10^26 FLOPs and licensed corpora at petabyte scale
> for image, video, and speech. This repository provides the architecture, the training path,
> and the evaluation to measure progress; it does not provide trained frontier weights. See
> [TODOS.md](TODOS.md) for the per-modality architecture roadmap.

## Image size and aspect ratio

Images are **512x512** by default. A request can ask for a different shape three ways, and the
more explicit one always wins:

1. **Explicit dimensions** — `width` and `height`, given together.
2. **A named aspect ratio** at a size tier — `aspect` and `tier`.
3. **The prompt itself** — "make it 16:9", "1024x768", "a widescreen shot", "phone wallpaper",
   "a 4k panorama".

Otherwise the configured default applies. The response reports the resolved `width`, `height`,
`aspect`, and a `source` of `explicit`, `prompt`, or `default`, so it is always clear what was
understood rather than guessed at silently.

Every named ratio resolves to a bucket holding roughly the same pixel count, so shape does not
change how long generation takes. At the 512 tier:

| Ratio | Size | Ratio | Size |
|-------|------|-------|------|
| `1:1` | 512 x 512 | `16:9` | 688 x 384 |
| `4:3` | 592 x 448 | `9:16` | 384 x 688 |
| `3:4` | 448 x 592 | `21:9` | 784 x 336 |
| `3:2` | 624 x 416 | `2:3` | 416 x 624 |

Tiers of 256, 512, 768, and 1024 rescale the same table. Both dimensions are always a multiple
of 16, because the latent grid must divide by `vae_downsample * dit_patch_size`; an off-bucket
request is snapped to the nearest legal size and the response says so via `snapped`.

A sizing phrase recognised in the prompt is removed from the text used for conditioning, but
only when it reads as an instruction. "a woman in portrait orientation" generates a 416x624
image of "a woman"; "a portrait of a woman" also generates 416x624, but keeps the full prompt,
because there the word is the subject rather than the instruction.

```bash
curl -s localhost:3001/api/generate/image \
  -H 'content-type: application/json' \
  -d '{"prompt": "a red bicycle", "aspect": "21:9", "tier": 1024}'
```

Arbitrary aspect ratios need `image_gen_arch="latent_dit"`. The pixel-space U-Net generates
square images only and says so rather than producing something misshapen.

## API endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check and capabilities |
| POST | `/api/chat/conversations` | Create conversation |
| GET | `/api/chat/conversations` | List conversations |
| POST | `/api/chat/conversations/:id/messages` | Send message |
| POST | `/api/generate/image` | Generate image from text (`width`+`height`, or `aspect`+`tier`, or `seed`; `resolution` is the deprecated square-only alias) |
| POST | `/api/generate/video` | Generate video from text |
| POST | `/api/generate/audio` | Generate audio/speech from text |
| POST | `/api/generate/code` | Generate code |
| POST | `/api/generate/understand` | Analyze an uploaded image |
| POST | `/api/generate/transcribe` | Transcribe an uploaded audio file |
| WS | `/ws` | Real-time streaming |

### Errors

Request bodies, path parameters, and uploads are validated before they reach
the model. Every failure returns the same shape:

```json
{
  "error": "Request validation failed",
  "code": "VALIDATION_ERROR",
  "details": [{ "field": "prompt", "message": "is required" }]
}
```

`error` is a readable message, `code` is stable enough to branch on
(`VALIDATION_ERROR`, `NOT_FOUND`, `INVALID_JSON`, `PAYLOAD_TOO_LARGE`,
`UPLOAD_ERROR`, `RATE_LIMITED`, `INTERNAL_ERROR`), and `details` is only
present when one or more fields failed validation.

### Limits

Requests are rate limited per client address and payloads are capped. Every
limit is configurable in `backend/.env`:

| Setting | Default | Applies to |
|---------|---------|------------|
| `RATE_LIMIT_WINDOW_MS` | `60000` | Window both limiters count within |
| `RATE_LIMIT_MAX` | `300` | All `/api` traffic |
| `GENERATE_RATE_LIMIT_MAX` | `20` | Generation routes, chat messages, and WebSocket chat frames |
| `JSON_BODY_LIMIT` | `1mb` | Parsed JSON bodies |
| `MAX_FILE_SIZE` | `52428800` | A single upload |
| `MAX_WS_PAYLOAD` | `1048576` | A single WebSocket frame |
| `TRUST_PROXY` | `false` | Proxy hops in front of the backend |

Set a max to `0` to turn that limiter off. Over the limit, REST returns `429`
with `Retry-After` and `RateLimit-*` headers, and the WebSocket replies with an
error frame carrying `code: "RATE_LIMITED"`.

Generation shares one set of buckets across REST and WebSocket, so the limit
cannot be sidestepped by switching transport.

`TRUST_PROXY` matters when the backend sits behind a reverse proxy: without it
every request looks like it comes from the proxy and shares a single bucket.
The bundled compose stack sets `TRUST_PROXY=1`, since the website's nginx is
the only hop.

## Inference bridge

The backend talks to the Python model through a lightweight worker
([model/serve.py](model/serve.py)). When `MODEL_ENABLED=true` and `MODEL_PATH`
points to an exported checkpoint, the backend spawns the worker and runs real
inference. When no checkpoint is present or the worker is unavailable, it falls
back to placeholder responses, so the stack always runs.

Configure it in `backend/.env`:

```bash
MODEL_ENABLED=true
MODEL_PATH=../checkpoints/export/framerai_model.pt
TOKENIZER_PATH=../checkpoints/export/tokenizer
PYTHON_BIN=python3
```

## Cognition layer

A checkpoint answers every prompt from its weights and the current context, and
nothing else. `model/cognition/` is an optional layer that gives it a history:

| Part | What it does |
| --- | --- |
| **Episodic memory** | Stores each experience with its embedding, salience, and the affect at the time. Traces decay with disuse, are strengthened by recall, and the weakest are evicted first. |
| **Semantic memory** | Repeated episodes collapse into concepts - a centroid, a visit count, an exemplar - so generalities accrete without being trained in. |
| **Curiosity** | Novelty from random network distillation (so the familiar stops being interesting) blended with learning progress (so it chases what it is *getting better at*, not what is merely unpredictable). |
| **Affect** | A five-dimensional homeostat - valence, arousal, confidence, curiosity, fatigue - that decays toward setpoints, responds to appraisal, and modulates temperature, top-p, and top-k. |
| **Self-model** | Competence and interest per subject, per language, and per sense; goals it raises for itself; a first-person narrative it writes during sleep. |
| **Sleep** | Fatigue accumulates with effort. Past threshold, prioritised replay rehearses episodes, forms concepts, forgets weak traces, and can run a real gradient step through a `train_step` callback. |
| **Live senses** | Camera and microphone stream into the same loop, gated by change so a static scene produces one memory instead of thousands. |

```python
from model.cognition import Mind
from model.generate import FramerGenerator

gen = FramerGenerator.from_checkpoint("model.pt", "tokenizer.json")
mind = Mind.from_generator(gen)

reply, trace = mind.converse("what is a rectified flow?")
print(trace.feeling, trace.novelty, trace.recalled)

print(mind.wonder())        # a question it asked itself
mind.rest()                 # consolidate now instead of waiting for fatigue
mind.save("mind.pt")        # continuity across restarts
```

Live camera and microphone (needs `pip install opencv-python sounddevice`):

```python
from model.cognition import CameraSource, LiveSession, MicrophoneSource

session = LiveSession(mind, [CameraSource(0), MicrophoneSource()], fps=2, describe=True)
session.run(seconds=30)
print(session.summary())    # polls, attended, attention_rate
```

Serving it: start the worker with `--mind PATH` (or `MIND_PATH`) and chat runs
through the mind, returning its trace alongside the reply, plus the `see`,
`hear`, `watch`, `live`, `wonder`, `reflect`, `feedback`, and `introspect` ops.
Without the flag the worker behaves exactly as before.

This is not a claim that FramerAI is conscious, and the code says so where it
matters. These are functional analogues - memory, intrinsic motivation, affect
that changes behaviour, offline consolidation - each observable in the trace and
tested in isolation. [GUIDE.md](GUIDE.md#cognition-layer) covers the details.

## build.py usage

```bash
# List every preset with its parameter budget (instantiates nothing)
python build.py --list-presets

# Full parameter and memory breakdown for one preset, then exit
python build.py --preset framer-1t-a32b --estimate

# Build model only (initialize weights + tokenizer)
python build.py --mode build

# Train from local data
python build.py --mode train --data-dir data

# Export for serving
python build.py --mode export

# Evaluate a trained checkpoint on standard benchmarks
python build.py --mode eval --benchmark-dir benchmarks

# Full pipeline: build, train, export
python build.py --mode all --size small

# Custom configuration
python build.py --mode all \
  --d-model 512 --n-layers 12 --n-heads 8 \
  --max-steps 5000 --batch-size 16 --lr 1e-4 \
  --data-dir data --device cuda

# Extended context via RoPE scaling
python build.py --mode all --size small --rope-scaling 8.0
```

## Project structure

```txt
framerai/
├── model/                      # Model architecture & training
│   ├── modules/
│   │   ├── transformer.py      # Core transformer (RoPE, SwiGLU, RMSNorm)
│   │   ├── vision_encoder.py   # ViT image encoder
│   │   ├── audio_encoder.py    # Mel-spectrogram audio encoder
│   │   ├── diffusion.py        # U-Net diffusion for images
│   │   ├── video_generator.py  # Temporal diffusion for video
│   │   ├── audio_generator.py  # Mel diffusion for audio
│   │   └── multimodal_projector.py
│   ├── tokenizer/              # BPE tokenizer with special tokens
│   ├── configs/                # Model configuration
│   ├── cognition/              # Optional mind: memory, curiosity, affect, sleep
│   │   ├── mind.py             # The tick loop tying it together
│   │   ├── memory.py           # Episodic, semantic, and working memory
│   │   ├── curiosity.py        # Novelty (RND) + learning progress
│   │   ├── affect.py           # Affective homeostat, modulates decoding
│   │   ├── perception.py       # Live camera/microphone streams + attention gate
│   │   ├── language.py         # Script and language identification
│   │   ├── self_model.py       # Competence, interests, goals, narrative
│   │   └── consolidation.py    # Sleep: replay, concept formation, forgetting
│   ├── data.py                 # Local-corpus datasets
│   ├── framer.py               # Unified FramerModel
│   ├── generate.py             # Inference & generation utilities
│   └── serve.py                # Inference worker for the backend
├── backend/                    # Express API server
│   ├── src/
│   │   ├── routes/             # REST API endpoints
│   │   ├── middleware/         # Validation, errors, rate limiting
│   │   ├── services/           # Model bridge & WebSocket
│   │   ├── app.js              # App and WebSocket wiring
│   │   └── index.js            # Server entry point
│   └── tests/                  # Route and WebSocket tests
├── website/                    # React frontend
│   └── src/
├── data/                       # Local training data (yours)
├── build.py                    # Model builder & trainer
├── Dockerfile                  # Model image
└── docker-compose.yml          # Full stack
```

## Tests

CI runs all three suites on every change, and they all run on CPU in seconds.

```bash
# Model: backbone, MoE routing, multimodal towers, tokenizer, RoPE, data
# pipeline, generation, presets, estimator, cognition layer
pip install -r requirements.txt -r requirements-dev.txt
python -m pytest -q

# Backend: routes and WebSocket
cd backend && npm test

# Website: components (Vitest + Testing Library)
cd website && npm test
```

The Python tests build tiny models on CPU, so no GPU or trained checkpoint is
needed; the preset and estimator tests cover the trillion-parameter flagship
without instantiating it. The backend tests use the built-in Node test runner
with supertest and stub the model service, so no Python worker is needed.

## Documentation

- [Guide](GUIDE.md) - full walkthrough of the model, backend, and website.
- [Contributing](CONTRIBUTING.md) - development setup, branching model, and pull request flow.
- [Roadmap and TODOs](TODOS.md) - planned work and open tasks.
- [Changelog](CHANGELOG.md) - notable changes per release.
- [Code of Conduct](CODE_OF_CONDUCT.md) - community expectations.
- [Security Policy](SECURITY.md) - how to report vulnerabilities.

## Contributing

Contributions are welcome. Read the [contributing guide](CONTRIBUTING.md) and the
[code of conduct](CODE_OF_CONDUCT.md) before opening a pull request.

For questions, proposals, and project showcases, use
[Discussions](https://github.com/Spyxpo/framerai/discussions). Issues are reserved for
defects and specified work.

## License

Open source under the [MIT License](LICENSE).

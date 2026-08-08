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
- Image decoder: a KL-VAE plus a diffusion transformer on a rectified-flow objective, or the
  original pixel-space U-Net, selected by `image_gen_arch`.
- Video decoder: a causal 3D VAE plus a spacetime diffusion transformer, or the original 3D
  U-Net, selected by `video_gen_arch`.
- Audio decoder: an RVQ codec producing discrete acoustic tokens plus a neural vocoder, or the
  original mel diffusion, selected by `audio_gen_arch`.
- Multimodal projector: aligns vision and audio embeddings with the language model space.

Model sizes are named presets in `model/configs/presets.py` (run
`python build.py --list-presets` to print them). Text is the transformer backbone,
multimodal is the encoders plus the diffusion decoders, and model total is the whole thing:

| Preset | d_model | Layers | Heads (Q/KV) | Text | Active | Multimodal | Model total |
|--------|---------|--------|--------------|------|--------|------------|-------------|
| `framer-tiny` | 256 | 6 | 8 / 4 | ~19M | ~19M | ~20M | ~39M |
| `framer-small` | 768 | 12 | 12 / 4 | ~142M | ~142M | ~94M | ~236M |
| `framer-medium` | 1024 | 24 | 16 / 8 | ~429M | ~429M | ~512M | ~941M |
| `framer-large` | 2048 | 24 | 16 / 8 | ~1.2B | ~1.2B | ~2.5B | ~3.7B |
| `framer-8b` | 4096 | 32 | 32 / 8 | ~7.2B | ~7.2B | ~596M | ~7.8B |
| `framer-30b-a3b` | 2048 | 28 | 16 / 4 | ~34B | ~3.0B | ~536M | ~34B |
| `framer-160b-a16b` | 4096 | 48 | 32 / 8 | ~152B | ~15B | ~5.4B | ~157B |
| `framer-200b-a20b` | 5120 | 48 | 40 / 8 | ~193B | ~20B | ~11.5B | ~205B |
| `framer-1t-a32b` | 8192 | 64 | 64 / 8 | ~983B | ~32B | ~21.1B | ~1.00T |
| `framer-2t-a49b` | 10240 | 84 | 80 / 8 | ~1.96T | ~49B | ~39.3B | ~2.00T |

Dense presets train on a single consumer GPU or CPU; the MoE presets (`*-a*b`)
scale total parameters via sparse experts and need multi-node hardware to train.
The four largest scale their vision, audio, and diffusion towers with the backbone, so
`framer-2t-a49b` holds two trillion parameters across all five modalities rather than in the
language model alone. `--size tiny|small|medium|large` remains as a legacy alias, and
`--preset NAME` selects any preset by name (also available on `train.sh`).

### Why 384 fine-grained experts

The flagship uses 384 experts of width 2048 rather than, say, 192 of width 4096. Total
parameters scale with the expert *count*, active parameters with `n_experts_per_tok`, so
narrower and more numerous experts buy sparsity at the same total. 384 is 3 x 128, which
divides evenly across 8, 16, 32, 64, and 128-way expert-parallel meshes - a placement
constraint worth respecting before the sharding code exists rather than after.

Attention is roughly 39% of the flagship's active budget (84 layers x 230.7M). That is a
consequence of `d_model=10240` with only four routed experts per token, and it is the knob to
turn if more of the active budget should sit in the FFN.

### How the estimator sizes a trillion parameters on a laptop

`estimate_params(config)` in `model/utils/helpers.py` never builds the model. The text
backbone is computed analytically from the config (attention, SwiGLU or MoE experts,
embeddings, norms), and each multimodal tower is constructed under `torch.device("meta")`,
which materializes shapes without storage. That keeps the number honest — it tracks the real
module code rather than a formula that can drift from it — while staying instant and
allocation-free. `python build.py --preset NAME --estimate` prints the per-tower breakdown
plus the bf16 weight footprint and the full AdamW training state.

The tower list itself lives in one place, `MULTIMODAL_TOWERS` in `model/utils/helpers.py`,
which the estimator, `build.py`'s breakdown, and the tests all import. `estimate_params` and
`estimate_multimodal_params` take `strict=True` to re-raise instead of degrading to
"could not be sized"; the tests use it so a tower that resists meta construction fails loudly
rather than silently shrinking the reported model.

### Choosing an architecture per modality

Each decoder family is selected by a config field that defaults to the original implementation,
so the laptop-scale presets keep their existing path byte for byte and only the large presets
opt in. The estimator meta-constructs whatever the config selects, so `--estimate` stays exact
across the switch.

| Field | Default | Alternative | Set by |
|---|---|---|---|
| `image_gen_arch` | `unet` | `latent_dit` | the four large MoE presets |
| `video_gen_arch` | `unet3d` | `spacetime_dit` | the four large MoE presets |
| `audio_gen_arch` | `mel_diffusion` | `rvq_lm` | the four large MoE presets |
| `vocoder_arch` | `griffin_lim` | `istft` | the four large MoE presets |
| `mm_token_placement` | `prefix` | `interleaved` | `framer-1t-a32b`, `framer-2t-a49b` |
| `vision_tiling` | `False` | `True` | `framer-1t-a32b`, `framer-2t-a49b` |

`unet` is the pixel-space U-Net. Its attention is quadratic in pixel count, which is why it
cannot run at the resolutions the large presets configure. `latent_dit` is a KL-VAE that
compresses 8x in each spatial dimension plus a diffusion transformer over the resulting latent
grid, so a 512x512 image is denoised as a 64x64 latent - 64x fewer positions to attend over.

The latent path brings three things the U-Net did not have:

- **adaLN-zero conditioning.** Timestep and pooled text modulate every block through a
  zero-initialized projection, so each block starts as the identity and the residual stream is
  undisturbed at step zero. This is what makes a deep denoiser trainable from scratch.
- **Rectified flow.** The model predicts a straight-line velocity between noise and data
  rather than the noise itself, which makes the sampling trajectory nearly straight and
  solvable in 20-50 ODE steps instead of a 1000-step ancestral chain.
- **Classifier-free guidance.** `cfg_dropout_prob` replaces whole examples' conditioning with a
  learned `null_context` embedding during training; at inference the conditional and
  unconditional fields are evaluated in one batch-doubled forward and extrapolated apart by
  `cfg_scale`. The README advertised this from the beginning and no such code existed until now.

Positions in the transformer come from an on-the-fly 2D sin-cos grid rather than a learned
table, so one set of weights denoises any resolution and aspect ratio the VAE can produce.

### Placing modalities in the sequence

Under `prefix`, encoded image and audio embeddings are concatenated ahead of the tokens. That
costs extra sequence positions and, more importantly, discards *where* each modality appeared:
"how does the first image differ from the second" is unanswerable from a prefix.

Under `interleaved`, the sequence reserves one placeholder token per embedding and
`_scatter_modality_embeds` writes them in with `masked_scatter`. Length, attention mask, and
label alignment are all unchanged, because this replaces rather than inserts.

The two halves of the contract are `InterleavedSequenceBuilder` in `model/data.py`, which emits
`<img> <img_patch> x N <img_end>` runs, and the encoder, which produces N embeddings. A count
mismatch raises rather than corrupting the sequence silently.

### High-resolution tiling

With `vision_tiling` on, `image_size` becomes the *tile* size and `DynamicTiler` splits an image
into the `(rows, cols)` grid whose aspect ratio best matches it, bounded by `vision_max_tiles`.
Attention is quadratic within a tile but only linear across tiles, so 12 tiles cover roughly
3.5x the linear resolution at a fraction of what one large forward would cost. A downscaled
thumbnail is prepended because tiles alone destroy global layout.

`PatchEmbedding.interpolate_pos_encoding` bicubically resamples the learned position table when
the patch grid differs from the trained one, so the same weights accept a tile of any shape.

`ContrastiveVisionTrainer` in `model/training/contrastive.py` gives the vision tower a training
signal of its own: symmetric InfoNCE between pooled image and text embeddings with a learned
temperature. Without it the encoder learns only from whatever gradient reaches it through the
language-model loss, which is weak and indirect.

### Requesting a size

`resolve_image_request` in `model/utils/image_request.py` turns a request into concrete
dimensions. Precedence is explicit over implicit, always:

| Source | Wins over | Example |
|---|---|---|
| `width` + `height` | everything | `{"width": 1024, "height": 768}` |
| `aspect` + `tier` | prompt and default | `{"aspect": "16:9", "tier": 1024}` |
| prompt intent | default | `"a widescreen shot of a valley"` |
| config default | - | `image_default_aspect`, `image_size_tier` |

Because an explicit parameter short-circuits parsing entirely, a sizing word in the prompt can
never quietly override what the caller asked for. The resolved `ImageRequest` carries `source`,
so a caller can distinguish an instruction from a guess, and `snapped`, so an off-bucket
request is visibly adjusted rather than silently changed.

Prompt parsing recognises explicit sizes (`1024x1024`, `1024 by 768`), ratios (`16:9`), named
shapes (`widescreen`, `portrait`, `phone wallpaper`, `ultrawide`, `banner`), and tier words
(`hd`, `4k`). Set `image_allow_custom_size=False` to turn it off and honour only explicit
parameters.

A recognised phrase is stripped from the conditioning text only when it reads as an
instruction. `_SUBJECT_WORDS` lists the shape words that are just as likely to be describing
the subject: "a portrait of a woman" still resolves to 2:3, because the guess is a good one,
but the prompt is left intact, since removing the word would leave "a of a woman". "a woman in
portrait orientation" is a directive and is stripped.

Buckets are generated, not listed: for tier area `A` and ratio `r`, the size is
`round16(sqrt(A * r))` by `round16(sqrt(A / r))`. Every ratio therefore costs about the same
compute, and both dimensions divide `vae_downsample * dit_patch_size`, which is what the DiT
patch embedding requires.

Arbitrary ratios need `image_gen_arch="latent_dit"`. Under `unet` a non-square request raises a
message naming the setting, rather than producing something misshapen.

### Constructing a model too large for one host

`build.py` allocates the whole model on one device, which works up to a few billion
parameters and not beyond. `framer-2t-a49b` needs roughly 7.4 TiB in fp32 to instantiate, so
`--mode build` refuses it up front and points at `--estimate` rather than being OOM-killed.
Pass `--force` to override.

To actually materialise a model at that scale, build its shapes first and fill them in
per-rank afterwards:

```python
model = FramerModel.from_config_meta(config)   # shapes only, allocates nothing
# ... apply FSDP / expert-parallel sharding to the meta module ...
model.to_empty(device="cuda")                  # storage, no contents
model.init_weights_(buffer_device="cuda")      # parameters and derived buffers
```

`to_empty()` allocates without initialising, so `init_weights_` has to fill everything:
parameters get their distribution, and every module owning a *derived* buffer (RoPE inverse
frequencies, diffusion noise schedules, mel filterbanks) recomputes it through
`reset_buffers`. Modules with hand-rolled parameters that torch cannot initialise generically
(`RMSNorm`, the patch and frame position embeddings) implement `reset_parameters`.

Checkpoints go through `model/training/checkpoint.py`. `save_sharded` writes one file per
rank into a directory and `load_sharded` reassembles them, resharding if the world size
changed. `gather_full_state_dict` collects the complete tensors on rank 0 for a portable
single-file export, which is only possible when the model fits on one host. Under FSDP2 each
rank holds a slice of every parameter, so the old single-file save was persisting one rank's
shard under a name that claimed to be the whole model.

`FramerConfig.validate()` runs whenever a preset is built and after CLI overrides are applied.
It checks the invariants the modules assume: head divisibility for attention and GQA, patch
size dividing image size, MoE routing width, and the `GroupNorm(32)` channel granularity that
the image and video U-Nets require (which makes 64 the real step for `diffusion_channels`,
since the video U-Net runs at half width).

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

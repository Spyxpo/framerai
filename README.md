# FramerAI

[![CI](https://github.com/Spyxpo/framerai/actions/workflows/ci.yml/badge.svg)](https://github.com/Spyxpo/framerai/actions/workflows/ci.yml)
[![CodeQL](https://github.com/Spyxpo/framerai/actions/workflows/codeql.yml/badge.svg)](https://github.com/Spyxpo/framerai/actions/workflows/codeql.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![PRs Welcome](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](CONTRIBUTING.md)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](requirements.txt)
[![Node 18+](https://img.shields.io/badge/node-18%2B-green.svg)](backend/package.json)

A multimodal open-source AI model that can understand and generate text, code, images, and videos. Built with a transformer backbone, diffusion-based image generation, and temporal video synthesis.

## Documentation

- [Guide](GUIDE.md) - full walkthrough of the model, backend, and website.
- [Contributing](CONTRIBUTING.md) - development setup, branching model, and pull request flow.
- [Roadmap and TODOs](TODOS.md) - planned work and open tasks.
- [Changelog](CHANGELOG.md) - notable changes per release.
- [Code of Conduct](CODE_OF_CONDUCT.md) - community expectations.
- [Security Policy](SECURITY.md) - how to report vulnerabilities.

## Architecture

FramerAI combines multiple neural architectures into a unified model:

- **Transformer Backbone** - Autoregressive decoder with RoPE, SwiGLU, and RMSNorm for text and code generation
- **Vision Encoder** - ViT-style encoder for image understanding with patch embeddings
- **Diffusion Module** - U-Net with cross-attention for text-conditioned image generation
- **Video Generator** - Spatial-temporal diffusion with 3D U-Net for video synthesis
- **Multimodal Projector** - Aligns vision embeddings with the language model space

## Project Structure

```txt
framerai/
├── model/                    # Model architecture & training
│   ├── modules/
│   │   ├── transformer.py    # Core transformer (RoPE, SwiGLU, RMSNorm)
│   │   ├── vision_encoder.py # ViT image encoder
│   │   ├── diffusion.py      # U-Net diffusion for images
│   │   ├── video_generator.py# Temporal diffusion for video
│   │   └── multimodal_projector.py
│   ├── distillation/         # Knowledge distillation pipeline
│   │   ├── providers.py      # Multi-LLM provider client
│   │   ├── data_generator.py # Training data generation
│   │   └── distill_trainer.py# Distillation training loop
│   ├── tokenizer/            # BPE tokenizer with special tokens
│   ├── configs/              # Model configurations
│   ├── framer.py             # Unified FramerModel
│   └── generate.py           # Inference & generation utilities
├── backend/                  # Express.js API server
│   └── src/
│       ├── routes/           # REST API endpoints
│       ├── services/         # Model interface & WebSocket
│       └── index.js          # Server entry point
├── website/                  # React frontend
│   └── src/
│       ├── components/       # Chat, Sidebar, CodeBlock
│       ├── hooks/            # useChat hook
│       ├── services/         # API & WebSocket clients
│       └── styles/           # CSS styles
├── build.py                  # Model builder & trainer
└── requirements.txt          # Python dependencies
```

## Quick Start

### 1. Build the Model

```bash
# Install Python dependencies
pip install -r requirements.txt

# Build a tiny model (for testing)
python build.py --mode build --size tiny

# Build and train
python build.py --mode all --size tiny

# Train a larger model
python build.py --mode all --size small --max-steps 10000
```

### 2. Start the Backend

```bash
cd backend
npm install
cp .env.example .env
npm run dev
```

The API server starts at `http://localhost:3001`.

### 3. Start the Frontend

```bash
cd website
npm install
npm run dev
```

The web app opens at `http://localhost:5173`.

## Model Sizes

| Size   | d_model | Layers | Heads | Parameters |
|--------|---------|--------|-------|------------|
| Tiny   | 256     | 6      | 4     | ~25M       |
| Small  | 512     | 12     | 8     | ~150M      |
| Medium | 1024    | 24     | 16    | ~600M      |
| Large  | 2048    | 32     | 32    | ~2.5B      |

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/health` | Health check & capabilities |
| POST | `/api/chat/conversations` | Create conversation |
| GET | `/api/chat/conversations` | List conversations |
| POST | `/api/chat/conversations/:id/messages` | Send message |
| POST | `/api/generate/image` | Generate image from text |
| POST | `/api/generate/video` | Generate video from text |
| POST | `/api/generate/code` | Generate code |
| POST | `/api/generate/understand` | Analyze uploaded image |
| WS | `/ws` | Real-time streaming |

## build.py Usage

```bash
# Build model only (initialize weights)
python build.py --mode build

# Train from scratch
python build.py --mode train

# Export for serving
python build.py --mode export

# Full pipeline
python build.py --mode all

# Custom configuration
python build.py --mode all \
  --d-model 512 \
  --n-layers 12 \
  --n-heads 8 \
  --max-steps 5000 \
  --batch-size 16 \
  --lr 1e-4 \
  --device cuda
```

## Features

- **Text Generation** - Chat and Q&A with autoregressive decoding (top-k, top-p, temperature)
- **Code Generation** - Write code with lower-temperature sampling for deterministic output
- **Image Generation** - Text-to-image via classifier-free diffusion
- **Video Generation** - Text-to-video with spatial-temporal attention
- **Image Understanding** - Vision encoder processes uploaded images for multimodal chat
- **Streaming** - WebSocket-based real-time token streaming
- **BPE Tokenizer** - Byte-level tokenizer with special multimodal tokens
- **Knowledge Distillation** - Train using local open-source LLMs (Llama 3, DeepSeek R1, Mistral, etc.)

## Knowledge Distillation

Train FramerAI using open-source teacher models downloaded from HuggingFace. Everything runs locally — no API keys, no cloud calls.

### Supported Teacher Models

| Shorthand | HuggingFace Model | Parameters |
|-----------|-------------------|------------|
| `llama3-8b` | meta-llama/Meta-Llama-3-8B-Instruct | 8B |
| `llama3.1-70b` | meta-llama/Llama-3.1-70B-Instruct | 70B |
| `deepseek-r1-7b` | deepseek-ai/DeepSeek-R1-Distill-Qwen-7B | 7B |
| `deepseek-r1-70b` | deepseek-ai/DeepSeek-R1-Distill-Llama-70B | 70B |
| `deepseek-coder-33b` | deepseek-ai/deepseek-coder-33b-instruct | 33B |
| `mistral-7b` | mistralai/Mistral-7B-Instruct-v0.3 | 7B |
| `mixtral-8x7b` | mistralai/Mixtral-8x7B-Instruct-v0.1 | 46.7B |
| `qwen2.5-72b` | Qwen/Qwen2.5-72B-Instruct | 72B |
| `phi-4-mini` | microsoft/Phi-4-mini-instruct | 3.8B |
| `gemma2-27b` | google/gemma-2-27b-it | 27B |
| `llava-1.5-13b` | llava-hf/llava-1.5-13b-hf | 13B |

You can also pass any HuggingFace model path directly (e.g., `--teacher-model your-org/your-model`).

### Multi-Teacher Training

Use `--teacher-model all` to train with **every supported model**. Models are loaded one at a time — only one model's VRAM is needed at once. Each model generates data or distills for its share of steps, then gets unloaded before the next one loads.

| `--teacher-model` value | What it does |
|-------------------------|--------------|
| `all` | 9 recommended 7-9B models (fits on 8GB+ GPU with 4bit) |
| `all-small` | Models <=3B only (fits on any GPU) |
| `all-medium` | All models <=14B |
| `all-large` | Models >14B (needs 24GB+ VRAM with 4bit) |
| `llama3-8b,mistral-7b,deepseek-r1-7b` | Custom comma-separated list |
| `llama3-8b` | Single model |

### Step 1: Generate Training Data

```bash
# USE ALL MODELS — each model generates its share, loaded one at a time
python build.py --mode generate-data \
  --teacher-model all \
  --quantize 4bit \
  --num-samples 10000 \
  --conversations

# All small models (fits on any GPU)
python build.py --mode generate-data \
  --teacher-model all-small \
  --quantize 4bit \
  --num-samples 5000

# Custom combination
python build.py --mode generate-data \
  --teacher-model llama3-8b,deepseek-r1-7b,mistral-7b,qwen2.5-7b \
  --quantize 4bit \
  --num-samples 8000 \
  --conversations

# Single model
python build.py --mode generate-data \
  --teacher-model deepseek-r1-7b \
  --quantize 4bit \
  --num-samples 5000

# Code-focused data from code-specialized models
python build.py --mode generate-data \
  --teacher-model deepseek-coder-7b,qwen2.5-coder-7b \
  --quantize 4bit \
  --domains code_generation,reasoning \
  --num-samples 5000

# Gated models (Llama) need a HuggingFace token
export HF_TOKEN=hf_...
python build.py --mode generate-data \
  --teacher-model all \
  --quantize 4bit \
  --num-samples 20000
```

### Step 2: Distill into FramerAI

The distillation step loads teacher models alongside FramerAI and trains the student
using real teacher logits (soft labels). With multiple teachers, each model teaches for
its portion of training steps, then gets swapped. Knowledge accumulates across all teachers.

```bash
# Build base model first
python build.py --mode build --size small

# DISTILL WITH ALL MODELS — each teacher trains for its share of steps
python build.py --mode distill --size small \
  --teacher-model all \
  --quantize 4bit \
  --max-steps 100000

# Distill with specific models
python build.py --mode distill --size small \
  --teacher-model llama3-8b,deepseek-r1-7b,mistral-7b \
  --quantize 4bit \
  --max-steps 60000

# Single teacher with feature alignment
python build.py --mode distill --size small \
  --teacher-model deepseek-r1-7b \
  --quantize 4bit \
  --alpha-feature 0.1 \
  --max-steps 50000

# Extended context (8x = 16K tokens)
python build.py --mode distill --size small \
  --teacher-model all \
  --quantize 4bit \
  --rope-scaling 8.0 \
  --target-seq-len 16384

# FULL PIPELINE — build, generate from all models, distill from all, export
python build.py --mode build --size large
python build.py --mode generate-data --teacher-model all --quantize 4bit --num-samples 20000 --conversations
python build.py --mode distill --size large --teacher-model all --quantize 4bit --max-steps 200000
python build.py --mode export --size large
```

### VRAM Requirements

| Teacher Model | Full Precision | 8-bit | 4-bit |
|--------------|----------------|-------|-------|
| 1-3B (Phi, Gemma-2B) | ~6 GB | ~3 GB | ~2 GB |
| 7-8B (Llama, DeepSeek, Mistral) | ~16 GB | ~8 GB | ~5 GB |
| 13-14B (LLaVA, Qwen) | ~28 GB | ~14 GB | ~8 GB |
| 33B (DeepSeek Coder) | ~66 GB | ~33 GB | ~18 GB |
| 70-72B (Llama, DeepSeek, Qwen) | ~140 GB | ~70 GB | ~38 GB |

Use `--quantize 4bit` to fit larger models on consumer GPUs.
Use `--teacher-device cpu` for CPU-only (slow but no VRAM needed).

### Training Data Domains

- **general_knowledge** - Facts, explanations, comparisons
- **reasoning** - Step-by-step problem solving, logic
- **code_generation** - Functions, algorithms, tests, refactoring
- **instruction_following** - Tutorials, guides, structured output
- **creative** - Stories, poetry, dialogue
- **long_context** - Document analysis, summarization

### Context Extension

RoPE scaling for extended context windows:

| Scaling Factor | Effective Context |
|---------------|-------------------|
| 1.0 (default) | 2,048 tokens |
| 4.0 | 8,192 tokens |
| 8.0 | 16,384 tokens |
| 32.0 | 65,536 tokens |
| 512.0 | 1,048,576 tokens |

```bash
python build.py --mode distill --size large \
  --teacher-model deepseek-r1-70b --quantize 4bit \
  --rope-scaling 512.0 --target-seq-len 1048576 --no-progressive
```

## License

Open source. See [LICENSE](LICENSE) for details.

# FramerAI

A multimodal open-source AI model that can understand and generate text, code, images, and videos. Built with a transformer backbone, diffusion-based image generation, and temporal video synthesis.

## Architecture

FramerAI combines multiple neural architectures into a unified model:

- **Transformer Backbone** - Autoregressive decoder with RoPE, SwiGLU, and RMSNorm for text and code generation
- **Vision Encoder** - ViT-style encoder for image understanding with patch embeddings
- **Diffusion Module** - U-Net with cross-attention for text-conditioned image generation
- **Video Generator** - Spatial-temporal diffusion with 3D U-Net for video synthesis
- **Multimodal Projector** - Aligns vision embeddings with the language model space

## Project Structure

```
framerai/
├── model/                    # Model architecture & training
│   ├── modules/
│   │   ├── transformer.py    # Core transformer (RoPE, SwiGLU, RMSNorm)
│   │   ├── vision_encoder.py # ViT image encoder
│   │   ├── diffusion.py      # U-Net diffusion for images
│   │   ├── video_generator.py# Temporal diffusion for video
│   │   └── multimodal_projector.py
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

## License

Open source. See [LICENSE](LICENSE) for details.

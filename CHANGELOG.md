# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Scale-capable LLM core: **grouped-query attention (GQA)**, fused
  scaled-dot-product (flash / memory-efficient) attention, an incremental **KV
  cache** for O(n) decoding, RoPE **context-extension scaling** (linear / NTK),
  and optional QK-normalization + depth-scaled init.
- **Mixture-of-Experts** FFN (`model/modules/moe.py`): top-k routing, optional
  shared experts, and a load-balancing + router z-loss folded into the model's
  total loss.
- Named **preset registry** (`model/configs/presets.py`) from `framer-tiny`
  (~19M) to `framer-1t-a32b` (~1T total / ~32B active), a `FramerConfig.from_preset`
  constructor, `--preset` / `--list-presets` CLI, and a parameter **estimator**
  (`estimate_params`) that sizes even the 1T preset without instantiating it.
- Training stack (`model/training/`): warmup→cosine LR, precision-aware autocast
  (bf16/fp16 on GPU, fp32 on CPU), activation checkpointing, weight-decay
  parameter groups, and a guarded torch-native **FSDP2** multi-GPU path.
- Streaming, packed token data: `scripts/prepare_data.py` writes memory-mapped
  shards; `PackedTokenDataset` yields padding-free, rank-sharded next-token blocks.
- `--text-only` build scope for training the LLM core without the multimodal
  submodules; optional **safetensors** export; a `tests/` pytest suite.
- Audio modality end to end: an audio encoder for understanding and a
  text-conditioned mel-diffusion generator with Griffin-Lim reconstruction.
- `<audio>` and `<audio_end>` tokenizer special tokens.
- Local-corpus data loader (`model/data.py`) for text, image-caption, and
  audio-caption training data, replacing teacher-generated data.
- Python inference worker (`model/serve.py`) and a backend bridge that runs real
  inference when a checkpoint exists and falls back to placeholders otherwise.
- Backend endpoints `POST /api/generate/audio` and `POST /api/generate/transcribe`,
  and `audio` in the health capabilities.
- Website audio mode, an audio-upload transcription control, and image, video,
  and audio players in the chat.
- Docker build for the model plus a `docker-compose.yml` for the full stack.
- Continuous integration workflows for the model, backend, and website.
- CodeQL security analysis and a scheduled stale issue workflow.
- Dependabot configuration for pip, npm, and GitHub Actions updates.
- Issue and pull request templates, CODEOWNERS, and community health files.
- Contributor guide, code of conduct, security policy, project guide, and roadmap.

### Changed

- FramerAI now trains from scratch on a local corpus. Removed the knowledge
  distillation pipeline and all external teacher-model dependencies
  (`transformers`, `accelerate`, `bitsandbytes`, `sentencepiece`, `huggingface-hub`).
- Added `torchaudio` and `soundfile` for audio I/O.

### Removed

- The `model/distillation/` package and the `generate-data` and `distill` build modes.

## [1.0.0] - 2026-02-16

### Added

- Initial public release of FramerAI.
- Transformer backbone with RoPE, SwiGLU, and RMSNorm for text and code.
- Vision encoder for image understanding.
- Diffusion module for text-to-image generation.
- Spatial-temporal video generator.
- Multimodal projector aligning vision and language spaces.
- Knowledge distillation pipeline using local open-source teacher models.
- Express backend with REST endpoints and WebSocket streaming.
- React web frontend for chat and generation.

[Unreleased]: https://github.com/Spyxpo/framerai/compare/v1.0.0...HEAD
[1.0.0]: https://github.com/Spyxpo/framerai/releases/tag/v1.0.0

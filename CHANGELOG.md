# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

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

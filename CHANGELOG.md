# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **Video generation raised `RuntimeError` on every forward pass.** The 3D U-Net
  sized its timestep embedding from the input channel count instead of the base
  channel count, so `forward_video`, `VideoGenerator.sample`, and the
  `/api/generate/video` route could never produce a frame.
- **`decode()` returned an empty string after the tokenizer was reloaded.**
  `FramerTokenizer.load` rebuilt the merge map but not the merge vocabulary, so
  every merged token silently decoded to nothing on the live inference path.
  Saved tokenizers now carry a version, their own special-token table, and a
  fixed-capacity reserved marker block placed after the byte range, so adding a
  marker no longer shifts byte or merge ids. Version 1 files still load.
- **`rope_scaling_type="yarn"` applied no context extension at all.** It matched
  neither the NTK nor the linear branch and silently behaved like `"none"`. YaRN
  is now implemented as per-dimension NTK-by-parts interpolation with the
  attention-factor compensation, configurable through `rope_low_freq_factor`,
  `rope_high_freq_factor`, and `rope_original_max_seq_len`. `validate()` rejects
  an unrecognised `rope_scaling_type` rather than ignoring it.
- Removed a dead zero-initialisation in the MoE auxiliary-loss path.

### Changed

- The vision, audio, temporal, and diffusion attention paths now use the fused
  `scaled_dot_product_attention` kernel instead of materializing an explicit
  attention matrix. The vision encoder is shared by the audio encoder, so both
  benefit. Parameter counts are unchanged; the memory cost stops being quadratic
  in patch or pixel count, which is what put the large presets' configured
  resolutions out of reach. Attention dropout is now correctly gated on training
  mode, so eval-mode inference is deterministic.

### Added

- Forward-pass tests for every multimodal tower (`tests/test_modality_forward.py`),
  attention parity tests for the SDPA conversion (`tests/test_attention_parity.py`),
  tokenizer save/load round-trip and id-layout tests (`tests/test_tokenizer.py`),
  and RoPE context-extension tests (`tests/test_rope.py`). No test previously
  executed a vision, audio, image-diffusion, or video forward pass.
- **All-modality flagship**: `framer-1t-a32b` now scales its vision and audio
  encoders and its image, video, and audio diffusion decoders with the backbone -
  ~999B parameters of text, code, image, video, and audio in one model, ~32B
  active per text token. New `framer-200b-a20b` (~202B total / ~20B active) sits
  between it and `framer-30b-a3b`.
- **Whole-model parameter estimator**: `estimate_params` reports `multimodal`,
  `model_total`, and the bf16 / AdamW memory footprint alongside the existing
  text-core counts, and `estimate_multimodal_params` counts each tower by building
  it on the meta device (no allocation). QK-norm parameters are now counted.
- `build.py --estimate` prints the per-tower parameter and memory budget for a
  config and exits; `--list-presets` gains multimodal and model-total columns;
  `train.sh --preset NAME` reaches any preset from the wrapper script.
- `FramerConfig.validate()` checks the shape invariants the modules assume (head
  divisibility, patch/image size, MoE routing width, `GroupNorm(32)` channel
  granularity) when a preset is built and after CLI overrides are applied.
- CI runs the Python test suite on CPU wheels plus an estimator smoke check,
  lints and byte-compiles `tests/`, `scripts/`, and `conftest.py`, and caches pip.

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

### Fixed

- CI: the website job no longer runs on Node 18, which its build tooling cannot
  support (Node `^20.19 || >=22.12` is required). The job had been failing on
  every run. Both packages now declare `engines`, and the documented
  requirements match what CI runs.

### Changed

- `framer-160b-a16b` scales its multimodal towers on the same basis as the
  flagship (~156B total across all modalities).
- `framer-tiny-moe` uses the tiny towers its dense sibling uses, so the
  laptop-validation preset stays laptop-sized.
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

# Roadmap and TODOs

This is the working checklist for FramerAI. Items are grouped by area. Larger
items are tracked as GitHub issues; check them off here when the matching issue
is closed. New ideas are welcome through a feature request.

Legend: `[ ]` open, `[x]` done, `[~]` in progress.

## Testing and quality

- [ ] Add a `pytest` suite covering the tokenizer, transformer forward pass, and generation utilities.
- [x] Add unit tests for the backend routes and WebSocket service (node:test with supertest).
- [ ] Add component tests for the website chat flow (Vitest and Testing Library).
- [ ] Add end-to-end smoke tests that boot the backend and exercise the core endpoints.
- [ ] Add code coverage reporting to CI and publish a coverage badge.
- [ ] Introduce pre-commit hooks running ruff and lightweight JS checks.

## Model and training

- [x] Publish reproducible size presets (registry in `model/configs/presets.py`,
      `framer-tiny` … `framer-1t-a32b`) with a parameter estimator.
- [x] Grouped-query attention, fused SDPA/flash attention, and an incremental KV cache.
- [x] Mixture-of-Experts FFN with load-balancing + router z-loss for trillion-scale totals.
- [x] Support gradient checkpointing and mixed precision (bf16/fp16/fp32) flags end to end.
- [x] Warmup→cosine LR schedule wired through `config.warmup_steps`.
- [x] Streaming, packed token-shard data pipeline (`scripts/prepare_data.py`).
- [x] Optional safetensors export.
- [~] Multi-GPU / distributed training via torch-native FSDP2 (guarded; validated single-device).
      Tensor / expert / pipeline parallelism for the 1T preset remain to be built.
- [ ] Add a full state-dict gather for FSDP checkpoint save/load.
- [ ] Add evaluation harness with standard benchmarks and a results table in the docs.
- [ ] Add ONNX export and a safetensors round-trip validation test.

## Phase 2 — generation quality (image / video / audio)

- [ ] Latent diffusion for image generation (VAE + DiT), DALL·E-3-style.
- [ ] Spacetime-latent-patch diffusion transformer for video, Sora-style.
- [ ] Neural vocoder to replace Griffin-Lim (also tracked below).
- [ ] Interleaved multimodal token placement (LLaVA/Qwen-VL-style) beyond prefix concat.

## Multimodal and audio

- [ ] Replace Griffin-Lim with a neural vocoder for higher-fidelity audio output.
- [ ] Add streaming audio generation and playback over the WebSocket.
- [ ] Add in-browser microphone capture (MediaRecorder) for audio input.
- [ ] Add a mel-spectrogram cache to speed up audio training.
- [ ] Add example image and audio caption datasets with real media.
- [ ] Benchmark and tune the audio encoder and generator dimensions per size.

## Backend

- [x] Add request validation and consistent error responses across all routes.
- [x] Add rate limiting and payload size limits to generation endpoints.
- [ ] Add structured logging and a request id for traceability.
- [ ] Add OpenAPI or a documented schema for the REST API.
- [ ] Pool or reuse the inference worker under concurrent load.

## Frontend

- [x] Add loading, empty, and error states to the chat and generation views.
- [x] Add accessibility passes for keyboard navigation and screen readers.
- [x] Add a settings panel for model size, temperature, and sampling controls.
- [ ] Add persistence of conversations to local storage.
- [ ] Add a production build and static hosting guide.

## Documentation

- [ ] Expand GUIDE.md with architecture diagrams for each module.
- [ ] Add a from-scratch training tutorial that walks through a full run on a single GPU.
- [ ] Add a troubleshooting page for common CUDA, VRAM, and dependency issues.
- [ ] Add API reference documentation generated from source.

## DevOps and CI/CD

- [x] Build the model, backend, and website images in CI on every change.
- [ ] Add a release workflow that tags versions and drafts release notes.
- [ ] Add container image publishing to a registry on release.
- [ ] Add caching for Python dependencies to speed up CI.
- [ ] Add a workflow that validates documentation links.

## Community and governance

- [ ] Enable GitHub Discussions and seed categories for questions and ideas.
- [ ] Add a curated list of good first issues for new contributors.
- [ ] Add a maintainers file describing review ownership and release duties.

## Done

- [x] Make FramerAI self-contained: remove teacher-model distillation and its dependencies.
- [x] Add the audio modality (understanding and generation) across model, backend, and website.
- [x] Add a local-corpus data loader for from-scratch training.
- [x] Add a Python inference bridge with placeholder fallback.
- [x] Add a Dockerfile and docker-compose for the model and full stack.
- [x] Add continuous integration for model, backend, and website.
- [x] Add CodeQL analysis and stale issue automation.
- [x] Add Dependabot for pip, npm, and GitHub Actions.
- [x] Add issue and pull request templates and CODEOWNERS.
- [x] Add contributor guide, code of conduct, and security policy.

# Roadmap and TODOs

This is the working checklist for FramerAI. Items are grouped by area. Larger
items are tracked as GitHub issues; check them off here when the matching issue
is closed. New ideas are welcome through a feature request.

Legend: `[ ]` open, `[x]` done, `[~]` in progress.

## Testing and quality

- [x] Add a `pytest` suite covering the backbone, MoE routing, data pipeline, generation,
      presets, and the parameter estimator, and run it in CI.
- [x] Add unit tests for the backend routes and WebSocket service (node:test with supertest).
- [x] Add component tests for the website (Vitest, Testing Library, jsdom) and an ESLint flat
      config, both blocking in CI. The website previously had no lint and no test leg at all.
- [ ] Extend website component tests past the settings panel to the full chat flow.
- [ ] Add end-to-end smoke tests that boot the backend and exercise the core endpoints.
- [x] Add code coverage reporting to CI (`pytest --cov`, uploaded as an artifact).
- [ ] Publish a coverage badge.
- [ ] Introduce pre-commit hooks running ruff and lightweight JS checks.

## Model and training

- [x] Publish reproducible size presets (registry in `model/configs/presets.py`,
      `framer-tiny` … `framer-2t-a49b`) with a parameter estimator.
- [x] Scale every modality on the large presets, so `framer-2t-a49b` is ~2T parameters of
      text, code, image, video, and audio in one model, and report the whole-model number
      (`--estimate`, `--list-presets`) without instantiating anything.
- [x] Validate config shape invariants up front (`FramerConfig.validate()`).
- [x] Benchmark and tune the audio encoder and generator dimensions per size.
- [x] Grouped-query attention, fused SDPA/flash attention, and an incremental KV cache.
- [x] Use fused SDPA in the vision, audio, temporal, and diffusion attention paths, which
      materialised an explicit attention matrix and made memory quadratic in patch/pixel
      count. Compute in the pixel-space image U-Net stays quadratic; only the latent
      diffusion path below makes high resolutions practical.
- [x] Mixture-of-Experts FFN with load-balancing + router z-loss for trillion-scale totals.
- [x] Support gradient checkpointing and mixed precision (bf16/fp16/fp32) flags end to end.
- [x] Warmup→cosine LR schedule wired through `config.warmup_steps`.
- [x] Streaming, packed token-shard data pipeline (`scripts/prepare_data.py`).
- [x] Optional safetensors export.
- [~] Multi-GPU / distributed training via torch-native FSDP2 (guarded; validated single-device).
      Tensor / expert / pipeline parallelism for the 2T preset remain to be built.
- [x] Add a full state-dict gather for FSDP checkpoint save/load (`gather_full_state_dict`).
      The previous path wrote a single rank's shard under a filename claiming to be the
      whole model.
- [x] Add deferred (meta-device) initialisation (`FramerModel.from_config_meta`,
      `init_weights_`, `reset_buffers`), without which `framer-2t-a49b` could not be
      constructed at all.
- [x] Add sharded checkpoint save/load (`model/training/checkpoint.py`).
- [ ] Add expert-parallel sharding, without which the 2T preset's experts cannot be
      distributed across hosts.
- [ ] Train the tokenizer to the full vocabulary (`build.py` currently caps merges at 1000).
- [x] Implement `yarn` RoPE scaling (per-dimension NTK-by-parts interpolation with the
      attention-factor compensation). It was accepted by the config and silently applied no
      extension at all; `validate()` now rejects an unrecognised `rope_scaling_type`.
- [ ] Add instruction and preference post-training (chat template, SFT, DPO).
- [ ] Add ONNX export and a safetensors round-trip validation test.

## Architecture roadmap — reaching frontier-class output

Parameter count is the ceiling, not the quality. The current decoders cannot reach
frontier-class output at any size, for architectural reasons; each item below replaces one
of them with the family that can, selected by a config field so the small presets keep
their laptop-runnable path. Every phase ships tiny-scale tests and the metric that measures
it. Training compute and licensed data remain a separate, external problem.

### Image generation — latent diffusion transformer

- [x] KL-VAE (8x spatial downsample) so training runs in latent space, not pixel space.
- [x] Diffusion transformer denoiser with adaLN-zero timestep and text conditioning.
- [x] Rectified-flow objective with an ODE sampler at 20–50 steps, replacing 1000-step
      ancestral sampling.
- [x] Classifier-free guidance with a learned null-context embedding (advertised in the
      README for a long time and never implemented until now).
- [ ] Train the VAE properly and re-measure `scale_factor`; the default is a starting value.
- [ ] Adversarial / perceptual reconstruction loss for the VAE (MSE alone blurs).
- [ ] Caption enrichment pass over the training corpus; caption quality dominates prompt
      adherence.

### Video generation — spacetime latent diffusion

- [ ] 3D causal video VAE (4x temporal, 8x spatial compression).
- [ ] Spacetime-patch diffusion transformer with factorised spatial/temporal attention,
      variable duration, resolution, and aspect ratio, and frame-rate conditioning.
- [ ] Remove the per-frame Python loop in the 3D U-Net forward pass, the current
      throughput wall.

### Audio — neural codec and vocoder

- [ ] Residual-vector-quantized audio codec at 24 kHz, giving discrete acoustic tokens the
      language model can predict directly.
- [ ] ISTFT-head neural vocoder replacing Griffin-Lim phase reconstruction, which caps
      output quality no matter how well the model is trained.
- [ ] Speaker and prosody conditioning from a reference clip.
- [ ] CTC auxiliary head and aligned `<audio>` token placement, so transcription is trained
      rather than hoped for.
- [ ] Streaming audio generation and playback over the WebSocket.
- [ ] Mel-spectrogram cache to speed up audio training.

### Understanding

- [ ] Dynamic high-resolution tiling in the vision encoder (tiles plus a global thumbnail).
- [ ] Interleaved multimodal token placement, replacing prefix concatenation.
- [ ] Contrastive pretraining entry point for the vision tower.
- [ ] Example image and audio caption datasets with real media.

### Evaluation

- [ ] `model/eval/` harness: text perplexity and task accuracy, image contrastive-alignment
      score and FID, audio mel-distance / speaker similarity / WER, video FVD and temporal
      consistency. This is what turns architecture parity into a measurable claim.

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
- [x] Add in-browser microphone capture (MediaRecorder) for audio input.
- [x] Add a local-corpus data loader for from-scratch training.
- [x] Add a Python inference bridge with placeholder fallback.
- [x] Add a Dockerfile and docker-compose for the model and full stack.
- [x] Add continuous integration for model, backend, and website.
- [x] Add CodeQL analysis and stale issue automation.
- [x] Add Dependabot for pip, npm, and GitHub Actions.
- [x] Add issue and pull request templates and CODEOWNERS.
- [x] Add contributor guide, code of conduct, and security policy.

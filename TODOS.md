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
- [x] Provide ready-to-use, reproducible training configurations for the four main presets
      (`framer-tiny`, `framer-small`, `framer-medium`, `framer-large`): fixed seed (42),
      documented hyperparameters, hardware expectations, and a `--seed` CLI flag. See
      `model/configs/training_configs.py` and the "Reproducible training" section in GUIDE.md.
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
      Expert parallelism is built; tensor and pipeline parallelism remain.
- [x] Add a full state-dict gather for FSDP checkpoint save/load (`gather_full_state_dict`).
      The previous path wrote a single rank's shard under a filename claiming to be the
      whole model.
- [x] Add deferred (meta-device) initialisation (`FramerModel.from_config_meta`,
      `init_weights_`, `reset_buffers`), without which `framer-2t-a49b` could not be
      constructed at all.
- [x] Add sharded checkpoint save/load (`model/training/checkpoint.py`).
- [x] Add expert-parallel sharding (`model/training/expert_parallel.py`), so a MoE layer's
      experts divide across a mesh instead of every rank holding all of them. Composes with
      FSDP through a 2D `(dp, ep)` device mesh, so expert weights are not sharded twice.
- [ ] Exercise expert parallelism on real multi-rank hardware; the placement arithmetic and
      the single-rank path are tested, the collectives are not.
- [x] Replace the per-expert `index_add_` dispatch with grouped GEMM. At 384 experts across
      80 MoE layers that loop runs 30,720 Python-level expert calls per forward.
- [ ] Train the tokenizer to the full vocabulary (`build.py` currently caps merges at 1000).
- [x] Implement `yarn` RoPE scaling (per-dimension NTK-by-parts interpolation with the
      attention-factor compensation). It was accepted by the config and silently applied no
      extension at all; `validate()` now rejects an unrecognised `rope_scaling_type`.
- [x] Add instruction and preference post-training (chat template, SFT, DPO).
- [x] Add a tool protocol and a bounded tool-calling loop (`model/tools/`), with internet
      search and page fetch behind `--tools web`. Standard library only, no API key.
- [x] Add command line access behind `--tools cli`: a sandboxed shell with a three-mode
      permission policy, an always-on deny list, and read-only file helpers.
- [ ] Surface the approver for `--cli-mode ask` over the backend, so a person can approve
      a command from the website instead of the worker refusing for want of one.
- [x] Add ONNX export and a safetensors round-trip validation test.

## Architecture roadmap — reaching frontier-class output

Parameter count is the ceiling, not the quality. The current decoders cannot reach
frontier-class output at any size, for architectural reasons; each item below replaces one
of them with the family that can, selected by a config field so the small presets keep
their laptop-runnable path. Every phase ships tiny-scale tests and the metric that measures
it. Training compute and licensed data remain a separate, external problem.

### Scale and reasoning — the 3T flagship

- [x] Add a `framer-3t-a64b` preset that scales every modality, not just the backbone:
      512 fine-grained experts at top-6, and vision, audio, image, video and audio-LM towers
      roughly doubled. ~2.93T text parameters, ~63.98B active per token, ~76.73B multimodal,
      ~3.01T total. Cluster-only, like `framer-2t-a49b`.
- [x] Extend the trillion-scale presets to a 1,048,576-token context with yarn RoPE
      scaling, validated against the scaling factor, with chunked prefill so the window is
      reachable and a KV-cache figure in `--estimate` so its cost is visible.
- [ ] Add a reasoning segment to the chat template, with its own special tokens, so a
      thinking span is part of the trained format rather than prose the model happens to emit.
- [ ] Add a reasoning budget and a reasoning-effort control (off / low / medium / high)
      through `build.py`, `model/serve.py`, the backend routes and the website settings panel.
- [ ] Add self-consistency and a verification pass as opt-in test-time compute strategies.
- [ ] Report every quality claim through `model/eval/`, as a before-and-after table on the
      same checkpoint.
- [x] Make the declared window reachable. Prompts are bounded against `max_seq_len`
      rather than extrapolating RoPE, the API derives its limits from the window
      the worker reports instead of fixed constants, and conversation history
      reaches the model through the `messages` parameter that had no caller.
- [x] Page and optionally quantise the KV cache. The cache was concatenated per
      decoded token and stored densely in the activation dtype; on
      `framer-1t-a32b` a million-token sequence is 256 GiB in bf16 and 130 GiB in
      int8, and `--estimate` now reports whichever the config selects.
- [ ] Long-context training curriculum: staged packing from
      `rope_original_max_seq_len` up to `max_seq_len`, so the extension is trained
      rather than only configured and measured.
- [ ] Assemble repository-scale inputs by relevance, so a large project is packed
      rather than truncated at an arbitrary cut.

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
- [x] Few-step sampling (`FlowDistiller`): the student learns the guided field, so
      a step is one denoiser forward instead of two and the count drops to single
      digits. Four steps against fifty is twenty-five times fewer calls.
- [ ] Train a distilled student and report it against the teacher on the same
      checkpoint. The objective and the cost accounting are in; nothing has been
      distilled yet.
- [ ] Move `framer-3b`, `framer-8b` and `framer-30b-a3b` off the 1000-step pixel
      U-Net. The README justifies `unet` for laptop presets, which does not
      describe a 30B mixture-of-experts. Changing them shifts their reported
      multimodal parameter counts, so it belongs with the preset tables.
- [ ] Legible text inside generated images, measured by rendering and reading it
      back through the dense text suite.

### Video generation — spacetime latent diffusion

- [x] 3D causal video VAE (4x temporal, 8x spatial compression).
- [x] Spacetime-patch diffusion transformer with factorised spatial/temporal attention,
      variable duration, resolution, and aspect ratio, and frame-rate conditioning.
- [x] Remove the per-frame Python loop from the video forward pass. The spacetime
      transformer's two attention passes are batched reshapes; the 3D U-Net remains
      available under `video_gen_arch="unet3d"` and still has the loop.
- [ ] Train the video VAE and re-measure its `scale_factor`.
- [x] Streaming decode, which the causal VAE made possible and nothing exercised.
      `sample_long` overlaps the denoising windows and holds each one's opening
      latent frames to the closing frames of the one before, so duration is
      bounded by memory over time rather than by one window.
- [x] Write a real container at the requested frame rate. The writer hardcoded
      100 ms a frame, so every clip came back at 10 fps whatever the decoder had
      been conditioned on, quantised to a 256-colour palette.
- [ ] Report FVD and temporal consistency at increasing durations, so the length
      at which coherence breaks is measured rather than assumed.

### Audio — neural codec and vocoder

- [x] Residual-vector-quantized audio codec at 24 kHz, giving discrete acoustic tokens the
      language model can predict directly.
- [x] ISTFT-head neural vocoder replacing Griffin-Lim phase reconstruction, which caps
      output quality no matter how well the model is trained.
- [x] Speaker conditioning from a reference clip (`use_speaker_conditioning`).
- [x] CTC auxiliary head, so transcription can be trained rather than hoped for. Aligned
      `<audio>` token placement is part of the interleaved-token work below.
- [ ] Wire the CTC head into the training loop; the module exists but nothing calls it yet.
- [ ] Adversarial and multi-scale spectral losses for the codec (MSE alone under-trains it).
- [ ] Prosody conditioning beyond speaker identity.
- [ ] Streaming audio generation and playback over the WebSocket.
- [ ] Mel-spectrogram cache to speed up audio training.

### Understanding

- [x] Dynamic high-resolution tiling in the vision encoder (tiles plus a global thumbnail),
      with bicubic position-embedding resampling so one encoder accepts any tile shape.
- [x] Interleaved multimodal token placement, replacing prefix concatenation
      (`mm_token_placement="interleaved"`, `InterleavedSequenceBuilder`).
- [x] Contrastive pretraining entry point for the vision tower
      (`model/training/contrastive.py`).
- [ ] Wire the contrastive trainer into `build.py` as a `--mode pretrain-vision` entry point.
- [x] Extend interleaved placement to the generation path, not just understanding.
      `<img>` and `<audio>` markers in a prompt say where a modality belongs, and
      chunked prefill walks the embeddings alongside the chunks so the scatter
      still gets exactly as many as the window holds placeholders.
- [x] Use the tiler on the inference path. `encode_image_tiles` was reached only
      from training, contrastive pretraining and eval; a served request ran the
      plain encoder on an image squashed to a fixed square, so a page was seen at
      one tile's resolution however many tiles were configured.
- [x] Add document ingestion (`model/document.py`): PDF text layers in reading
      order, page markers, scanned-page detection, and a pluggable optional
      rasteriser. Wired to `model/data.py`, a `document` worker op, and the
      upload routes.
- [x] Deliver chat attachments to the model. The array was validated, stored and
      dropped at `processMessage`, the website always sent an empty one, and the
      only file input took audio.
- [ ] Train dense text recognition. `model/eval/dense_text.py` reports the
      character and word error rate per font size; nothing yet moves it.
- [ ] Extract tables, figures and charts from document pages as structure rather
      than as a flat reading of the text.
- [ ] Example image and audio caption datasets with real media.

### Evaluation

- [x] `model/eval/` harness: text perplexity, token accuracy, and bits per byte; image
      Frechet distance and contrastive alignment; audio SI-SDR, mel distance, WER, CER, and
      speaker similarity; video FVD and temporal consistency. Suites that cannot run report
      why rather than reporting nothing.
- [x] Standard benchmark adapters so the numbers can be compared outside this repository.
- [x] A `build.py --mode eval` entry point wired to the harness.
- [x] Long-context retrieval suite (`model/eval/longcontext.py`): single-fact
      retrieval swept by depth, multi-hop, and window-wide aggregation, each a
      forced choice against a known chance rate, reported per length bucket.
      Perplexity stays low on long text while retrieval fails, so it could never
      have answered this.
- [x] Dense text recognition suite (`model/eval/dense_text.py`), the image-side
      counterpart to the character error rate the audio suites already had.

## Cognition layer

- [x] Add a persistent cognitive layer (`model/cognition/`): episodic and semantic memory
      with decay, recall-strengthening, and weakest-first eviction; curiosity from RND
      novelty plus per-topic learning progress; a five-dimensional affective homeostat that
      modulates decoding; a self-model; and sleep-time replay, consolidation, and forgetting.
- [x] Stream live camera and microphone into the same loop, gated by change so a static
      scene does not flood memory, with video remembered as one event rather than N stills.
- [x] Identify script and language for every experience, track competence per language and
      per sense, and ask for the reply in the language the input was in.
- [x] Expose it from the inference worker behind `--mind PATH`, leaving the default path
      byte-identical.
- [ ] Wire the trace (recalled memories, affect, sampling) into the backend and the website,
      so a user can see why an answer came out the way it did.
- [ ] Implement a real `train_step` for sleep: LoRA or a small-LR update over replayed
      episodes, with a guard against catastrophic forgetting, and measure it.
- [ ] Replace the fixed random projection in the experience encoder with a learned one
      trained on retrieval quality, once there is a checkpoint worth measuring against.
- [ ] Benchmark retrieval on a held-out episode set (recall@k against a known-relevant set)
      instead of relying on unit tests of the scoring formula.
- [ ] Extend language profiles beyond the current set, and report per-language competence
      against a real multilingual eval rather than the mind's own error estimate.

## Backend

- [x] Add request validation and consistent error responses across all routes.
- [x] Add rate limiting and payload size limits to generation endpoints.
- [ ] Add structured logging and a request id for traceability.
- [x] Derive input limits from the loaded model rather than from fixed constants,
      and share the conversation store between the REST and WebSocket paths.
- [x] Add OpenAPI or a documented schema for the REST API.
- [x] Pool or reuse the inference worker under concurrent load.

## Frontend

- [x] Add loading, empty, and error states to the chat and generation views.
- [x] Add accessibility passes for keyboard navigation and screen readers.
- [x] Add a settings panel for model size, temperature, and sampling controls.
- [ ] Add persistence of conversations to local storage.
- [ ] Add a production build and static hosting guide.

## Documentation

- [ ] Expand GUIDE.md with architecture diagrams for each module.
- [x] Add a from-scratch training tutorial that walks through a full run on a single GPU.
- [x] Add a troubleshooting page for common CUDA, VRAM, and dependency issues.
- [ ] Add API reference documentation generated from source.

## DevOps and CI/CD

- [x] Build the model, backend, and website images in CI on every change.
- [ ] Add a release workflow that tags versions and drafts release notes.
- [ ] Add container image publishing to a registry on release.
- [ ] Add caching for Python dependencies to speed up CI.
- [ ] Add a workflow that validates documentation links.

## Community and governance

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
- [x] Enable GitHub Discussions, seed categories, and add category forms.

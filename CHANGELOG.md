# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Fixed

- **Prompts past `max_seq_len` extrapolated RoPE instead of failing.** The
  tokenizer had no truncation argument, generation did not check, and RoPE
  computes positions on the fly, so an over-long request produced angles the
  model never trained on and returned degraded output with no diagnostic. The
  window is now divided up front: a share is reserved for the answer, the prompt
  is trimmed keeping its end, generation is clamped to what is left, and both
  decisions are reported. A position past the window raises.
- **The API capped input far below the model's own window.** A message was
  limited to 8000 characters, a prompt to 4000, `max_new_tokens` to 2048, and the
  body to 1 MB, whatever was loaded, so the presets declaring a 1,048,576-token
  context were unreachable through it. Limits now follow the window the worker
  reports, with the old constants as the floor.
- **Conversation history never reached the model.** `generate_chat` and a
  `messages` parameter had been in the worker all along with no caller: the
  service reduced the list to its last message, and the WebSocket path built a
  one-element array. The conversation store is now shared by both transports.
- **Inference used neither tiling nor interleaved placement.** `encode_image_tiles`
  was reached only from training, contrastive pretraining and evaluation, while a
  served request ran the plain encoder on an image squashed to a fixed square. The
  inference path now tiles, keeps aspect ratio, and honours
  `mm_token_placement="interleaved"`.
- **Video ignored the frame rate it was asked for.** The writer hardcoded 100 ms a
  frame, so every clip was 10 fps whatever the request said and whatever the
  decoder had been conditioned on. The route also exposed only a prompt and a
  frame count although the worker accepted size, rate and seed, the chat path
  dropped the aspect ratio and size tier the settings panel collects, and the
  reported frame count was the number requested rather than produced.
- **nginx cut long generations at sixty seconds**, well inside the backend's
  three-minute budget, because no `proxy_read_timeout` was set.
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

- **Image generation defaults to 512x512 and accepts any aspect ratio.** It was
  square-only at a hardcoded 256 pixels, with no width, no height, and no ratio.
  A request now takes explicit `width`/`height`, or a named `aspect` at a size
  `tier`, or sizing intent read from the prompt itself ("make it 16:9",
  "1024x768", "a widescreen shot", "phone wallpaper", "a 4k panorama"), in that
  order of precedence, falling back to the configured default. The response
  reports the resolved size, the ratio, and whether it came from a parameter,
  the prompt, or the default. `resolution` remains accepted as the deprecated
  square-only alias. Arbitrary ratios require `image_gen_arch="latent_dit"`; the
  pixel U-Net says so rather than producing something misshapen.
  `diffusion_resolution` is renamed `image_train_resolution`, since it describes
  what the decoder is trained at and never had anything to do with request size.
- CI installs the model's dependencies from `requirements.txt` plus a new
  `requirements-dev.txt` instead of a hand-maintained list, so the tested
  dependency set is the shipped one. `torchvision` and `tqdm` are dropped from
  the requirements entirely (nothing imports them) and `torchaudio` moves to the
  optional extras, where its single lazy fallback in `model/data.py` belongs.
- The full `ruff.toml` rule set is now blocking in CI rather than informational,
  and the redundant error-subset step is gone.
- The website gains an ESLint flat config and a Vitest + Testing Library suite,
  both blocking in CI. It previously had neither, only `vite build`.
- The vision, audio, temporal, and diffusion attention paths now use the fused
  `scaled_dot_product_attention` kernel instead of materializing an explicit
  attention matrix. The vision encoder is shared by the audio encoder, so both
  benefit. Parameter counts are unchanged; the memory cost stops being quadratic
  in patch or pixel count, which is what put the large presets' configured
  resolutions out of reach. Attention dropout is now correctly gated on training
  mode, so eval-mode inference is deterministic.

### Added

- **Document understanding** (`model/document.py`). PDFs are a first-class input:
  page text is extracted, sorted back into reading order by finding the column
  gutter rather than trusting the content stream's order, and joined with page
  markers so a page number survives into the sequence and can be cited. A page
  with no text layer is reported as a scan rather than as an empty page. The PDF
  reader and the page rasteriser are both optional and lazily imported, and
  rasterisation is a pluggable backend so no copyleft renderer becomes a
  dependency. `POST /api/generate/document` reads an uploaded document, and
  `{"document": path}` records train on one.
- **Chat attachments reach the model.** The `attachments` array was validated,
  stored, and never read; the website always sent an empty one, and the only
  file input accepted audio. `POST /api/generate/upload` stores an image or
  document and returns the path a later turn references, the composer uploads on
  pick so an attachment can be removed before sending, and both the REST and
  WebSocket paths deliver it.
- **Long-context retrieval evaluation** (`model/eval/longcontext.py`): single-fact
  retrieval swept across the window by depth, multi-hop retrieval, and window-wide
  aggregation, each a forced choice against a known chance rate. Perplexity stays
  low on long text while retrieval fails, so it could never have answered this.
- **Dense text recognition evaluation** (`model/eval/dense_text.py`): a character
  and word error rate over rendered text, reported per font size. The audio side
  has had this metric since the harness was written; the image side had none.
- **Block-paged KV cache** (`model/modules/kv_cache.py`), with optional int8
  storage. On `framer-1t-a32b` a million-token sequence falls from 256 GiB to
  130 GiB, and `--estimate` reports the cache the config selects rather than a
  bf16 one it does not.
- **Few-step image sampling** (`FlowDistiller`). A student distilled against the
  guided teacher produces the guided field itself, so a step costs one denoiser
  forward instead of two and the step count drops to single digits: four steps
  against fifty is twenty-five times fewer calls. Opt-in, because the guidance
  scale is baked into the student.
- **Video longer than one denoising window** (`LatentVideoGenerator.sample_long`),
  through overlapped windows with latent carry-over, so duration is bounded by
  memory over time rather than by what fits in one window.
- **Cognition layer** (`model/cognition/`), optional and off by default. A
  checkpoint answers every prompt from its weights and the current context and
  keeps nothing; this wraps it in a persistent mind. `EpisodicMemory` stores each
  experience with its embedding, salience, and the affect at the time, decays
  traces with disuse, strengthens what recall returns, and evicts the weakest
  rather than the oldest. `SemanticMemory` collapses recurring episodes into
  concepts online. `CuriosityEngine` blends random-network-distillation novelty
  (which habituates, so the familiar stops being interesting) with per-topic
  learning progress (so the drive follows what is being learned rather than what
  is merely unpredictable). `AffectState` is a five-dimensional homeostat -
  valence, arousal, confidence, curiosity, fatigue - that decays toward its
  setpoints, responds to appraisal, and modulates temperature, top-p, and top-k,
  so the state reaches the decoder instead of decorating a log. `Consolidator`
  sleeps once fatigue passes threshold: prioritised replay, concept formation,
  forgetting, a first-person reflection, and an optional `train_step` callback
  that carries replayed experience into the weights. `SelfModel` tracks
  competence, interests, self-authored goals, and a bounded narrative. `Mind`
  runs the tick loop and saves and reloads the whole state, so a restart is not
  a new mind. No new dependencies, and nothing else in the model, training
  pipeline, or backend depends on it.
- **Live camera and microphone perception** (`model/cognition/perception.py`).
  `LiveSession` polls sensory sources on a schedule, in the foreground or on a
  background thread, and feeds what it sees and hears through the same tick
  loop. A `ChangeGate` attends only to readings that differ from the last
  attended one, with a forced check-in after `max_skip` polls, so a static scene
  produces one memory instead of thousands; a 2 fps camera writes only what
  changed. `CameraSource` takes a device index or a video path, so live watching
  and clip watching are one code path, and `perceive_video` pools keyframes into
  a single episode. OpenCV and sounddevice are optional and imported lazily;
  `CallableSource` drives the same pipeline with no hardware. On a text-only or
  untrained build, readings are pooled rather than understood, and
  `session.grounded` says which path is in use rather than implying the towers
  ran.
- **Script and language identification** (`model/cognition/language.py`). Script
  detection covers the world's writing systems - Latin, Cyrillic, Greek, Arabic,
  Hebrew, Devanagari, Bengali, Tamil, Telugu, Kannada, Malayalam, Sinhala, Thai,
  Lao, Khmer, Myanmar, Georgian, Armenian, Ethiopic, Cherokee, Hangul, kana,
  Han, and more - and function-word profiles resolve the languages that share a
  script. Unresolved input returns its script with low confidence rather than
  defaulting to English. Every episode remembers the language it arrived in,
  competence is tracked per language and per sense alongside per subject, topic
  labelling works for scripts that do not space their words, and a confident
  identification asks for the reply in the same language. The byte-level
  tokenizer already admits every script; what the model understands still
  follows its training corpus, and the layer makes that gap visible.
- **Cognition ops in the inference worker.** `python -m model.serve --mind PATH`
  (or `MIND_PATH`) attaches the layer, routes chat through it, returns the
  trace - what was recalled, how it felt, what sampling that produced - with
  each reply, and adds `see`, `hear`, `watch`, `live`, `wonder`, `reflect`,
  `feedback`, and `introspect`. The mind is saved after every request. Without
  the flag the worker's behaviour is unchanged.
- **Expert-parallel MoE sharding** (`model/training/expert_parallel.py`).
  `ExpertParallelPlan` assigns each rank a contiguous slice of the experts and
  `shard_experts` drops the rest, applied to the meta-device module so weights a
  rank will never hold are never allocated. `MoEFeedForward` gains an
  `expert_offset` so routing still uses global expert ids; at `ep_world == 1`
  the path is byte-identical to before and a test asserts it. `build_device_mesh`
  produces a 2D `(dp, ep)` mesh and `maybe_wrap_fsdp` shards over `dp` only,
  since sharding expert weights that expert parallelism already split would
  leave each rank with a fragment of a fragment.
- **`model/eval/` evaluation harness.** Text perplexity, token accuracy, and
  bits per byte; image Frechet distance and text-image alignment; audio SI-SDR,
  mel distance, word and character error rate, and speaker similarity; video
  Frechet distance and temporal consistency. No new dependency: the matrix
  square root the Frechet distance needs comes from `torch.linalg.eigh` on the
  symmetric PSD covariance rather than from scipy. Feature-based scores use the
  model's own encoders, so they compare FramerAI checkpoints to each other and
  **not** to published FID or FVD numbers, which the harness documentation says
  plainly. A suite that cannot run is reported as skipped with its reason,
  because a missing input is not the same as a good score.
- **Interleaved multimodal token placement and dynamic high-resolution tiling.**
  `mm_token_placement="interleaved"` writes encoded modality embeddings into
  placeholder token positions with `masked_scatter` instead of concatenating
  them ahead of the sequence, so an image sits where it was mentioned and the
  sequence does not grow. `InterleavedSequenceBuilder` in `model/data.py` emits
  the matching placeholder runs, and a count mismatch raises rather than
  corrupting the sequence. `vision_tiling` splits a high-resolution image into
  encoder-sized tiles plus a global thumbnail, choosing the grid whose aspect
  ratio best matches the image; `PatchEmbedding.interpolate_pos_encoding`
  resamples the learned position table so one encoder accepts any tile shape.
  `model/training/contrastive.py` adds symmetric InfoNCE pretraining for the
  vision tower with a learned temperature. `framer-1t-a32b` and
  `framer-2t-a49b` opt in; the defaults are unchanged.
- **Residual vector-quantized audio codec and neural vocoder**
  (`audio_gen_arch="rvq_lm"`). A causal convolutional codec at 24 kHz with a
  75 Hz acoustic frame rate turns audio into discrete tokens
  (`model/modules/rvq.py`, `model/modules/audio_codec.py`), which a
  text-conditioned transformer predicts as a classification problem
  (`model/modules/audio_lm.py`, `model/modules/rvq_audio.py`) rather than
  regressing onto a spectrogram. An inverse-STFT vocoder
  (`model/modules/vocoder.py`) predicts magnitude *and* phase and inverts once,
  replacing 32 Griffin-Lim iterations that guessed phase back after discarding
  it - the ceiling no amount of training could lift. Adds speaker conditioning
  from a reference clip and a CTC head so transcription can be trained rather
  than hoped for. The four large MoE presets opt in; the default stays
  `mel_diffusion`.
- **Spacetime diffusion transformer for video generation**
  (`video_gen_arch="spacetime_dit"`). A causal 3D VAE compressing 4x in time and
  8x in space (`model/modules/video_vae.py`), and a transformer with factorised
  spatial and temporal attention, 3D sin-cos positions, and frame-rate
  conditioning (`model/modules/spacetime_dit.py`). Both attention passes are
  batched reshapes, which removes the per-frame Python loop that made the 3D
  U-Net's throughput fall linearly with clip length. Causal convolutions and
  per-frame normalization mean frame *t* never depends on frame *t+1*, so
  duration is variable and streaming decode is possible. `generate_video` gains
  `width`, `height`, `aspect`, `fps`, and `seed`, reusing the image size buckets.
  The four large MoE presets opt in; the default stays `unet3d`.
- **Latent diffusion transformer for image generation** (`image_gen_arch="latent_dit"`).
  A KL-VAE compressing 8x into a latent grid (`model/modules/vae.py`), a diffusion
  transformer denoiser with adaLN-zero timestep and text conditioning and
  on-the-fly 2D sin-cos positions so any resolution and aspect ratio works
  (`model/modules/dit.py`), a rectified-flow objective with a 20-50 step Euler or
  Heun ODE sampler replacing the 1000-step ancestral chain
  (`model/modules/flow.py`), and **classifier-free guidance against a learned
  null-context embedding** (`model/modules/latent_diffusion.py`) - which the
  README had advertised from the beginning without any such code existing.
  `image_gen_arch` defaults to the original `unet`, so the laptop-scale presets
  are unchanged; the four large MoE presets opt in. `framer-2t-a49b` is now
  2.00T parameters.
- **Deferred initialization.** `FramerModel.from_config_meta(config)` builds the
  model's shapes on the meta device, allocating nothing, and `init_weights_()` /
  `reset_buffers()` materialize it afterwards. Without this `framer-2t-a49b`
  could not be constructed at all: the old initializer called `nn.init.normal_`,
  which fails on meta tensors. Modules owning derived buffers (RoPE frequencies,
  noise schedules, mel filterbanks) implement `reset_buffers`, and modules with
  hand-rolled parameters implement `reset_parameters`, so nothing is left holding
  uninitialized memory after `to_empty()`.
- **`model/training/checkpoint.py`**: `save_sharded`, `load_sharded`,
  `gather_full_state_dict`, and `save_full`, built on
  `torch.distributed.checkpoint`. Sharded checkpoints reshard on load, so a run
  saved on N ranks resumes on M.
- `build.py` refuses to instantiate a config that cannot fit in memory, naming
  `--estimate` and `--force`, instead of being OOM-killed without explanation.
- **`framer-2t-a49b`, the two-trillion-parameter all-modality flagship.** 1.96T
  text backbone, 49.40B active per token, 39.32B across the vision and audio
  encoders and the image, video, and audio decoders, for 2.00T in total.
  `d_model=10240`, 84 layers, 80/8 heads, 16K context, 384 fine-grained experts
  with top-4 routing and one shared expert. 384 is 3 x 128, so the experts shard
  evenly across 8/16/32/64/128-way expert-parallel meshes. Sizing it allocates
  nothing, so `--estimate` still runs on a laptop and in CI.
- `MULTIMODAL_TOWERS` is exported from `model/utils/helpers.py` as the single
  source of truth for the tower list, replacing three copies of the same tuple.
  `estimate_params` and `estimate_multimodal_params` accept `strict=True` to
  re-raise rather than degrade to "could not be sized"; the tests use it.
- `tests/test_scale_config.py` pins the flagship's total, active budget,
  per-tower floors, memory arithmetic, and expert-mesh divisibility, so the
  decoder replacements that follow cannot silently drift the documented numbers.
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

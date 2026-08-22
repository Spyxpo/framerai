# FramerAI Troubleshooting Guide

This guide provides practical solutions for common CUDA, PyTorch, VRAM/OOM, dependency, and environment setup issues when running or training FramerAI.

---

## Table of Contents

- [CUDA and PyTorch Environment Issues](#cuda-and-pytorch-environment-issues)
  - [PyTorch Cannot Detect GPU (`torch.cuda.is_available()` is False)](#pytorch-cannot-detect-gpu-torchcudais_available-is-false)
  - [CUDA Version and Driver Mismatches](#cuda-version-and-driver-mismatches)
  - [Apple Silicon (MPS) Execution & Fallbacks](#apple-silicon-mps-execution--fallbacks)
  - [Selecting Specific GPUs (`CUDA_VISIBLE_DEVICES`)](#selecting-specific-gpus-cuda_visible_devices)
- [VRAM and Out-Of-Memory (OOM) Management](#vram-and-out-of-memory-oom-management)
  - [Estimating Memory Requirements Before Training](#estimating-memory-requirements-before-training)
  - [Choosing an Appropriate Model Preset](#choosing-an-appropriate-model-preset)
  - [Practical VRAM Reduction Strategies](#practical-vram-reduction-strategies)
  - [PyTorch Memory Allocator Tuning](#pytorch-memory-allocator-tuning)
- [Python and System Dependency Issues](#python-and-system-dependency-issues)
  - [Python Version Requirements](#python-version-requirements)
  - [`libsndfile` Missing Error for `soundfile`](#libsndfile-missing-error-for-soundfile)
  - [Optional Extras and Development Tooling](#optional-extras-and-development-tooling)
  - [Node.js Environment Issues (Backend & Website)](#nodejs-environment-issues-backend--website)
- [CPU vs GPU Execution Guidance](#cpu-vs-gpu-execution-guidance)
  - [Automatic Device Resolution (`--device auto`)](#automatic-device-resolution---device-auto)
  - [Running on CPU](#running-on-cpu)
  - [Docker GPU Pass-through](#docker-gpu-pass-through)
- [Useful Diagnostic Commands](#useful-diagnostic-commands)
- [Related Documentation](#related-documentation)

---

## CUDA and PyTorch Environment Issues

### PyTorch Cannot Detect GPU (`torch.cuda.is_available()` is False)

If `python3 build.py` defaults to CPU or reports that CUDA is unavailable:

1. **Verify PyTorch CUDA Installation**: Standard `pip install torch` may install CPU-only wheels on some platforms. Reinstall PyTorch with explicit CUDA support from the official PyTorch repository:
   ```bash
   # Example for CUDA 12.1:
   pip install --force-reinstall torch --index-url https://download.pytorch.org/whl/cu121
   ```
2. **Check GPU Drivers**: Verify that NVIDIA drivers are installed and recognized by your operating system:
   ```bash
   nvidia-smi
   ```
3. **Verify CUDA Availability via Python**:
   ```bash
   python3 -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('Device Count:', torch.cuda.device_count()); print('Device Name:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"
   ```

### CUDA Version and Driver Mismatches

If you encounter errors like `CUDA error: no kernel image is available for execution on the device` or `CUDA driver version is insufficient for CUDA runtime version`:

- Ensure your installed NVIDIA driver supports the CUDA toolkit version used by PyTorch.
- Check PyTorch and CUDA driver compatibility matrix at [pytorch.org](https://pytorch.org/).

### Apple Silicon (MPS) Execution & Fallbacks

On macOS with Apple Silicon (M1/M2/M3/M4):

- FramerAI supports `--device mps` for GPU acceleration via Metal Performance Shaders.
- If an unsupported PyTorch operation throws an MPS fallback warning or error, enable CPU fallback:
  ```bash
  export PYTORCH_ENABLE_MPS_FALLBACK=1
  ```

### Selecting Specific GPUs (`CUDA_VISIBLE_DEVICES`)

To run FramerAI on a specific GPU in a multi-GPU system:

```bash
# Run on GPU 0 only
CUDA_VISIBLE_DEVICES=0 python3 build.py --mode all --size small --device cuda

# Run on GPU 1 only
CUDA_VISIBLE_DEVICES=1 python3 build.py --mode all --size small --device cuda
```

---

## VRAM and Out-Of-Memory (OOM) Management

### Estimating Memory Requirements Before Training

FramerAI provides a zero-allocation estimator that evaluates model parameter counts and memory budgets on PyTorch's `meta` device without allocating GPU memory. Always estimate your configuration before running training:

```bash
# Print budget for a specific preset
python3 build.py --preset framer-small --estimate

# List all built-in presets and parameter ladders
python3 build.py --list-presets
```

### Choosing an Appropriate Model Preset

Select a preset matching your hardware capabilities:

| Preset | Parameters | Active/Token | Typical Hardware / Target |
|--------|------------|--------------|---------------------------|
| `framer-tiny` | ~39M | ~19M | CPU / Laptop / Rapid local testing |
| `framer-small` | ~236M | ~142M | Single consumer GPU (4–8 GB VRAM) |
| `framer-medium` | ~941M | ~429M | Mid-range GPU (12–16 GB VRAM) |
| `framer-large` | ~3.7B | ~1.2B | High-end GPU (24 GB+ VRAM) |
| `framer-1t-a32b` | ~1.00T | ~32B | Distributed multi-node cluster (requires FSDP2) |
| `framer-2t-a49b` | ~2.00T | ~49B | Multi-node flagship cluster (3.6+ TiB weights) |

> **Note**: Ultra-large presets like `framer-1t-a32b` and `framer-2t-a49b` are multi-node architecture definitions that cannot fit on single GPUs or laptops. Attempting to instantiate them without cluster resources will trigger memory guardrails unless `--force` is supplied.

### Practical VRAM Reduction Strategies

If you encounter `torch.cuda.OutOfMemoryError: CUDA out of memory` during training:

1. **Lower Micro Batch Size (`--batch-size`)**:
   Reduce `--batch-size` (e.g., from `8` to `2` or `1`).
2. **Use Gradient Accumulation (`--grad-accum`)**:
   Maintain effective batch size while reducing peak memory by accumulating gradients over multiple steps:
   ```bash
   # Effective batch size = 2 * 4 = 8, with 1/4th the micro-batch memory footprint
   python3 build.py --mode train --size small --batch-size 2 --grad-accum 4
   ```
3. **Enable Mixed Precision (`--precision bf16` / `--precision fp16`)**:
   Use 16-bit precision autocast (default is `bf16` on supported hardware; use `fp16` on older GPUs):
   ```bash
   python3 build.py --mode train --size small --precision bf16
   ```
4. **Enable Activation Checkpointing (`--grad-checkpointing`)**:
   Recompute intermediate activations during backward pass instead of storing them:
   ```bash
   python3 build.py --mode train --size small --grad-checkpointing
   ```
5. **Train Text-Only Core (`--text-only`)**:
   Skip instantiating vision/audio encoders and decoders if only training the LLM backbone:
   ```bash
   python3 build.py --mode train --size small --text-only
   ```
6. **Combine Memory-Saving Flags**:
   ```bash
   python3 build.py --mode all --size small \
     --batch-size 2 --grad-accum 4 \
     --precision bf16 --grad-checkpointing \
     --data-dir data
   ```

### PyTorch Memory Allocator Tuning

To reduce CUDA memory fragmentation on long training runs, set the PyTorch allocator environment variable before running training:

```bash
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

---

## Python and System Dependency Issues

### Python Version Requirements

FramerAI requires **Python 3.10 or newer**. Check your installed version:

```bash
python3 --version
```

If using multiple Python versions, create your virtual environment explicitly with Python 3.10+:

```bash
python3.12 -m venv venv
source venv/bin/activate
```

> **Note**: After activating your virtual environment (`source venv/bin/activate`), `python` will point directly to your environment's Python executable.

### `libsndfile` Missing Error for `soundfile`

If loading audio files fails with an error indicating `libsndfile` cannot be found:

1. **Install System Library**:
   - **Linux (Ubuntu / Debian)**:
     ```bash
     sudo apt-get update && sudo apt-get install -y libsndfile1
     ```
   - **macOS (Homebrew)**:
     ```bash
     brew install libsndfile
     ```
2. **Optional Fallback Loader**:
   FramerAI's data pipeline (`model/data.py`) automatically falls back to `torchaudio` if `soundfile` is unavailable:
   ```bash
   pip install torchaudio
   ```

### Optional Extras and Development Tooling

The core runtime dependencies are listed in `requirements.txt`. Additional optional extras can be installed based on your use case:

```bash
# Optional runtime extras:
#   torchaudio   - audio loader fallback
#   safetensors  - fast/safe weight serialization (build.py --mode export)
#   flash-attn   - accelerated attention kernels on supported CUDA GPUs
#   opencv-python - live camera input for model/cognition
#   sounddevice   - live microphone input for model/cognition
pip install torchaudio safetensors

# Full development and testing stack:
pip install -r requirements.txt -r requirements-dev.txt
```

### Node.js Environment Issues (Backend & Website)

If starting the Express backend or React frontend fails:

- **Backend Requirement**: Node 18 or newer.
  ```bash
  cd backend
  npm install
  cp .env.example .env
  npm run dev
  ```
- **Website Requirement**: Node 20.19 or newer (required by Vite 8 tooling).
  ```bash
  cd website
  npm install
  npm run dev
  ```

---

## CPU vs GPU Execution Guidance

### Automatic Device Resolution (`--device auto`)

When `--device auto` (or default `./train.sh`) is specified, FramerAI automatically selects the best available device in order:
1. `cuda` (if `torch.cuda.is_available()` is True)
2. `mps` (if `torch.backends.mps.is_available()` is True)
3. `cpu` (fallback)

### Running on CPU

To force CPU execution (e.g. for testing on non-GPU instances):

```bash
python3 build.py --mode all --size tiny --device cpu
```

All core model components, data loaders, and tests are designed to execute cleanly on CPU.

### Docker GPU Pass-through

When running FramerAI in Docker on a GPU host:

1. Ensure [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) is installed.
2. Run the model image with GPU pass-through:
   ```bash
   docker run --rm --gpus all \
     -v $(pwd)/data:/app/data \
     -v framerai_checkpoints:/app/checkpoints \
     framerai-model --mode all --size small --data-dir data --device cuda
   ```

---

## Useful Diagnostic Commands

Run these diagnostic commands to verify your setup, check resources, or validate local changes (use `python3` or your virtual environment's `python` command):

```bash
# 1. Inspect Python & PyTorch CUDA / MPS status
python3 -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('MPS available:', getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available())"

# 2. Inspect NVIDIA GPU status & VRAM usage
nvidia-smi

# 3. Estimate model parameters and memory usage without allocation
python3 build.py --preset framer-small --estimate

# 4. List all available model presets
python3 build.py --list-presets

# 5. Run Python linter & syntax checks
ruff check model build.py scripts tests conftest.py
python3 -m compileall -q model build.py scripts tests conftest.py

# 6. Run Python test suite (runs on CPU in seconds)
python3 -m pytest -q

# 7. Run Backend tests
cd backend && npm test

# 8. Run Website linter and tests
cd website && npm run lint && npm test
```

---

## Related Documentation

- [README.md](README.md) - Project overview, features, model size table, and quick start guide.
- [GUIDE.md](GUIDE.md) - Detailed technical architecture, build.py usage, and cognition layer guide.
- [data/README.md](data/README.md) - Dataset layout, supported formats (text, image captions, audio captions), and examples.
- [CONTRIBUTING.md](CONTRIBUTING.md) - Development setup, branching model, coding standards, and running checks locally.
- [TODOS.md](TODOS.md) - Project roadmap and open tasks.

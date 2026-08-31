#!/usr/bin/env python3
"""
FramerAI Model Builder & Trainer

Usage:
    python build.py --mode build          # Build model architecture and save initial weights
    python build.py --mode train          # Train the model on data
    python build.py --mode export         # Export trained model for serving
    python build.py --mode all            # Build, train, and export
"""

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from model.configs import FramerConfig, list_presets, resolve_preset_name
from model.data import (
    AudioCaptionDataset,
    ImageCaptionDataset,
    TextCorpusDataset,
    build_packed_dataset,
    build_text_dataset,
    iter_text_records,
)
from model.framer import FramerModel
from model.tokenizer import FramerTokenizer
from model.training import (
    cleanup_distributed,
    get_rank,
    get_world_size,
    init_distributed,
    is_main_process,
    maybe_wrap_fsdp,
    train_language_model,
)
from model.training.optim import build_optimizer
from model.training.schedule import build_scheduler
from model.utils import (
    MULTIMODAL_TOWERS,
    apply_seed,
    count_parameters,
    estimate_params,
    get_device,
    human_params,
    load_checkpoint,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("framerai")


# Minimal built-in corpus used only when no local data is found, so a smoke
# test still runs. Add real .txt/.jsonl files under --data-dir for real training.
BUILTIN_SAMPLE_TEXTS = [
    "Hello, how can I help you today? I am FramerAI, your multimodal assistant.",
    "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
    "To create an image, simply describe what you want to see.",
    "import torch\nimport torch.nn as nn",
    "The transformer architecture revolutionized natural language processing.",
    "Videos are generated frame by frame using temporal diffusion.",
    "Audio is generated as a mel spectrogram and reconstructed into a waveform.",
    "Machine learning models learn patterns from data to make predictions.",
    "def hello_world():\n    print('Hello, World!')\n\nhello_world()",
] * 20


# ---------------------------------------------------------------------------
# Build & Train
# ---------------------------------------------------------------------------

def _available_memory_bytes() -> int:
    """Physical RAM, or 0 when it cannot be determined."""
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (AttributeError, ValueError, OSError):
        return 0


def check_buildable(config: FramerConfig, force: bool = False):
    """Refuse to instantiate a config that cannot fit in memory.

    ``build_model`` allocates the whole model on one device. For the largest
    presets that is multiple terabytes, and without this guard the process is
    simply OOM-killed with no explanation of why.
    """
    est = estimate_params(config)
    # fp32 parameters plus headroom for the copy torch.save makes.
    needed = est["model_total"] * 4
    available = _available_memory_bytes()
    if force or not available or needed < available * 0.8:
        return est

    gib = 1024 ** 3
    raise SystemExit(
        f"'{config.preset or 'custom'}' needs about {needed / gib:.0f} GiB to instantiate "
        f"({est['model_total_h']} parameters in fp32) but this machine has "
        f"{available / gib:.0f} GiB.\n"
        f"  Size it without allocating:  python build.py --preset {config.preset} --estimate\n"
        f"  Train it across hosts:       see the sharded checkpoint path in "
        f"model/training/checkpoint.py\n"
        f"  Build it anyway:             --force"
    )


def build_model(config: FramerConfig, output_dir: str, data_dir: str = "data", force: bool = False):
    """Initialize model architecture and save initial checkpoint."""
    logger.info("Building FramerAI model...")
    logger.info(f"Config: d_model={config.d_model}, layers={config.n_layers}, heads={config.n_heads}")

    check_buildable(config, force=force)
    model = FramerModel(config)
    num_params = count_parameters(model)
    logger.info(f"Total trainable parameters: {num_params:,}")

    # Build tokenizer
    logger.info("Building tokenizer...")
    tokenizer = FramerTokenizer(config.vocab_size)

    # Train the tokenizer on the local corpus when available, otherwise on the
    # built-in sample so a fresh clone still builds.
    corpus = list(iter_text_records(data_dir))
    if corpus:
        logger.info(f"Training tokenizer on {len(corpus)} local text records from '{data_dir}'")
    else:
        logger.warning(f"No local data in '{data_dir}'; training tokenizer on the built-in sample.")
        corpus = BUILTIN_SAMPLE_TEXTS
    tokenizer.train(corpus, target_vocab_size=config.vocab_size)
    actual_vocab_size = tokenizer.first_merge_id + len(tokenizer.merges)
    logger.info(f"Tokenizer vocabulary size: {actual_vocab_size}")

    # Warn if actual vocabulary is smaller than configured target
    if actual_vocab_size < config.vocab_size:
        logger.warning(
            f"Trained vocabulary size ({actual_vocab_size}) is smaller than configured "
            f"target ({config.vocab_size}). This may indicate insufficient training corpus "
            f"or the target size exceeds what can be learned from the available data."
        )

    # Save
    os.makedirs(output_dir, exist_ok=True)
    checkpoint_path = os.path.join(output_dir, "model_init.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": asdict(config),
        "step": 0,
        "loss": float("inf"),
    }, checkpoint_path)

    tokenizer.save(os.path.join(output_dir, "tokenizer"))

    # Save config
    with open(os.path.join(output_dir, "config.json"), "w") as f:
        json.dump(asdict(config), f, indent=2)

    logger.info(f"Model saved to {checkpoint_path}")
    logger.info(f"Tokenizer saved to {os.path.join(output_dir, 'tokenizer')}")
    return model, tokenizer


def train_model(config: FramerConfig, output_dir: str, resume: str = None,
                data_dir: str = "data", train_modalities: bool = False,
                shard_dir: str = None):
    """Train the LM core from scratch on a local corpus.

    Uses the modern training stack (bf16/fp16 autocast, warmup-cosine LR, MoE
    aux loss, activation checkpointing, optional FSDP2 under torchrun). Streams
    packed token shards when ``shard_dir`` is provided, else falls back to the
    in-memory corpus loader.
    """
    device = get_device(config.device)
    distributed = init_distributed(device)
    rank, world = get_rank(), get_world_size()
    if is_main_process():
        est = estimate_params(config)
        logger.info(f"Training on {device} | world_size={world} | {est['summary']}")

    # Load or build model
    model = FramerModel(config).to(device)

    start_step = 0

    # When resuming, load model weights BEFORE FSDP wrapping
    if resume:
        logger.info(f"Resuming from {resume}")
        start_step, prev_loss = load_checkpoint(resume, model=model)
        logger.info(f"Loaded model weights from step {start_step} (loss: {prev_loss:.4f})")
    else:
        init_path = os.path.join(output_dir, "model_init.pt")
        if os.path.exists(init_path):
            start_step, _ = load_checkpoint(init_path, model=model)

    tokenizer_path = os.path.join(output_dir, "tokenizer")
    tokenizer = FramerTokenizer.load(tokenizer_path) if os.path.exists(tokenizer_path) else FramerTokenizer(config.vocab_size)

    # Shard the model across ranks when distributed (no-op single-device).
    # This must happen BEFORE optimizer creation so optimizer sees wrapped parameters.
    model = maybe_wrap_fsdp(model, config, device)

    # Create optimizer and scheduler from the (potentially wrapped) model
    optimizer = None
    scheduler = None
    if resume:
        # Now create optimizer/scheduler from wrapped model and restore their state
        optimizer = build_optimizer(model, config)
        scheduler = build_scheduler(optimizer, config)
        # Restore optimizer/scheduler state (model already loaded above)
        load_checkpoint(resume, model=None, optimizer=optimizer, scheduler=scheduler)
        logger.info(f"Restored optimizer and scheduler state at step {start_step}")

    seq_len = min(config.max_seq_len, 1024)

    # Prefer streaming packed shards; fall back to the in-memory corpus loader.
    packed = build_packed_dataset(shard_dir, seq_len, rank=rank, world_size=world, seed=config.seed) if shard_dir else None
    if packed is not None:
        logger.info(f"Streaming packed shards from '{shard_dir}' (seq_len={seq_len})")
        loader = DataLoader(packed, batch_size=config.batch_size, num_workers=2, pin_memory=(device.type == "cuda"))
    else:
        text_dataset = build_text_dataset(data_dir, tokenizer, max_len=seq_len)
        if text_dataset is None:
            logger.warning(
                f"No text data found in '{data_dir}'. Using the built-in sample corpus. "
                f"Add .txt/.jsonl under '{data_dir}', or prepare shards with scripts/prepare_data.py."
            )
            text_dataset = TextCorpusDataset(BUILTIN_SAMPLE_TEXTS, tokenizer, max_len=128)
        logger.info(f"Loaded {len(text_dataset)} text samples from '{data_dir}'")
        sampler = DistributedSampler(text_dataset) if distributed else None
        loader = DataLoader(
            text_dataset, batch_size=config.batch_size,
            shuffle=(sampler is None), sampler=sampler,
            pin_memory=(device.type == "cuda"),
        )

    train_language_model(
        config, model, loader, device, output_dir,
        start_step=start_step, logger=logger,
        optimizer=optimizer, scheduler=scheduler,
    )

    # Optional image/audio generation training (single-device, full multimodal).
    if train_modalities and not config.text_only and not distributed and is_main_process():
        train_modality_generators(model, tokenizer, config, data_dir, device)

    cleanup_distributed()
    return model


def train_modality_generators(model, tokenizer, config, data_dir, device, max_steps: int = 200):
    """Train the image and audio generators on local caption pairs.

    Runs only when caption pairs exist under the data directory. Each generator
    trains for a small, bounded number of steps so the pipeline stays runnable
    on modest hardware; scale up for real training.
    """
    model.train()

    def _run(dataset, key, label):
        if len(dataset) == 0:
            logger.info(f"No {label} caption pairs found in '{data_dir}'; skipping {label} training.")
            return
        collate_fn = getattr(dataset, "collate_fn", None)
        loader = DataLoader(dataset, batch_size=max(1, config.batch_size // 2), shuffle=True, collate_fn=collate_fn)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
        logger.info(f"Training {label} generator on {len(dataset)} pairs")
        steps = 0
        while steps < max_steps:
            for batch in loader:
                if steps >= max_steps:
                    break
                input_ids = batch["input_ids"].to(device)
                target = batch[key].to(device)
                kwargs = {key: target}
                if key == "target_audio":
                    if "target_waveform" in batch:
                        kwargs["target_waveform"] = batch["target_waveform"].to(device)
                    if config.use_ctc_head and "audio" in batch and "ctc_targets" in batch:
                        kwargs["audio"] = batch["audio"].to(device)
                        kwargs["ctc_targets"] = batch["ctc_targets"].to(device)
                        if "ctc_input_lengths" in batch:
                            kwargs["ctc_input_lengths"] = batch["ctc_input_lengths"].to(device)
                        if "ctc_target_lengths" in batch:
                            kwargs["ctc_target_lengths"] = batch["ctc_target_lengths"].to(device)
                results = model(input_ids=input_ids, **kwargs)
                loss = results["loss"] if "loss" in results else results[f"{label}_loss"]
                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                steps += 1
                if steps % 20 == 0:
                    ctc_str = f" | ctc {results['ctc_loss'].item():.4f}" if "ctc_loss" in results else ""
                    logger.info(f"  {label} step {steps}/{max_steps} | loss {loss.item():.4f}{ctc_str}")


    _run(ImageCaptionDataset(data_dir, tokenizer, resolution=config.image_train_resolution), "target_images", "image")
    _run(AudioCaptionDataset(data_dir, tokenizer, config), "target_audio", "audio")


def export_model(config: FramerConfig, output_dir: str, export_dir: str = None):
    """Export trained model for serving."""
    export_dir = export_dir or os.path.join(output_dir, "export")
    os.makedirs(export_dir, exist_ok=True)

    # Load the final or best checkpoint
    final_path = os.path.join(output_dir, "model_final.pt")
    init_path = os.path.join(output_dir, "model_init.pt")
    ckpt_path = final_path if os.path.exists(final_path) else init_path

    if not os.path.exists(ckpt_path):
        logger.error(f"No checkpoint found at {ckpt_path}. Run build or train first.")
        sys.exit(1)

    logger.info(f"Loading checkpoint: {ckpt_path}")
    model = FramerModel(config)
    load_checkpoint(ckpt_path, model)
    model.eval()

    # Save model state dict (inference-ready)
    export_path = os.path.join(export_dir, "framerai_model.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": asdict(config),
    }, export_path)

    # Optional safetensors export (secure, framework-portable weights).
    try:
        from safetensors.torch import save_file

        st_path = os.path.join(export_dir, "framerai_model.safetensors")
        tensors = {k: v.contiguous() for k, v in model.state_dict().items()}
        # Weight tying makes lm_head share storage with token_embed; drop the
        # duplicate so safetensors (which forbids shared storage) accepts it.
        tensors.pop("lm_head.weight", None)
        save_file(tensors, st_path, metadata={"format": "pt", "framer_preset": str(config.preset)})
        logger.info(f"  Safetensors: {st_path}")
    except ImportError:
        logger.info("  (install 'safetensors' to also export a .safetensors file)")

    # Optional ONNX export.
    try:
        import onnx  # noqa: F401

        onnx_path = os.path.join(export_dir, "framerai_model.onnx")
        dummy_ids = torch.zeros((1, 8), dtype=torch.long)

        class _ONNXWrapper(nn.Module):
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, input_ids):
                return self.m.forward_text(input_ids)

        torch.onnx.export(
            _ONNXWrapper(model),
            (dummy_ids,),
            onnx_path,
            input_names=["input_ids"],
            output_names=["logits"],
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "logits": {0: "batch", 1: "sequence"},
            },
            opset_version=14,
            dynamo=False,
        )
        logger.info(f"  ONNX: {onnx_path}")
    except ImportError:
        logger.info("  (install 'onnx' and 'onnxruntime' to also export an .onnx model)")

    # Copy tokenizer
    tokenizer_src = os.path.join(output_dir, "tokenizer")
    tokenizer_dst = os.path.join(export_dir, "tokenizer")
    if os.path.exists(tokenizer_src):
        import shutil
        if os.path.exists(tokenizer_dst):
            shutil.rmtree(tokenizer_dst)
        shutil.copytree(tokenizer_src, tokenizer_dst)

    # Save model info
    info = {
        "model_name": "FramerAI",
        "parameters": count_parameters(model),
        "config": asdict(config),
        "modalities": ["text", "code", "image", "video", "audio"],
    }
    with open(os.path.join(export_dir, "model_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    logger.info(f"Model exported to {export_dir}")
    logger.info(f"  Model: {export_path}")
    logger.info(f"  Parameters: {count_parameters(model):,}")


def eval_model(config: FramerConfig, output_dir: str, benchmark_dir: str = "benchmarks",
               eval_output: str = None, seq_len: int = 128, batch_size: int = 4,
               code_limit: int = None):
    """Run standard benchmarks on a trained checkpoint and report results.

    Loads the final or best checkpoint, runs text and code benchmarks, and
    reports results both to stdout and optionally to a JSON file. Missing
    benchmark data is reported as skipped rather than producing fake results.
    """
    device = get_device(config.device)
    logger.info(f"Running evaluation on {device}")
    logger.info(f"Config: {config.preset or 'custom'} | seed={config.seed}")

    # Load checkpoint
    final_path = os.path.join(output_dir, "model_final.pt")
    init_path = os.path.join(output_dir, "model_init.pt")
    ckpt_path = final_path if os.path.exists(final_path) else init_path

    if not os.path.exists(ckpt_path):
        logger.error(f"No checkpoint found at {ckpt_path}. Run build or train first.")
        sys.exit(1)

    logger.info(f"Loading checkpoint: {ckpt_path}")
    model = FramerModel(config).to(device)
    load_checkpoint(ckpt_path, model)
    model.eval()

    tokenizer_path = os.path.join(output_dir, "tokenizer")
    if not os.path.exists(tokenizer_path):
        logger.error(f"No tokenizer found at {tokenizer_path}. Run build first.")
        sys.exit(1)

    tokenizer = FramerTokenizer.load(tokenizer_path)

    # Build generator for code evaluation
    from model.generate import FramerGenerator
    generator = FramerGenerator(model, tokenizer, str(device))

    # Register benchmark suites
    from model.eval import EvalHarness
    from model.eval.benchmarks import (
        evaluate_code_benchmark,
        evaluate_instruction_following,
        evaluate_text_benchmark,
    )

    harness = EvalHarness(model, str(device))

    @harness.suite("instruction-following")
    def _instruction_following(model, device, **_):
        result = evaluate_instruction_following(generator)
        return {**result.metrics, "samples": result.samples}

    @harness.suite("wikitext-2")
    def _wikitext(model, device, **_):
        text_path = os.path.join(benchmark_dir, "wikitext-2", "test.txt")
        result = evaluate_text_benchmark(
            model, tokenizer, text_path,
            device=device, seq_len=seq_len, batch_size=batch_size,
        )
        # Flatten metrics and add sample count
        return {**result.metrics, "samples": result.samples}

    @harness.suite("humaneval")
    def _humaneval(model, device, **_):
        code_path = os.path.join(benchmark_dir, "humaneval", "HumanEval.jsonl")
        result = evaluate_code_benchmark(
            generator, code_path, seed=config.seed, limit=code_limit,
        )
        # Flatten metrics and add sample count
        return {**result.metrics, "samples": result.samples}

    # Run evaluation
    logger.info("Running benchmarks...")
    report = harness.run()

    # Print summary
    logger.info("=" * 60)
    logger.info("Evaluation Results")
    logger.info("=" * 60)
    print(report.summary())

    # Optionally write JSON
    if eval_output:
        os.makedirs(os.path.dirname(eval_output) or ".", exist_ok=True)
        with open(eval_output, "w") as f:
            f.write(report.to_json())
        logger.info(f"Results written to {eval_output}")

    # Exit with error if all suites were skipped
    if report.metrics == {}:
        logger.error("All benchmark suites were skipped. Check benchmark data paths.")
        sys.exit(1)


def print_estimate(config: FramerConfig):
    """Print the whole-model parameter and memory budget without building it."""
    est = estimate_params(config)
    gb = 1024 ** 3

    print(f"Preset: {config.preset or 'custom'}")
    print(f"  d_model={config.d_model} layers={config.n_layers} "
          f"heads={config.n_heads}/{config.kv_heads} seq={config.max_seq_len}")
    if config.use_moe:
        print(f"  MoE: {config.n_experts} experts, top-{config.n_experts_per_tok}, "
              f"{config.n_shared_experts} shared, expert_d_ff={config.expert_d_ff or config.d_ff}, "
              f"{config.first_dense_layers} dense layers first")
    print()
    print(f"  Text backbone      {est['total_h']:>9s} total   {est['active_h']:>9s} active/token")

    mm = est["multimodal"]
    if mm is None:
        print("  Multimodal towers  (could not be sized on this build)")
    elif est["multimodal_total"]:
        for name in MULTIMODAL_TOWERS:
            print(f"    {name:<18s} {human_params(mm[name]):>9s}")
        print(f"  Multimodal total   {est['multimodal_h']:>9s}")
    else:
        print("  Multimodal towers  none (text_only build)")

    print()
    print(f"  MODEL TOTAL        {est['model_total_h']:>9s} parameters")
    print(f"  Weights (bf16)     {est['bf16_bytes'] / gb:>9.1f} GiB")
    print(f"  Training state     {est['adamw_bytes'] / gb:>9.1f} GiB "
          f"(weights + fp32 master + AdamW moments)")
    print(f"  KV cache           {est['kv_cache_bytes'] / gb:>9.1f} GiB "
          f"(bf16, one sequence at {config.max_seq_len:,} tokens)")
    if config.context_extension > 1.0:
        print(f"  Context            {config.max_seq_len:,} tokens "
              f"({config.context_extension:.0f}x {config.rope_original_max_seq_len:,} "
              f"via {config.rope_scaling_type} RoPE scaling)")


def train_sft_model(config: FramerConfig, output_dir: str, data_dir: str = "data", resume: str = None):
    """Run Supervised Fine-Tuning (SFT) pass."""
    from model.data import SFTDataset
    from model.training import train_sft

    device = get_device(config.device)
    distributed = init_distributed(device)
    world = get_world_size()

    if is_main_process():
        logger.info(f"SFT Training on {device} | world_size={world}")

    model = FramerModel(config).to(device)
    if resume:
        load_checkpoint(resume, model=model)
    else:
        init_path = os.path.join(output_dir, "model_init.pt")
        if os.path.exists(init_path):
            load_checkpoint(init_path, model=model)

    tokenizer_path = os.path.join(output_dir, "tokenizer")
    tokenizer = FramerTokenizer.load(tokenizer_path) if os.path.exists(tokenizer_path) else FramerTokenizer(config.vocab_size)

    dataset = SFTDataset(data_dir, tokenizer, max_len=min(config.max_seq_len, 512))
    if len(dataset) == 0:
        sample_path = os.path.join(os.path.dirname(__file__), "data", "examples", "sft_sample.jsonl")
        if os.path.exists(sample_path):
            logger.warning(f"No SFT data found in '{data_dir}'. Using sample SFT dataset.")
            dataset = SFTDataset(sample_path, tokenizer, max_len=min(config.max_seq_len, 512))

    if len(dataset) == 0:
        raise ValueError("No SFT data available.")

    sampler = DistributedSampler(dataset) if distributed else None
    loader = DataLoader(
        dataset, batch_size=config.batch_size,
        shuffle=(sampler is None), sampler=sampler,
        pin_memory=(device.type == "cuda"),
    )

    model = maybe_wrap_fsdp(model, config, device)
    train_sft(config, model, loader, device, output_dir, logger_obj=logger)
    cleanup_distributed()
    return model


def train_dpo_model(config: FramerConfig, output_dir: str, data_dir: str = "data", beta: float = 0.1, resume: str = None):
    """Run Direct Preference Optimization (DPO) pass."""
    from model.data import DPODataset
    from model.training import train_dpo

    device = get_device(config.device)
    distributed = init_distributed(device)
    world = get_world_size()

    if is_main_process():
        logger.info(f"DPO Training on {device} | beta={beta} | world_size={world}")

    policy_model = FramerModel(config).to(device)
    if resume:
        load_checkpoint(resume, model=policy_model)
    else:
        final_path = os.path.join(output_dir, "model_final.pt")
        init_path = os.path.join(output_dir, "model_init.pt")
        ckpt_path = final_path if os.path.exists(final_path) else init_path
        if os.path.exists(ckpt_path):
            load_checkpoint(ckpt_path, model=policy_model)

    ref_model = FramerModel.from_config_meta(config)
    ref_model.to_empty(device=device)
    ref_model.load_state_dict(policy_model.state_dict())
    ref_model.eval()
    for param in ref_model.parameters():
        param.requires_grad = False

    tokenizer_path = os.path.join(output_dir, "tokenizer")
    tokenizer = FramerTokenizer.load(tokenizer_path) if os.path.exists(tokenizer_path) else FramerTokenizer(config.vocab_size)

    dataset = DPODataset(data_dir, tokenizer, max_len=min(config.max_seq_len, 512))
    if len(dataset) == 0:
        sample_path = os.path.join(os.path.dirname(__file__), "data", "examples", "dpo_sample.jsonl")
        if os.path.exists(sample_path):
            logger.warning(f"No DPO data found in '{data_dir}'. Using sample DPO dataset.")
            dataset = DPODataset(sample_path, tokenizer, max_len=min(config.max_seq_len, 512))

    if len(dataset) == 0:
        raise ValueError("No DPO data available.")

    sampler = DistributedSampler(dataset) if distributed else None
    loader = DataLoader(
        dataset, batch_size=config.batch_size,
        shuffle=(sampler is None), sampler=sampler,
        pin_memory=(device.type == "cuda"),
    )

    policy_model = maybe_wrap_fsdp(policy_model, config, device)
    train_dpo(config, policy_model, ref_model, loader, device, output_dir, beta=beta, logger=logger)
    cleanup_distributed()
    return policy_model


def _make_parser() -> argparse.ArgumentParser:
    """Return the argument parser. Extracted so tests can invoke it directly."""
    parser = argparse.ArgumentParser(description="FramerAI Model Builder")
    parser.add_argument("--mode", choices=["build", "train", "sft", "dpo", "export", "eval", "all"], default="build", help="Operation mode")
    parser.add_argument("--output-dir", default="checkpoints", help="Output directory")
    parser.add_argument("--export-dir", default=None, help="Export directory")
    parser.add_argument("--resume", default=None, help="Checkpoint to resume from")

    # Training data
    parser.add_argument("--data-dir", default="data", help="Directory with local training data (.txt / .jsonl)")
    parser.add_argument("--train-modalities", action="store_true",
                        help="Also train the image and audio generators on local caption pairs")
    parser.add_argument("--use-ctc-head", action="store_true",
                        help="Enable CTC auxiliary objective over audio encoder output")
    parser.add_argument("--ctc-loss-weight", type=float, default=None,
                        help="Weight for CTC auxiliary loss term")
    parser.add_argument("--beta", type=float, default=0.1, help="DPO beta scale hyperparameter (default 0.1)")

    # Model config overrides
    parser.add_argument("--d-model", type=int, default=None, help="Model dimension")
    parser.add_argument("--n-layers", type=int, default=None, help="Number of transformer layers")
    parser.add_argument("--n-heads", type=int, default=None, help="Number of attention heads")
    parser.add_argument("--max-steps", type=int, default=None, help="Max training steps")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--warmup-steps", type=int, default=None, help="LR warmup steps")
    parser.add_argument("--grad-accum", type=int, default=None,
                        help="Gradient accumulation steps (effective batch = batch-size × grad-accum)")
    parser.add_argument("--device", type=str, default=None, help="Device (auto, cpu, cuda, mps)")
    parser.add_argument("--tokenizer-vocab-size", type=int, default=None,
                        help="Override tokenizer vocabulary size for smoke/development runs (e.g., 1000)")

    # Size presets (named registry, scaling from ~15M to 1T-MoE)
    parser.add_argument("--preset", default=None,
                        help="Named preset, e.g. framer-small / framer-8b / framer-200b-a20b / framer-1t-a32b")
    parser.add_argument("--size", choices=["tiny", "small", "medium", "large"], default=None,
                        help="Legacy size alias for framer-{tiny,small,medium,large}")
    parser.add_argument("--list-presets", action="store_true", help="List presets and exit")
    parser.add_argument("--estimate", action="store_true",
                        help="Print the parameter and memory budget for the resolved config, then exit")
    parser.add_argument("--text-only", action="store_true",
                        help="Build only the LLM core (skip multimodal encoders/decoders)")
    parser.add_argument("--force", action="store_true",
                        help="Instantiate even when the config does not fit in memory")
    parser.add_argument("--shard-dir", default=None,
                        help="Directory of packed token shards (see scripts/prepare_data.py)")
    parser.add_argument("--precision", choices=["bf16", "fp16", "fp32"], default=None,
                        help="Autocast precision (default bf16)")
    parser.add_argument("--grad-checkpointing", action="store_true",
                        help="Enable activation checkpointing to save memory")

    # Reproducibility
    parser.add_argument("--seed", type=int, default=None,
                        help="RNG seed for reproducible runs (default: config.seed, usually 42)")

    # Context extension
    parser.add_argument("--rope-scaling", type=float, default=None,
                        help="RoPE scaling factor for extended context (e.g., 8.0 for 8x)")

    # Evaluation
    parser.add_argument("--benchmark-dir", default="benchmarks",
                        help="Directory containing benchmark data (wikitext-2/, humaneval/)")
    parser.add_argument("--eval-output", default=None,
                        help="Write evaluation results to JSON file")
    parser.add_argument("--eval-seq-len", type=int, default=128,
                        help="Sequence length for text benchmarks")
    parser.add_argument("--eval-batch-size", type=int, default=4,
                        help="Batch size for text benchmarks")
    parser.add_argument("--eval-code-limit", type=int, default=None,
                        help="Limit number of HumanEval problems (for fast smoke tests)")

    return parser


def _build_config_from_args(args: argparse.Namespace) -> FramerConfig:
    """Build and return a FramerConfig from parsed CLI args (no I/O, no seeding).

    Extracted from main() so tests can exercise the full override path without
    running training, building, or exporting anything.
    """
    preset_name = args.preset or args.size or "framer-medium"
    config = FramerConfig.from_preset(preset_name)

    if args.text_only:
        config.text_only = True
    if args.shard_dir:
        config.data_dir = args.shard_dir
    if args.precision:
        config.precision = args.precision
    if args.grad_checkpointing:
        config.use_gradient_checkpointing = True
    if args.d_model:
        config.d_model = args.d_model
    if args.n_layers:
        config.n_layers = args.n_layers
    if args.n_heads:
        config.n_heads = args.n_heads
    if args.max_steps:
        config.max_steps = args.max_steps
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.learning_rate = args.lr
    if args.warmup_steps:
        config.warmup_steps = args.warmup_steps
    if args.grad_accum:
        config.gradient_accumulation_steps = args.grad_accum
    if args.device:
        config.device = args.device
    if args.rope_scaling:
        config.rope_scaling_factor = args.rope_scaling
    if args.seed is not None:
        config.seed = args.seed
    if args.tokenizer_vocab_size is not None:
        config.vocab_size = args.tokenizer_vocab_size
    if args.use_ctc_head:
        config.use_ctc_head = True
    if args.ctc_loss_weight is not None:
        config.ctc_loss_weight = args.ctc_loss_weight

    config.validate()
    return config


def main():
    args = _make_parser().parse_args()

    if args.list_presets:
        print(f"  {'preset':<18s} {'text':>9s} {'active/tok':>11s} {'multimodal':>11s} {'model total':>12s}")
        for name in list_presets():
            est = estimate_params(FramerConfig.from_preset(name))
            print(
                f"  {name:<18s} {est['total_h']:>9s} {est['active_h']:>11s} "
                f"{est['multimodal_h']:>11s} {est['model_total_h']:>12s}"
            )
        return

    config = _build_config_from_args(args)

    if args.estimate:
        print_estimate(config)
        return

    # Seed Python / NumPy / PyTorch RNGs before any model is instantiated so
    # that weight initialization is reproducible regardless of which mode runs.
    apply_seed(config.seed)

    preset_name = args.preset or args.size or "framer-medium"
    logger.info("=" * 60)
    logger.info(f"FramerAI Model Builder | preset={resolve_preset_name(preset_name)}")
    logger.info("=" * 60)

    if args.mode in ("build", "all"):
        build_model(config, args.output_dir, args.data_dir, force=args.force)

    if args.mode in ("train", "all"):
        train_model(config, args.output_dir, args.resume, args.data_dir,
                    args.train_modalities, shard_dir=args.shard_dir)

    if args.mode == "sft":
        train_sft_model(config, args.output_dir, args.data_dir, args.resume)

    if args.mode == "dpo":
        train_dpo_model(config, args.output_dir, args.data_dir, args.beta, args.resume)

    if args.mode in ("export", "all"):
        export_model(config, args.output_dir, args.export_dir)

    if args.mode == "eval":
        eval_model(
            config, args.output_dir,
            benchmark_dir=args.benchmark_dir,
            eval_output=args.eval_output,
            seq_len=args.eval_seq_len,
            batch_size=args.eval_batch_size,
            code_limit=args.eval_code_limit,
        )

    logger.info("Done!")


if __name__ == "__main__":
    main()

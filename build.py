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
import os
import sys
import time
import logging
from pathlib import Path
from dataclasses import asdict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from model.configs import FramerConfig
from model.framer import FramerModel
from model.tokenizer import FramerTokenizer
from model.utils import get_device, count_parameters, save_checkpoint, load_checkpoint

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("framerai")


# ---------------------------------------------------------------------------
# Synthetic datasets for demonstration / initial training
# ---------------------------------------------------------------------------

class TextDataset(Dataset):
    """Simple text dataset for language model training."""

    def __init__(self, texts: list, tokenizer: FramerTokenizer, max_len: int = 512):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.samples = []
        for text in texts:
            tokens = tokenizer.encode(text, add_special=True)
            if len(tokens) > max_len:
                tokens = tokens[:max_len]
            else:
                tokens += [tokenizer.pad_id] * (max_len - len(tokens))
            self.samples.append(tokens)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        tokens = torch.tensor(self.samples[idx], dtype=torch.long)
        input_ids = tokens[:-1]
        labels = tokens[1:]
        return {"input_ids": input_ids, "labels": labels}


class ImageTextDataset(Dataset):
    """Synthetic image-text pairs for diffusion training."""

    def __init__(self, num_samples: int = 100, resolution: int = 256, tokenizer: FramerTokenizer = None):
        self.num_samples = num_samples
        self.resolution = resolution
        self.tokenizer = tokenizer
        self.prompts = [
            "a red circle on a white background",
            "a blue square with yellow border",
            "abstract art with vibrant colors",
            "a simple landscape with mountains",
            "geometric patterns in pastel colors",
        ] * (num_samples // 5 + 1)

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # Synthetic image (random noise shaped like an image)
        image = torch.randn(3, self.resolution, self.resolution)
        prompt = self.prompts[idx % len(self.prompts)]
        tokens = self.tokenizer.encode(prompt, add_special=True)
        tokens = tokens[:64] + [self.tokenizer.pad_id] * max(0, 64 - len(tokens))
        return {
            "image": image,
            "input_ids": torch.tensor(tokens, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Build & Train
# ---------------------------------------------------------------------------

def build_model(config: FramerConfig, output_dir: str):
    """Initialize model architecture and save initial checkpoint."""
    logger.info("Building FramerAI model...")
    logger.info(f"Config: d_model={config.d_model}, layers={config.n_layers}, heads={config.n_heads}")

    model = FramerModel(config)
    num_params = count_parameters(model)
    logger.info(f"Total trainable parameters: {num_params:,}")

    # Build tokenizer
    logger.info("Building tokenizer...")
    tokenizer = FramerTokenizer(config.vocab_size)

    # Train tokenizer on sample data
    sample_texts = [
        "Hello, how can I help you today?",
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
        "The quick brown fox jumps over the lazy dog.",
        "import torch\nimport torch.nn as nn\n\nclass Model(nn.Module):\n    pass",
        "Generate an image of a sunset over the ocean.",
        "Create a video of a cat playing with a ball.",
        "What is the meaning of life? The meaning of life is subjective.",
        "Machine learning is a subset of artificial intelligence.",
        "def quicksort(arr):\n    if len(arr) <= 1:\n        return arr\n    pivot = arr[0]\n    left = [x for x in arr[1:] if x < pivot]\n    right = [x for x in arr[1:] if x >= pivot]\n    return quicksort(left) + [pivot] + quicksort(right)",
        "Neural networks are composed of layers of interconnected nodes.",
    ]
    tokenizer.train(sample_texts * 10, target_vocab_size=min(1000, config.vocab_size))
    logger.info(f"Tokenizer vocabulary size: {config.vocab_size}")

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


def train_model(config: FramerConfig, output_dir: str, resume: str = None):
    """Train the model."""
    device = get_device(config.device)
    logger.info(f"Training on device: {device}")

    # Load or build model
    model = FramerModel(config).to(device)
    tokenizer_path = os.path.join(output_dir, "tokenizer")

    if resume:
        logger.info(f"Resuming from {resume}")
        step, prev_loss = load_checkpoint(resume, model)
        logger.info(f"Resumed at step {step}, loss={prev_loss:.4f}")
    else:
        init_path = os.path.join(output_dir, "model_init.pt")
        if os.path.exists(init_path):
            step, _ = load_checkpoint(init_path, model)
        else:
            step = 0

    tokenizer = FramerTokenizer.load(tokenizer_path) if os.path.exists(tokenizer_path) else FramerTokenizer(config.vocab_size)

    # Create datasets
    sample_texts = [
        "Hello, how can I help you today? I am FramerAI, your multimodal assistant.",
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
        "To create an image, simply describe what you want to see.",
        "import torch\nimport torch.nn as nn",
        "The transformer architecture revolutionized natural language processing.",
        "Videos are generated frame by frame using temporal diffusion.",
        "Machine learning models learn patterns from data to make predictions.",
        "def hello_world():\n    print('Hello, World!')\n\nhello_world()",
    ] * 20

    text_dataset = TextDataset(sample_texts, tokenizer, max_len=128)
    text_loader = DataLoader(text_dataset, batch_size=config.batch_size, shuffle=True)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=(0.9, 0.95),
    )

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.max_steps, eta_min=1e-6)

    # Mixed precision
    scaler = torch.amp.GradScaler("cuda") if config.mixed_precision and device.type == "cuda" else None

    logger.info(f"Starting training from step {step}")
    logger.info(f"Parameters: {count_parameters(model):,}")
    model.train()

    total_loss = 0
    log_interval = 10
    save_interval = 500
    start_time = time.time()

    while step < config.max_steps:
        for batch in text_loader:
            if step >= config.max_steps:
                break

            input_ids = batch["input_ids"].to(device)
            labels = batch["labels"].to(device)

            if scaler:
                with torch.amp.autocast("cuda"):
                    results = model(input_ids=input_ids, labels=labels)
                    loss = results["text_loss"] / config.gradient_accumulation_steps
                scaler.scale(loss).backward()
            else:
                results = model(input_ids=input_ids, labels=labels)
                loss = results["text_loss"] / config.gradient_accumulation_steps
                loss.backward()

            total_loss += loss.item()

            if (step + 1) % config.gradient_accumulation_steps == 0:
                if scaler:
                    scaler.unscale_(optimizer)
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()

                optimizer.zero_grad()
                scheduler.step()

            step += 1

            if step % log_interval == 0:
                avg_loss = total_loss / log_interval
                elapsed = time.time() - start_time
                steps_per_sec = step / elapsed if elapsed > 0 else 0
                lr = scheduler.get_last_lr()[0]
                logger.info(
                    f"Step {step}/{config.max_steps} | Loss: {avg_loss:.4f} | "
                    f"LR: {lr:.2e} | Speed: {steps_per_sec:.1f} steps/s"
                )
                total_loss = 0

            if step % save_interval == 0:
                ckpt_path = os.path.join(output_dir, f"checkpoint_{step}.pt")
                save_checkpoint(model, optimizer, step, loss.item() * config.gradient_accumulation_steps, ckpt_path)
                logger.info(f"Checkpoint saved: {ckpt_path}")

    # Final save
    final_path = os.path.join(output_dir, "model_final.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": asdict(config),
        "step": step,
    }, final_path)
    logger.info(f"Training complete. Final model saved to {final_path}")
    return model


def _load_teacher(model_name, args):
    """Load a local open-source teacher model from HuggingFace."""
    from model.distillation.teacher_models import load_teacher_simple
    return load_teacher_simple(
        model_name=model_name,
        quantize=args.quantize,
        device_map=args.teacher_device or "auto",
        dtype=args.teacher_dtype or "auto",
        hf_token=args.hf_token or os.environ.get("HF_TOKEN"),
        cache_dir=args.cache_dir,
    )


def generate_distillation_data(args):
    """Generate training data using local open-source teacher models.

    Supports --teacher-model all to cycle through every supported model,
    or a comma-separated list like llama3-8b,deepseek-r1-7b,mistral-7b.
    Each model is loaded one at a time to fit in VRAM.
    """
    from model.distillation import DistillationDataGenerator
    from model.distillation.teacher_models import resolve_model_list

    model_names = resolve_model_list(args.teacher_model)
    data_dir = args.data_dir or "data/distillation"
    domains = args.domains.split(",") if args.domains else None
    total_samples = args.num_samples or 1000
    samples_per_model = max(1, total_samples // len(model_names))

    logger.info("=" * 60)
    logger.info("Multi-Teacher Data Generation")
    logger.info(f"  Teachers: {len(model_names)} models")
    for m in model_names:
        logger.info(f"    - {m}")
    logger.info(f"  Samples per model: {samples_per_model}")
    logger.info(f"  Total target: {samples_per_model * len(model_names)}")
    logger.info("=" * 60)

    all_data_files = []
    for i, model_name in enumerate(model_names):
        logger.info(f"\n[{i + 1}/{len(model_names)}] Loading teacher: {model_name}")
        try:
            teacher = _load_teacher(model_name, args)
        except Exception as e:
            logger.error(f"  Failed to load {model_name}: {e}")
            logger.error(f"  Skipping this model and continuing...")
            continue

        # Each model gets its own subfolder so data doesn't overwrite
        model_data_dir = os.path.join(data_dir, model_name.replace("/", "_"))
        generator = DistillationDataGenerator(teacher, output_dir=model_data_dir)

        logger.info(f"  Generating {samples_per_model} samples...")
        output_path = generator.generate_dataset(
            num_samples=samples_per_model,
            domains=domains,
            max_new_tokens=args.teacher_max_tokens or 2048,
            temperature=args.teacher_temperature or 0.7,
            save_logits=args.save_logits,
        )
        all_data_files.append(output_path)

        if args.conversations:
            conv_per_model = max(1, (args.num_conversations or 100) // len(model_names))
            logger.info(f"  Generating {conv_per_model} conversations...")
            conv_path = generator.generate_conversation_dataset(
                num_conversations=conv_per_model,
            )
            all_data_files.append(conv_path)

        # Unload to free VRAM before loading the next model
        teacher.unload()
        logger.info(f"  Unloaded {model_name}")

    # Merge all data files into one combined JSONL
    combined_path = os.path.join(data_dir, "distillation_data.jsonl")
    logger.info(f"\nMerging {len(all_data_files)} data files into {combined_path}...")
    total_count = 0
    with open(combined_path, "w") as out_f:
        for data_file in all_data_files:
            if os.path.exists(data_file):
                with open(data_file) as in_f:
                    for line in in_f:
                        if line.strip():
                            out_f.write(line)
                            total_count += 1

    logger.info(f"Data generation complete: {total_count} total samples from {len(model_names)} teachers")
    logger.info(f"Combined dataset: {combined_path}")


def distill_train(config: FramerConfig, args):
    """Train FramerAI using knowledge distillation from local teacher models.

    With multiple teachers (--teacher-model all), each teacher is loaded
    for a portion of training steps, then swapped for the next one.
    This way only one teacher is in VRAM at a time.
    """
    from model.distillation import DistillationTrainer
    from model.distillation.distill_trainer import DistillationConfig
    from model.distillation.teacher_models import resolve_model_list

    model_names = resolve_model_list(args.teacher_model)

    # Build base distillation config
    total_steps = args.max_steps or config.max_steps
    distill_config = DistillationConfig(
        data_path=args.distill_data or "data/distillation/distillation_data.jsonl",
        output_dir=args.output_dir,
        max_steps=total_steps,
        batch_size=args.batch_size or config.batch_size,
        learning_rate=args.lr or 1e-4,
        weight_decay=config.weight_decay,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        temperature=config.distill_temperature,
        alpha_hard=config.distill_alpha_hard,
        alpha_soft=config.distill_alpha_soft,
        rope_scaling_factor=config.rope_scaling_factor,
        rope_scaling_type=config.rope_scaling_type,
        device=config.device,
        mixed_precision=config.mixed_precision,
    )

    if args.distill_temperature:
        distill_config.temperature = args.distill_temperature
    if args.distill_alpha_hard is not None:
        distill_config.alpha_hard = args.distill_alpha_hard
    if args.distill_alpha_soft is not None:
        distill_config.alpha_soft = args.distill_alpha_soft
    if args.alpha_feature is not None:
        distill_config.alpha_feature = args.alpha_feature
    if args.rope_scaling:
        distill_config.rope_scaling_factor = args.rope_scaling
    if args.no_progressive:
        distill_config.progressive_seq_len = False
    if args.target_seq_len:
        distill_config.target_seq_len = args.target_seq_len

    # Load student model
    model = FramerModel(config)
    tokenizer_path = os.path.join(args.output_dir, "tokenizer")

    init_path = os.path.join(args.output_dir, "model_final.pt")
    if not os.path.exists(init_path):
        init_path = os.path.join(args.output_dir, "model_init.pt")
    if os.path.exists(init_path):
        logger.info(f"Loading base model from {init_path}")
        load_checkpoint(init_path, model)

    tokenizer = FramerTokenizer.load(tokenizer_path) if os.path.exists(tokenizer_path) else FramerTokenizer(config.vocab_size)

    # Multi-teacher: split total steps across teachers, train sequentially
    # Each teacher gets an equal share of steps. The student checkpoint
    # carries over between teachers so knowledge accumulates.
    steps_per_teacher = max(1, total_steps // len(model_names))

    logger.info("=" * 60)
    logger.info("Multi-Teacher Distillation")
    logger.info(f"  Teachers: {len(model_names)} models")
    for m in model_names:
        logger.info(f"    - {m}")
    logger.info(f"  Total steps: {total_steps}")
    logger.info(f"  Steps per teacher: {steps_per_teacher}")
    logger.info("=" * 60)

    resume_path = args.resume
    completed_steps = 0

    for i, model_name in enumerate(model_names):
        remaining = total_steps - completed_steps
        if remaining <= 0:
            break

        current_steps = min(steps_per_teacher, remaining)
        logger.info(f"\n{'=' * 60}")
        logger.info(f"[Teacher {i + 1}/{len(model_names)}] {model_name}")
        logger.info(f"  Steps: {completed_steps} -> {completed_steps + current_steps}")
        logger.info(f"{'=' * 60}")

        try:
            teacher = _load_teacher(model_name, args)
        except Exception as e:
            logger.error(f"  Failed to load {model_name}: {e}")
            logger.error(f"  Skipping this teacher...")
            continue

        # Update config for this teacher's segment
        distill_config.max_steps = completed_steps + current_steps

        trainer = DistillationTrainer(model, tokenizer, teacher, distill_config)
        model = trainer.train(resume_from=resume_path)

        # After first teacher, resume from the checkpoint the trainer saved
        resume_path = os.path.join(args.output_dir, f"distill_ckpt_{completed_steps + current_steps}.pt")
        if not os.path.exists(resume_path):
            resume_path = os.path.join(args.output_dir, "distill_final.pt")

        completed_steps += current_steps

        teacher.unload()
        logger.info(f"  Unloaded {model_name}")

    # Final save with all teacher names
    from dataclasses import asdict
    final_path = os.path.join(args.output_dir, "distill_final.pt")
    torch.save({
        "model_state_dict": model.state_dict(),
        "config": asdict(config),
        "step": completed_steps,
        "teacher_models": model_names,
    }, final_path)
    logger.info(f"\nMulti-teacher distillation complete: {final_path}")
    logger.info(f"  Trained with: {', '.join(model_names)}")
    logger.info(f"  Total steps: {completed_steps}")


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
        "modalities": ["text", "code", "image", "video"],
    }
    with open(os.path.join(export_dir, "model_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    logger.info(f"Model exported to {export_dir}")
    logger.info(f"  Model: {export_path}")
    logger.info(f"  Parameters: {count_parameters(model):,}")


def main():
    parser = argparse.ArgumentParser(description="FramerAI Model Builder")
    parser.add_argument("--mode", choices=["build", "train", "export", "all", "generate-data", "distill"], default="build", help="Operation mode")
    parser.add_argument("--output-dir", default="checkpoints", help="Output directory")
    parser.add_argument("--export-dir", default=None, help="Export directory")
    parser.add_argument("--resume", default=None, help="Checkpoint to resume from")

    # Model config overrides
    parser.add_argument("--d-model", type=int, default=None, help="Model dimension")
    parser.add_argument("--n-layers", type=int, default=None, help="Number of transformer layers")
    parser.add_argument("--n-heads", type=int, default=None, help="Number of attention heads")
    parser.add_argument("--max-steps", type=int, default=None, help="Max training steps")
    parser.add_argument("--batch-size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--device", type=str, default=None, help="Device (auto, cpu, cuda, mps)")

    # Size presets
    parser.add_argument("--size", choices=["tiny", "small", "medium", "large"], default=None, help="Model size preset")

    # Teacher model args (local open-source models from HuggingFace)
    parser.add_argument("--teacher-model", type=str, default=None,
                        help="Teacher model(s): 'all', 'all-small', 'all-medium', 'all-large', "
                             "comma-separated (llama3-8b,deepseek-r1-7b,mistral-7b), "
                             "single name, or HuggingFace path")
    parser.add_argument("--quantize", type=str, default=None, choices=["4bit", "8bit"],
                        help="Quantize teacher model to reduce VRAM (4bit or 8bit)")
    parser.add_argument("--teacher-device", type=str, default=None,
                        help="Device for teacher model (auto, cpu, cuda:0)")
    parser.add_argument("--teacher-dtype", type=str, default=None,
                        help="Teacher dtype: auto, float16, bfloat16, float32")
    parser.add_argument("--hf-token", type=str, default=None,
                        help="HuggingFace token for gated models (Llama, etc.) or set HF_TOKEN env var")
    parser.add_argument("--cache-dir", type=str, default=None,
                        help="HuggingFace model cache directory")
    parser.add_argument("--teacher-temperature", type=float, default=None, help="Teacher sampling temperature")
    parser.add_argument("--teacher-max-tokens", type=int, default=None, help="Teacher max output tokens")

    # Data generation args
    parser.add_argument("--num-samples", type=int, default=None, help="Number of training samples to generate")
    parser.add_argument("--domains", type=str, default=None,
                        help="Comma-separated: general_knowledge,reasoning,code_generation,instruction_following,creative,long_context")
    parser.add_argument("--conversations", action="store_true", help="Also generate multi-turn conversation data")
    parser.add_argument("--num-conversations", type=int, default=None, help="Number of conversations to generate")
    parser.add_argument("--data-dir", type=str, default=None, help="Directory for generated data")
    parser.add_argument("--save-logits", action="store_true", help="Save teacher logits alongside text (large files)")

    # Distillation training args
    parser.add_argument("--distill-data", type=str, default=None, help="Path to distillation JSONL data")
    parser.add_argument("--distill-temperature", type=float, default=None, help="Distillation temperature")
    parser.add_argument("--distill-alpha-hard", type=float, default=None, help="Hard label loss weight")
    parser.add_argument("--distill-alpha-soft", type=float, default=None, help="Soft label loss weight")
    parser.add_argument("--alpha-feature", type=float, default=None,
                        help="Hidden state alignment loss weight (0 = off)")
    parser.add_argument("--rope-scaling", type=float, default=None,
                        help="RoPE scaling factor for extended context (e.g., 8.0 for 8x)")
    parser.add_argument("--target-seq-len", type=int, default=None,
                        help="Target sequence length (default 2048, increase with --rope-scaling)")
    parser.add_argument("--no-progressive", action="store_true", help="Disable progressive sequence length training")

    args = parser.parse_args()

    # Create config
    config = FramerConfig()

    # Apply size presets
    if args.size == "tiny":
        config.d_model = 256
        config.n_heads = 4
        config.n_layers = 6
        config.d_ff = 1024
        config.vision_d_model = 256
        config.vision_n_heads = 4
        config.vision_n_layers = 4
        config.diffusion_channels = 64
        config.max_steps = 1000
    elif args.size == "small":
        config.d_model = 512
        config.n_heads = 8
        config.n_layers = 12
        config.d_ff = 2048
        config.vision_d_model = 512
        config.vision_n_heads = 8
        config.vision_n_layers = 6
        config.diffusion_channels = 128
    elif args.size == "medium":
        pass  # defaults
    elif args.size == "large":
        config.d_model = 2048
        config.n_heads = 32
        config.n_layers = 32
        config.d_ff = 8192
        config.vision_d_model = 2048
        config.vision_n_heads = 32
        config.vision_n_layers = 16
        config.diffusion_channels = 512

    # Apply CLI overrides
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
    if args.device:
        config.device = args.device

    logger.info("=" * 60)
    logger.info("FramerAI Model Builder")
    logger.info("=" * 60)

    if args.mode in ("build", "all"):
        build_model(config, args.output_dir)

    if args.mode in ("train", "all"):
        train_model(config, args.output_dir, args.resume)

    if args.mode in ("export", "all"):
        export_model(config, args.output_dir, args.export_dir)

    if args.mode == "generate-data":
        generate_distillation_data(args)

    if args.mode == "distill":
        distill_train(config, args)

    logger.info("Done!")


if __name__ == "__main__":
    main()

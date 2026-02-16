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
    parser.add_argument("--mode", choices=["build", "train", "export", "all"], default="build", help="Operation mode")
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

    logger.info("Done!")


if __name__ == "__main__":
    main()

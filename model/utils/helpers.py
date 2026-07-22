"""Utility functions for FramerAI model."""

import os

import torch
import torch.nn as nn


def get_device(preference: str = "auto") -> torch.device:
    """Get the best available device."""
    if preference == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(preference)


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def _human(n: float) -> str:
    for unit, div in (("T", 1e12), ("B", 1e9), ("M", 1e6), ("K", 1e3)):
        if abs(n) >= div:
            return f"{n / div:.2f}{unit}"
    return str(int(n))


def estimate_params(config) -> dict:
    """Estimate the text-backbone parameter budget from a config alone.

    Returns total vs active (per-token) parameter counts *without instantiating*
    the model, so trillion-parameter presets can be sized on a laptop. Only the
    transformer backbone is counted (the multimodal encoders/decoders are small
    relative to a large LLM and are excluded here). Includes a human-readable
    ``summary`` string.
    """
    d = config.d_model
    n_layers = config.n_layers
    head_dim = d // config.n_heads
    kv_dim = config.kv_heads * head_dim

    # Attention block: q(d*d) + out(d*d) + k(d*kv) + v(d*kv)
    attn = 2 * d * d + 2 * d * kv_dim
    # Two RMSNorms per block.
    norm = 2 * d

    def dense_ffn(width):
        return 3 * d * width  # SwiGLU: w1, w2, w3

    total = 0
    active = 0
    for i in range(n_layers):
        layer_total = attn + norm
        layer_active = attn + norm
        if config.is_moe_layer(i):
            ew = config.expert_d_ff or config.d_ff
            router = d * config.n_experts
            experts = config.n_experts * dense_ffn(ew)
            shared = config.n_shared_experts * dense_ffn(ew)
            layer_total += router + experts + shared
            layer_active += router + config.n_experts_per_tok * dense_ffn(ew) + shared
        else:
            layer_total += dense_ffn(config.d_ff)
            layer_active += dense_ffn(config.d_ff)
        total += layer_total
        active += layer_active

    embed = config.vocab_size * d  # tied embedding / lm head (counted once)
    total += embed + d  # + final norm
    active += embed + d

    return {
        "total": total,
        "active": active,
        "total_h": _human(total),
        "active_h": _human(active),
        "summary": (
            f"{config.preset or 'custom'}: {_human(total)} total params, "
            f"{_human(active)} active/token"
            + (" (MoE)" if config.use_moe else " (dense)")
        ),
    }


def save_checkpoint(model: nn.Module, optimizer, step: int, loss: float, path: str):
    """Save a training checkpoint."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
        "loss": loss,
    }, path)


def load_checkpoint(path: str, model: nn.Module, optimizer=None):
    """Load a training checkpoint."""
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint.get("step", 0), checkpoint.get("loss", float("inf"))

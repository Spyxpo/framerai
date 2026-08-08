"""Optimizer construction with weight-decay parameter groups."""

import torch


def build_optimizer(model, config):
    """AdamW with decay on 2D weights only (norms/biases/embeddings excluded).

    This is the standard split used in transformer training: matrices get
    weight decay, 1-D parameters (RMSNorm gains, biases) do not.
    """
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim >= 2 and "embed" not in name:
            decay.append(param)
        else:
            no_decay.append(param)

    groups = [
        {"params": decay, "weight_decay": config.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]
    return torch.optim.AdamW(groups, lr=config.learning_rate, betas=(0.9, 0.95), eps=1e-8)

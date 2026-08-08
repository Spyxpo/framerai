"""Text metrics: perplexity, token accuracy, and bits per byte."""

import math

import torch
import torch.nn.functional as F


@torch.no_grad()
def perplexity(model, batches, device="cpu") -> float:
    """Token-level perplexity over an iterable of ``(input_ids, labels)``.

    Averaged over tokens rather than batches, so a short final batch does not
    count as much as a full one.
    """
    model.eval()
    total_loss, total_tokens = 0.0, 0

    for input_ids, labels in batches:
        input_ids, labels = input_ids.to(device), labels.to(device)
        logits = model(input_ids=input_ids)["logits"]
        loss = F.cross_entropy(
            logits.reshape(-1, logits.size(-1)), labels.reshape(-1),
            ignore_index=-100, reduction="sum",
        )
        counted = int((labels != -100).sum())
        total_loss += float(loss)
        total_tokens += counted

    if total_tokens == 0:
        return float("inf")
    return math.exp(min(total_loss / total_tokens, 700))  # exp overflows past ~709


@torch.no_grad()
def token_accuracy(model, batches, device="cpu") -> float:
    """Fraction of positions where the argmax matches the label."""
    model.eval()
    correct, total = 0, 0
    for input_ids, labels in batches:
        input_ids, labels = input_ids.to(device), labels.to(device)
        predictions = model(input_ids=input_ids)["logits"].argmax(dim=-1)
        mask = labels != -100
        correct += int((predictions[mask] == labels[mask]).sum())
        total += int(mask.sum())
    return correct / total if total else 0.0


def bits_per_byte(loss_nats: float, tokens: int, bytes_seen: int) -> float:
    """Convert a mean per-token loss into bits per byte.

    Tokenizer-independent, so two models with different vocabularies can be
    compared on the same corpus - which raw perplexity cannot do.
    """
    if bytes_seen == 0:
        return float("inf")
    return loss_nats * tokens / (bytes_seen * math.log(2))

#!/usr/bin/env python3
"""Measure serving throughput, for the pricing basis.

Usage:
    python scripts/measure_throughput.py --checkpoint checkpoints/export/framerai_model.pt \
        --tokenizer checkpoints/export/tokenizer --device cuda

Prints a ``throughput`` block for backend/config/pricing.json. Rates there are
GPU cost divided by these numbers, so run this on the hardware that serves and
paste the result in; prices then follow the deployment rather than a guess.

Decode is measured at batch size one, which is the pessimistic case: a server
that batches concurrent requests decodes more tokens per GPU-second.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from model.generate import FramerGenerator
from model.quantize import quantize_model


def _sync(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


@torch.no_grad()
def measure_prefill(gen, prompt_tokens, repeats):
    ids = torch.randint(10, gen.model.config.vocab_size, (1, prompt_tokens), device=gen.device)
    gen._prefill(ids)  # warm-up
    _sync(gen.device)
    start = time.perf_counter()
    for _ in range(repeats):
        gen._prefill(ids)
    _sync(gen.device)
    return prompt_tokens * repeats / (time.perf_counter() - start)


@torch.no_grad()
def measure_decode(gen, new_tokens):
    ids = list(range(10, 42))
    gen.generate_text(ids, max_new_tokens=8, temperature=0)  # warm-up
    _sync(gen.device)
    start = time.perf_counter()
    gen.generate_text(ids, max_new_tokens=new_tokens, temperature=0)
    _sync(gen.device)
    produced = max(1, gen.last_output_tokens)
    return produced / (time.perf_counter() - start)


def main():
    parser = argparse.ArgumentParser(description="Measure throughput for pricing")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--prompt-tokens", type=int, default=1024)
    parser.add_argument("--new-tokens", type=int, default=128)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--weight-dtype", default="auto", choices=("auto", "int8"))
    args = parser.parse_args()

    gen = FramerGenerator.from_checkpoint(args.checkpoint, args.tokenizer, device=args.device)
    quantize_model(gen.model, args.weight_dtype)
    window = int(getattr(gen.model.config, "max_seq_len", 0) or 0)
    prompt_tokens = min(args.prompt_tokens, window - args.new_tokens - 2) if window else args.prompt_tokens

    prefill = measure_prefill(gen, prompt_tokens, args.repeats)
    decode = measure_decode(gen, args.new_tokens)

    entry = {
        "prefill_tokens_per_second": round(prefill),
        "decode_tokens_per_second": round(decode),
        "cached_input_discount": 0.9,
    }
    print(json.dumps({"framerai-text": entry, "framerai-code": entry}, indent=2))


if __name__ == "__main__":
    main()

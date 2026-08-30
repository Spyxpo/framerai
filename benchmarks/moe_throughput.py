"""
Benchmark MoE throughput: loop-based vs grouped-GEMM dispatch.

Measures tokens/sec on a MoE preset before and after the grouped-GEMM optimization
(Issue #157). Compares the original per-expert loop against the vectorized einsum path.

Usage:
    python benchmarks/moe_throughput.py --preset framer-tiny-moe --device cuda:0
    python benchmarks/moe_throughput.py --preset framer-tiny-moe --device cpu
"""

import argparse
import time

import torch

from model.configs import FramerConfig
from model.modules.moe import MoEFeedForward


def benchmark_moe_forward(ffn, x, n_warmup=5, n_iter=20):
    """
    Benchmark forward pass throughput.

    Returns:
        tokens_per_sec: float
        latency_ms: float (mean per iteration)
    """
    device = x.device

    # Warmup
    with torch.no_grad():
        for _ in range(n_warmup):
            _ = ffn(x)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    # Benchmark
    start = time.perf_counter()
    with torch.no_grad():
        for _ in range(n_iter):
            _ = ffn(x)

    if device.type == "cuda":
        torch.cuda.synchronize(device)

    elapsed = time.perf_counter() - start

    total_tokens = x.shape[0] * x.shape[1] * n_iter  # B * T * n_iter
    tokens_per_sec = total_tokens / elapsed
    latency_ms = (elapsed / n_iter) * 1000

    return tokens_per_sec, latency_ms


def main():
    parser = argparse.ArgumentParser(description="Benchmark MoE throughput: loop vs grouped dispatch")
    parser.add_argument(
        "--preset", default="framer-tiny-moe", help="Model preset with MoE layers (default: framer-tiny-moe)"
    )
    parser.add_argument("--device", default="cuda:0", help="Device: cuda:0, cpu, etc. (default: cuda:0)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size (default: 4)")
    parser.add_argument("--seq-len", type=int, default=128, help="Sequence length (default: 128)")
    parser.add_argument("--n-warmup", type=int, default=5, help="Warmup iterations (default: 5)")
    parser.add_argument("--n-iter", type=int, default=20, help="Benchmark iterations (default: 20)")
    args = parser.parse_args()

    # Load config
    try:
        config = FramerConfig.from_preset(args.preset)
    except ValueError:
        print(f"Error: Unknown preset '{args.preset}'. Use --list-presets to see available presets.")
        return 1

    if not config.use_moe:
        print(f"Error: {args.preset} does not use MoE")
        return 1

    device = torch.device(args.device)

    # Create single MoE layer
    ffn = MoEFeedForward(
        d_model=config.d_model,
        expert_d_ff=config.expert_d_ff or config.d_ff,
        n_experts=config.n_experts,
        n_experts_per_tok=config.n_experts_per_tok,
        n_shared_experts=config.n_shared_experts,
        dropout=0.0,
    )
    ffn = ffn.to(device).eval()

    # Create input
    x = torch.randn(args.batch_size, args.seq_len, config.d_model, device=device)

    print("\nBenchmarking MoE Throughput (Issue #157)")
    print("=" * 60)
    print(f"Preset:       {args.preset}")
    print(f"Device:       {device}")
    print(f"Experts:      {config.n_experts} (top-{config.n_experts_per_tok})")
    print(f"Batch:        {args.batch_size} × {args.seq_len} = {args.batch_size * args.seq_len} tokens")
    print(f"Model dim:    {config.d_model}")
    print(f"Expert dim:   {config.expert_d_ff or config.d_ff}")
    print(f"Warmup:       {args.n_warmup} iterations")
    print(f"Benchmark:    {args.n_iter} iterations")
    print("=" * 60)

    # Benchmark loop-based dispatch
    print("\n[1/2] Benchmarking loop-based dispatch...")
    original_method = ffn._should_use_grouped_dispatch
    ffn._should_use_grouped_dispatch = lambda x: False
    loop_tps, loop_lat = benchmark_moe_forward(ffn, x, args.n_warmup, args.n_iter)
    print(f"      Tokens/sec: {loop_tps:>12,.1f}")
    print(f"      Latency:    {loop_lat:>12.2f} ms")

    # Benchmark grouped dispatch
    print("\n[2/2] Benchmarking grouped dispatch...")
    ffn._should_use_grouped_dispatch = lambda x: True
    grouped_tps, grouped_lat = benchmark_moe_forward(ffn, x, args.n_warmup, args.n_iter)
    print(f"      Tokens/sec: {grouped_tps:>12,.1f}")
    print(f"      Latency:    {grouped_lat:>12.2f} ms")

    # Restore
    ffn._should_use_grouped_dispatch = original_method

    # Results
    speedup = grouped_tps / loop_tps
    print("\n" + "=" * 60)
    print(f"{'Method':<25} {'Tokens/sec':<20} {'Latency (ms)'}")
    print("=" * 60)
    print(f"{'Loop dispatch':<25} {loop_tps:>15,.1f}     {loop_lat:>10.2f}")
    print(f"{'Grouped dispatch':<25} {grouped_tps:>15,.1f}     {grouped_lat:>10.2f}")
    print("=" * 60)
    print(f"Speedup: {speedup:.2f}x")
    print()

    return 0


if __name__ == "__main__":
    exit(main())


#!/usr/bin/env python3
"""Tokenize a local corpus into packed token shards for scalable training.

Usage:
    python -m scripts.prepare_data --data-dir data --tokenizer checkpoints/tokenizer \
        --out-dir data/shards --shard-tokens 1000000

The resulting shard directory can be fed to training via ``--shard-dir`` (see
build.py), which streams packed sequences with memory-mapped shards instead of
tokenizing the whole corpus into RAM.
"""

import argparse
import os
import sys

# Allow running as a plain script (python scripts/prepare_data.py) too.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.data import iter_text_records, prepare_shards
from model.tokenizer import FramerTokenizer


def main():
    parser = argparse.ArgumentParser(description="Prepare packed token shards")
    parser.add_argument("--data-dir", default="data", help="Corpus directory (.txt / .jsonl)")
    parser.add_argument("--tokenizer", required=True, help="Path to a trained tokenizer directory")
    parser.add_argument("--out-dir", default="data/shards", help="Output shard directory")
    parser.add_argument("--shard-tokens", type=int, default=1_000_000, help="Tokens per shard")
    parser.add_argument("--vocab-size", type=int, default=50304,
                        help="Fallback vocab size if the tokenizer dir is absent")
    args = parser.parse_args()

    if os.path.isdir(args.tokenizer):
        tokenizer = FramerTokenizer.load(args.tokenizer)
    else:
        print(f"[warn] tokenizer '{args.tokenizer}' not found; training a fresh one on the corpus.")
        tokenizer = FramerTokenizer(args.vocab_size)
        corpus = list(iter_text_records(args.data_dir))
        tokenizer.train(corpus, target_vocab_size=min(1000, args.vocab_size))

    meta = prepare_shards(args.data_dir, tokenizer, args.out_dir, shard_tokens=args.shard_tokens)
    print(f"Wrote {len(meta['shards'])} shard(s), {meta['total_tokens']:,} tokens "
          f"({meta['dtype']}) to {args.out_dir}")


if __name__ == "__main__":
    main()

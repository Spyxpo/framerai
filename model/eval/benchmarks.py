"""Standard text and code benchmark adapters.

The adapters consume benchmark files supplied by the caller. They do not
download benchmark data implicitly, which keeps evaluation reproducible.
"""

import json
import os
import random
import subprocess
import sys
from dataclasses import dataclass

import torch

from .text import perplexity, token_accuracy


@dataclass(frozen=True)
class BenchmarkResult:
    benchmark: str
    metrics: dict
    samples: int

    def to_dict(self) -> dict:
        return {
            "benchmark": self.benchmark,
            "metrics": self.metrics,
            "samples": self.samples,
        }


def _load_text(path: str) -> str:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"benchmark file not found: {path}")

    with open(path, encoding="utf-8") as handle:
        text = handle.read().strip()

    if not text:
        raise ValueError(f"benchmark file is empty: {path}")

    return text


def make_text_batches(
    text: str,
    tokenizer,
    seq_len: int = 128,
    batch_size: int = 4,
    device: str = "cpu",
):
    """Turn a text corpus into deterministic next-token evaluation batches."""
    token_ids = tokenizer.encode(text, add_special=False)

    if len(token_ids) < 2:
        raise ValueError("text benchmark needs at least two tokens")

    batches = []

    usable = ((len(token_ids) - 1) // seq_len) * seq_len

    for start in range(0, usable, seq_len):
        chunk = token_ids[start:start + seq_len + 1]

        if len(chunk) < seq_len + 1:
            continue

        input_ids = torch.tensor(
            chunk[:-1],
            dtype=torch.long,
            device=device,
        )
        labels = torch.tensor(
            chunk[1:],
            dtype=torch.long,
            device=device,
        )

        batches.append((input_ids, labels))

    grouped = []
    for start in range(0, len(batches), batch_size):
        group = batches[start:start + batch_size]

        if not group:
            continue

        inputs = torch.stack([item[0] for item in group])
        labels = torch.stack([item[1] for item in group])
        grouped.append((inputs, labels))

    if not grouped:
        raise ValueError("text benchmark did not produce any complete sequences")

    return grouped


def evaluate_text_benchmark(
    model,
    tokenizer,
    path: str,
    device: str = "cpu",
    seq_len: int = 128,
    batch_size: int = 4,
) -> BenchmarkResult:
    """Evaluate a standard text corpus using perplexity and token accuracy."""
    text = _load_text(path)

    batches = make_text_batches(
        text,
        tokenizer,
        seq_len=seq_len,
        batch_size=batch_size,
        device=device,
    )

    return BenchmarkResult(
        benchmark="wikitext-2",
        metrics={
            "perplexity": perplexity(model, batches, device),
            "token_accuracy": token_accuracy(model, batches, device),
        },
        samples=len(batches),
    )


def _load_humaneval(path: str) -> list[dict]:
    if not os.path.isfile(path):
        raise FileNotFoundError(f"benchmark file not found: {path}")

    cases = []

    with open(path, encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue

            try:
                case = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSON on line {line_number}: {error}"
                ) from error

            required = {"task_id", "prompt", "test"}
            missing = required - case.keys()

            if missing:
                raise ValueError(
                    f"HumanEval case {line_number} missing: "
                    f"{', '.join(sorted(missing))}"
                )

            cases.append(case)

    if not cases:
        raise ValueError("HumanEval benchmark is empty")

    return cases


def _run_humaneval_test(code: str, test: str, timeout: int = 5) -> bool:
    """Run one benchmark test in a separate Python process."""
    program = f"{code}\n\n{test}\n"

    try:
        result = subprocess.run(
            [sys.executable, "-I", "-c", program],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            timeout=timeout,
            check=False,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False

    return result.returncode == 0


def evaluate_code_benchmark(
    generator,
    path: str,
    seed: int = 0,
    max_new_tokens: int = 256,
    limit: int = None,
) -> BenchmarkResult:
    """Evaluate HumanEval-style problems using deterministic pass@1."""
    cases = _load_humaneval(path)

    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        cases = cases[:limit]

    random.seed(seed)
    torch.manual_seed(seed)

    passed = 0

    for case in cases:
        completion = generator.generate_text(
            case["prompt"],
            max_new_tokens=max_new_tokens,
            temperature=0.0,
            top_k=1,
            top_p=1.0,
        )

        # generate_text returns the prompt plus generated continuation.
        if completion.startswith(case["prompt"]):
            completion = completion[len(case["prompt"]):]

        code = case["prompt"] + completion

        if _run_humaneval_test(code, case["test"]):
            passed += 1

    return BenchmarkResult(
        benchmark="humaneval",
        metrics={
            "pass@1": passed / len(cases),
        },
        samples=len(cases),
    )


def evaluate_standard_benchmarks(
    model,
    tokenizer,
    generator,
    benchmark_dir: str,
    device: str = "cpu",
    seed: int = 0,
    seq_len: int = 128,
    batch_size: int = 4,
    code_limit: int = None,
) -> list[BenchmarkResult]:
    """Run the standard text and code benchmark adapters."""
    results = []

    text_path = os.path.join(benchmark_dir, "wikitext-2", "test.txt")
    code_path = os.path.join(benchmark_dir, "humaneval", "HumanEval.jsonl")

    results.append(
        evaluate_text_benchmark(
            model,
            tokenizer,
            text_path,
            device=device,
            seq_len=seq_len,
            batch_size=batch_size,
        )
    )

    results.append(
        evaluate_code_benchmark(
            generator,
            code_path,
            seed=seed,
            limit=code_limit,
        )
    )

    return results

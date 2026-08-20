# Benchmark Data

This directory contains standard benchmark data for evaluating FramerAI model quality.

## Structure

```
benchmarks/
├── samples/               # SMOKE TEST SAMPLES ONLY - NOT REAL BENCHMARKS
│   ├── wikitext-2/
│   │   └── test.txt      # 3-paragraph sample (NOT the full benchmark)
│   └── humaneval/
│       └── HumanEval.jsonl  # 3-problem sample (NOT the full 164-problem set)
├── wikitext-2/
│   └── test.txt          # Real WikiText-2 benchmark (place here after download)
└── humaneval/
    └── HumanEval.jsonl   # Real HumanEval benchmark (place here after download)
```

## ⚠️ CRITICAL: Sample Files vs. Real Benchmarks

The `samples/` directory contains **minimal stub files for smoke testing only**:
- `samples/wikitext-2/test.txt`: 3 paragraphs (~200 tokens)
- `samples/humaneval/HumanEval.jsonl`: 3 programming problems

**These are NOT the real benchmark datasets.** They exist solely to:
1. Verify the evaluation pipeline works
2. Run fast smoke tests in CI
3. Test the CLI without downloading large datasets

**Do NOT report or compare results from sample files.** Sample-based results are meaningless for model quality assessment.

## Obtaining Real Benchmark Data

Real benchmark data is **not included** in the repository. You must download it separately.

### WikiText-2

WikiText-2 is a language modeling benchmark based on Wikipedia articles.

1. Download from: https://huggingface.co/datasets/wikitext
2. Extract the `test.txt` file
3. Place it at: `benchmarks/wikitext-2/test.txt`

The file should be plain text with articles separated by blank lines.

### HumanEval

HumanEval is a code generation benchmark with 164 Python programming problems.

1. Download from: https://github.com/openai/human-eval
2. Place `HumanEval.jsonl` at: `benchmarks/humaneval/HumanEval.jsonl`

Each line in the JSONL file must contain:
- `task_id`: Unique problem identifier (e.g., "HumanEval/0")
- `prompt`: Function signature and docstring
- `test`: Unit tests to validate the generated code

## Running Evaluation

### With Real Benchmark Data

Once you have downloaded and placed the real benchmark data:

```bash
# Run all benchmarks (uses benchmarks/wikitext-2/ and benchmarks/humaneval/)
python build.py --mode eval --size tiny --benchmark-dir benchmarks

# Save results to JSON
python build.py --mode eval --size tiny --eval-output results.json
```

### With Sample Data (Smoke Test Only)

To verify the evaluation pipeline works without downloading real data:

```bash
# Run on sample data (uses benchmarks/samples/wikitext-2/ and benchmarks/samples/humaneval/)
python build.py --mode eval --size tiny --benchmark-dir benchmarks/samples --eval-code-limit 3

# Results will clearly show sample count (e.g., "samples": 3 for HumanEval)
```

**Remember:** Sample results are for pipeline testing only. They do not represent model quality.

## Reproducibility

Evaluation is deterministic:
- Fixed seed (default 42, configurable via `--seed`)
- Deterministic batch construction
- Explicit benchmark ordering

The same checkpoint and data will produce identical scores across runs.

## Security Note

Code evaluation executes generated code in a separate Python subprocess with a timeout. This
provides basic isolation but **does not sandbox** filesystem access or network calls.

**Do not run untrusted benchmarks** on systems with sensitive data.

## Missing Data Handling

If benchmark files are not found, the evaluation harness will:
1. Report the suite as **skipped** with a clear error message
2. Continue with other benchmarks that have data available
3. Exit with an error if all suites were skipped

This ensures missing data never produces fake results.

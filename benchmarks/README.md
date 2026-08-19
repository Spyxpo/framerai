# Benchmark Data

This directory contains standard benchmark data for evaluating FramerAI model quality.

## Structure

```
benchmarks/
├── wikitext-2/
│   └── test.txt         # Plain text corpus for perplexity and token accuracy
└── humaneval/
    └── HumanEval.jsonl  # Code problems with prompts and unit tests
```

## Obtaining Benchmark Data

Benchmark data is **not included** in the repository. You must obtain it separately.

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

Once benchmark data is in place:

```bash
# Run all benchmarks
python build.py --mode eval --size tiny --benchmark-dir benchmarks

# Save results to JSON
python build.py --mode eval --size tiny --eval-output results.json

# Run a subset of HumanEval (for quick smoke tests)
python build.py --mode eval --size tiny --eval-code-limit 10
```

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

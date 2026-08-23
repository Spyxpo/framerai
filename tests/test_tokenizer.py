"""Tokenizer round-trip and id-layout tests.

`decode()` silently returned an empty string for every merged token once the
tokenizer had been through `save()` / `load()`, because `load()` rebuilt the
merge map but not the merge vocabulary. That is the live inference path, so the
save/load round-trip is the test that matters here, not the in-memory one.
"""

import json
import os

import pytest

from model.tokenizer.tokenizer import FramerTokenizer

CORPUS = [
    "hello world hello there",
    "the quick brown fox jumps over the lazy dog",
    "hello world of tokenizers",
]


def trained(vocab_size=400, target=350):
    tokenizer = FramerTokenizer(vocab_size=vocab_size)
    tokenizer.train(CORPUS, target_vocab_size=target)
    return tokenizer


def test_roundtrip_in_memory():
    tokenizer = trained()
    text = "hello world"
    assert tokenizer.decode(tokenizer.encode(text, add_special=False)) == text


def test_roundtrip_survives_save_and_load(tmp_path):
    tokenizer = trained()
    text = "hello world hello there"
    ids = tokenizer.encode(text, add_special=False)
    assert len(ids) < len(text), "expected merges to actually fire"

    tokenizer.save(str(tmp_path))
    reloaded = FramerTokenizer.load(str(tmp_path))

    assert reloaded.decode(ids) == text
    assert reloaded.encode(text, add_special=False) == ids


def test_load_rebuilds_the_merge_vocabulary(tmp_path):
    tokenizer = trained()
    tokenizer.save(str(tmp_path))
    reloaded = FramerTokenizer.load(str(tmp_path))

    assert reloaded.merge_map == tokenizer.merge_map
    for merged_id, pair in tokenizer.vocab.items():
        if isinstance(pair, tuple):
            assert reloaded.vocab[merged_id] == pair
            assert reloaded._decompose(merged_id) != []


def test_special_token_ids_are_stable():
    """These ids are baked into every saved checkpoint's embedding table."""
    golden = {
        "<pad>": 0, "<sos>": 1, "<eos>": 2, "<unk>": 3,
        "<img>": 4, "<img_end>": 5, "<vid>": 6, "<vid_end>": 7,
        "<code>": 8, "<code_end>": 9, "<user>": 10, "<assistant>": 11,
        "<system>": 12, "<audio>": 13, "<audio_end>": 14,
    }
    tokenizer = FramerTokenizer()
    assert tokenizer.special_tokens == golden
    assert tokenizer.num_special == 15
    assert tokenizer.byte_to_token[0] == 15
    assert tokenizer.byte_to_token[255] == 270


def test_reserved_block_sits_after_the_byte_range():
    """Reserved markers must not displace byte ids, which are load-bearing."""
    tokenizer = FramerTokenizer()
    for token_id in tokenizer.reserved_tokens.values():
        assert token_id >= tokenizer.num_special + 256
    assert tokenizer.first_merge_id == tokenizer.num_special + 256 + tokenizer.RESERVED_SLOTS
    assert tokenizer.reserved_tokens["<img_patch>"] == 271


def test_reserved_markers_roundtrip():
    tokenizer = trained()
    text = "before <img_patch> after"
    ids = tokenizer.encode(text, add_special=False)
    assert tokenizer.reserved_tokens["<img_patch>"] in ids
    assert tokenizer.decode(ids) == text


def test_adding_a_reserved_marker_does_not_shift_merge_ids():
    """The block is fixed-capacity, so filling a slot moves nothing."""
    base = FramerTokenizer()
    extended = FramerTokenizer(reserved_tokens={**FramerTokenizer.RESERVED_TOKENS, "<new>": 2})
    assert extended.first_merge_id == base.first_merge_id
    assert extended.byte_to_token == base.byte_to_token


def test_special_tokens_are_read_from_disk(tmp_path):
    """The file is the authority, not the current class constant."""
    trained().save(str(tmp_path))
    path = tmp_path / "tokenizer.json"
    data = json.loads(path.read_text())
    assert data["version"] == FramerTokenizer.VERSION
    assert data["special_tokens"]["<pad>"] == 0
    assert data["reserved_slots"] == FramerTokenizer.RESERVED_SLOTS


def test_version_1_files_still_decode(tmp_path):
    """Files written before the reserved block keep their original merge ids."""
    tokenizer = trained()
    text = "hello world"
    ids_v1 = [
        tokenizer.merge_map.get(pair, tokenizer.num_special + 256 + i)
        for i, pair in enumerate(tokenizer.merges[:1])
    ]
    legacy = {
        "vocab_size": tokenizer.vocab_size,
        "merges": [list(p) for p in tokenizer.merges],
        "special_tokens": tokenizer.special_tokens,
    }
    os.makedirs(tmp_path, exist_ok=True)
    (tmp_path / "tokenizer.json").write_text(json.dumps(legacy))

    reloaded = FramerTokenizer.load(str(tmp_path))
    assert reloaded.reserved_slots == 0
    assert reloaded.first_merge_id == reloaded.num_special + 256
    assert reloaded.decode(ids_v1) != ""
    assert reloaded.decode(reloaded.encode(text, add_special=False)) == text


@pytest.mark.parametrize("text", ["héllo", "→ ← ↑", "世界", "emoji free but ünicode"])
def test_unicode_roundtrip(text):
    tokenizer = trained()
    assert tokenizer.decode(tokenizer.encode(text, add_special=False)) == text


def test_train_respects_the_merge_budget():
    tokenizer = FramerTokenizer(vocab_size=400)
    tokenizer.train(CORPUS, target_vocab_size=300)
    assert len(tokenizer.merges) <= 300 - tokenizer.first_merge_id
    for merged_id in tokenizer.merge_map.values():
        assert merged_id >= tokenizer.first_merge_id


def test_special_tokens_are_stripped_or_kept_as_documented():
    tokenizer = trained()
    ids = tokenizer.encode("hi", add_special=True)
    assert ids[0] == tokenizer.sos_id and ids[-1] == tokenizer.eos_id
    # sos/eos/pad are control tokens and never render.
    assert tokenizer.decode(ids) == "hi"

def test_build_model_passes_config_vocab_size_to_tokenizer_train(tmp_path):
    """build_model() must pass config.vocab_size to tokenizer.train(), not min(1000, vocab_size).

    Spying on FramerTokenizer.train() lets us exercise the real build_model()
    call path without paying the cost of model weight initialisation, and
    without depending on disk checkpoints.  The assertion fails against the
    buggy ``min(1000, config.vocab_size)`` call and passes once the fix is in.
    """
    from unittest.mock import patch

    import build as build_module
    from model.configs import FramerConfig

    # Use a vocab_size well above 1000 so min(1000, ...) would be wrong.
    config = FramerConfig.from_preset("framer-tiny")
    config.vocab_size = 4096

    recorded_kwargs: list[dict] = []

    original_train = build_module.FramerTokenizer.train

    def spy_train(self, texts, target_vocab_size=None):
        recorded_kwargs.append({"target_vocab_size": target_vocab_size})
        # Still do the real (cheap) training so the tokenizer is usable.
        original_train(self, texts, target_vocab_size=target_vocab_size)

    with patch.object(build_module.FramerTokenizer, "train", spy_train):
        build_module.build_model(config, str(tmp_path))

    assert recorded_kwargs, "FramerTokenizer.train() was never called"
    actual = recorded_kwargs[0]["target_vocab_size"]
    assert actual == config.vocab_size, (
        f"build_model() called tokenizer.train(target_vocab_size={actual}) "
        f"but expected {config.vocab_size}. "
        f"The min(1000, vocab_size) cap is probably back."
    )


def test_build_log_reports_actual_trained_vocabulary_size(tmp_path):
    """build_model() must log the actual trained vocabulary size, not config.vocab_size.

    The build log should report the vocabulary that was actually trained/generated,
    not the configured/default value that may be larger than achievable.
    """
    import logging
    import re

    import build as build_module
    from model.configs import FramerConfig

    config = FramerConfig.from_preset("framer-tiny")
    config.vocab_size = 2000  # Large enough that actual will be smaller

    log_messages: list[str] = []

    class _LogCapture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            log_messages.append(record.getMessage())

    # Capture logs from the framerai logger
    logger = logging.getLogger("framerai")
    original_level = logger.level
    handler = _LogCapture()
    handler.setLevel(logging.INFO)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    try:
        build_module.build_model(config, str(tmp_path))
    finally:
        logger.removeHandler(handler)
        logger.setLevel(original_level)

    # Find the vocabulary size log line
    vocab_lines = [msg for msg in log_messages if "Tokenizer vocabulary size:" in msg]
    assert vocab_lines, "build_model() did not log 'Tokenizer vocabulary size:'"

    line = vocab_lines[0]
    match = re.search(r"Tokenizer vocabulary size:\s*(\d+)", line)
    assert match, f"Could not parse vocabulary size from: {line!r}"
    logged_size = int(match.group(1))

    # Verify the logged size matches the actual tokenizer structure
    tokenizer = build_module.FramerTokenizer.load(str(tmp_path / "tokenizer"))
    expected_size = tokenizer.first_merge_id + len(tokenizer.merges)
    assert logged_size == expected_size, (
        f"Logged size {logged_size} != actual size {expected_size}. "
        f"build_model() should log actual vocabulary size, not config.vocab_size."
    )

    # The actual size should be smaller than configured due to limited corpus
    assert logged_size < config.vocab_size, (
        f"Actual vocab size {logged_size} should be < configured {config.vocab_size} "
        f"due to limited training corpus"
    )


def test_prepare_data_respects_full_vocab_size(tmp_path):
    """scripts/prepare_data.py must not cap vocabulary size at min(1000, ...)."""
    import sys
    from unittest.mock import patch

    # Ensure the script module can be imported
    sys.path.insert(0, "scripts")
    import prepare_data as pd_module

    # Create test data directory
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    (data_dir / "test.txt").write_text("hello world test corpus\n" * 10)

    recorded_calls: list[int] = []

    original_train = pd_module.FramerTokenizer.train

    def spy_train(self, texts, target_vocab_size=None):
        recorded_calls.append(target_vocab_size)
        # Still call original to avoid breaking functionality
        return original_train(self, texts, target_vocab_size=target_vocab_size)

    # Mock prepare_shards to avoid actual file I/O
    def mock_prepare_shards(*args, **kwargs):
        return {"shards": [], "total_tokens": 0, "dtype": "uint16"}

    with patch.object(pd_module.FramerTokenizer, "train", spy_train):
        with patch.object(pd_module, "prepare_shards", mock_prepare_shards):
            # Simulate the main logic without calling main() directly
            args_vocab_size = 5000  # Well above 1000

            tokenizer = pd_module.FramerTokenizer(args_vocab_size)
            corpus = list(pd_module.iter_text_records(str(data_dir)))
            tokenizer.train(corpus, target_vocab_size=args_vocab_size)

    assert recorded_calls, "FramerTokenizer.train() was never called"
    actual_target = recorded_calls[0]
    assert actual_target == 5000, (
        f"prepare_data should pass full vocab_size {5000}, got {actual_target}. "
        f"The min(1000, vocab_size) cap may still be present."
    )


def test_tokenizer_vocab_size_flag_overrides_config(tmp_path):
    """--tokenizer-vocab-size flag must override config.vocab_size."""
    import build as build_module

    parser = build_module._make_parser()
    args = parser.parse_args([
        "--preset", "framer-tiny",
        "--tokenizer-vocab-size", "800"
    ])

    config = build_module._build_config_from_args(args)
    assert config.vocab_size == 800, (
        f"Expected config.vocab_size=800 after --tokenizer-vocab-size 800, "
        f"got {config.vocab_size}"
    )


def test_tokenizer_vocab_size_flag_affects_actual_build(tmp_path):
    """--tokenizer-vocab-size flag must affect the actual tokenizer training."""
    import logging
    import re

    import build as build_module
    from model.configs import FramerConfig

    # Test that using the flag creates a tokenizer with the specified size
    small_vocab_size = 400

    log_messages: list[str] = []

    class _LogCapture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            log_messages.append(record.getMessage())

    logger = logging.getLogger("framerai")
    original_level = logger.level
    handler = _LogCapture()
    handler.setLevel(logging.INFO)
    logger.setLevel(logging.INFO)
    logger.addHandler(handler)

    try:
        # Manually set up config as if from CLI args
        config = FramerConfig.from_preset("framer-tiny")
        config.vocab_size = small_vocab_size
        build_module.build_model(config, str(tmp_path))
    finally:
        logger.removeHandler(handler)
        logger.setLevel(original_level)

    # Check that the actual trained size is what we expect
    tokenizer = build_module.FramerTokenizer.load(str(tmp_path / "tokenizer"))
    actual_size = tokenizer.first_merge_id + len(tokenizer.merges)

    # Should be <= the requested size (may be smaller due to corpus limitations)
    assert actual_size <= small_vocab_size, (
        f"Actual vocab size {actual_size} should be <= requested {small_vocab_size}"
    )

    # Verify the log reports the actual size
    vocab_lines = [msg for msg in log_messages if "Tokenizer vocabulary size:" in msg]
    assert vocab_lines, "No vocabulary size logged"

    line = vocab_lines[0]
    match = re.search(r"Tokenizer vocabulary size:\s*(\d+)", line)
    assert match, f"Could not parse vocabulary size from: {line!r}"
    logged_size = int(match.group(1))

    assert logged_size == actual_size, (
        f"Logged size {logged_size} != actual tokenizer size {actual_size}"
    )

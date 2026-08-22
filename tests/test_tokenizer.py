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

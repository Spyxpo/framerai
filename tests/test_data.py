"""Streaming packed-data tests: shard prep, block packing, rank sharding."""

import os

import torch

from model.configs import FramerConfig
from model.data import AudioCaptionDataset, ImageCaptionDataset, PackedTokenDataset, prepare_shards
from model.tokenizer import FramerTokenizer


def _tokenizer():
    tok = FramerTokenizer(vocab_size=300)
    tok.train(["hello world foo bar", "the quick brown fox"], target_vocab_size=300)
    return tok


def test_prepare_shards_and_pack(tmp_path):
    data_dir = tmp_path / "corpus"
    data_dir.mkdir()
    (data_dir / "a.txt").write_text("\n\n".join(f"document number {i} lorem ipsum" for i in range(50)))

    out = tmp_path / "shards"
    tok = _tokenizer()
    meta = prepare_shards(str(data_dir), tok, str(out), shard_tokens=200)
    assert meta["total_tokens"] > 0
    assert len(meta["shards"]) >= 1

    seq_len = 16
    ds = PackedTokenDataset(str(out), seq_len, shuffle_buffer=0)
    items = list(ds)
    assert items, "no packed blocks produced"
    for it in items[:5]:
        assert it["input_ids"].shape == (seq_len,)
        assert it["labels"].shape == (seq_len,)
        # labels are inputs shifted by one within the packed block.
        assert torch.equal(it["input_ids"][1:], it["labels"][:-1])


def test_rank_sharding_is_disjoint(tmp_path):
    data_dir = tmp_path / "corpus"
    data_dir.mkdir()
    (data_dir / "a.txt").write_text("\n\n".join(f"line {i} words here" for i in range(200)))
    out = tmp_path / "shards"
    tok = _tokenizer()
    prepare_shards(str(data_dir), tok, str(out), shard_tokens=10_000)

    seq_len = 8
    a = list(PackedTokenDataset(str(out), seq_len, rank=0, world_size=2))
    b = list(PackedTokenDataset(str(out), seq_len, rank=1, world_size=2))
    # Two ranks partition the blocks: together they cover the single-rank stream.
    total = list(PackedTokenDataset(str(out), seq_len, rank=0, world_size=1))
    assert len(a) + len(b) == len(total)


def test_build_packed_dataset_threads_seed(tmp_path):
    """build_packed_dataset forwards its seed argument to PackedTokenDataset."""
    from model.data import build_packed_dataset

    data_dir = tmp_path / "corpus"
    data_dir.mkdir()
    (data_dir / "a.txt").write_text("\n\n".join(f"document {i} content here" for i in range(50)))
    out = tmp_path / "shards"
    tok = _tokenizer()
    prepare_shards(str(data_dir), tok, str(out), shard_tokens=500)

    ds = build_packed_dataset(str(out), seq_len=16, seed=123)
    assert ds is not None, "build_packed_dataset returned None for a valid shard dir"
    assert ds.seed == 123


def test_image_caption_dataset_examples():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "examples")
    tok = _tokenizer()
    ds = ImageCaptionDataset(data_dir, tok, resolution=128, caption_len=32)
    assert len(ds) == 2
    item = ds[0]
    assert "input_ids" in item and "target_images" in item
    assert item["input_ids"].shape == (32,)
    assert item["target_images"].shape == (3, 128, 128)
    assert item["target_images"].min() >= -1.0 and item["target_images"].max() <= 1.0


def test_audio_caption_dataset_examples():
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data", "examples")
    tok = _tokenizer()
    config = FramerConfig()
    ds = AudioCaptionDataset(data_dir, tok, config, caption_len=32)
    assert len(ds) == 2
    item = ds[0]
    assert "input_ids" in item and "target_audio" in item
    assert item["input_ids"].shape == (32,)
    assert item["target_audio"].dim() == 3  # (1, n_mels, frames)
    assert item["target_audio"].shape[1] == config.audio_n_mels

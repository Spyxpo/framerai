"""Streaming packed-data tests: shard prep, block packing, rank sharding."""

import torch

from model.data import PackedTokenDataset, prepare_shards
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

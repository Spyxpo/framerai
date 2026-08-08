"""Backbone tests: forward/backward, GQA, SDPA mask-path agreement, KV cache."""

import torch

from conftest import random_ids, tiny_config
from model.framer import FramerModel


def test_forward_backward_shapes():
    cfg = tiny_config()
    model = FramerModel(cfg)
    ids = random_ids(cfg)
    out = model(input_ids=ids, labels=ids)
    assert out["logits"].shape == (ids.shape[0], ids.shape[1], cfg.vocab_size)
    assert torch.isfinite(out["loss"])
    out["loss"].backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_gqa_configures_kv_heads():
    cfg = tiny_config(n_heads=8, n_kv_heads=2)
    model = FramerModel(cfg).eval()
    attn = model.layers[0].attn
    assert attn.n_kv_heads == 2 and attn.n_rep == 4
    # K/V projections are narrower than Q under GQA.
    assert attn.k_proj.weight.shape[0] == 2 * attn.head_dim
    assert attn.q_proj.weight.shape[0] == 8 * attn.head_dim


def test_causal_fastpath_matches_masked_path():
    """is_causal SDPA (fast path) must equal the explicit-mask path."""
    cfg = tiny_config()
    model = FramerModel(cfg).eval()
    ids = random_ids(cfg, batch=2, length=32)
    with torch.no_grad():
        fast = model.forward_lm(ids)["logits"]
        full_mask = torch.ones(ids.shape[0], ids.shape[1])
        masked = model.forward_lm(ids, attention_mask=full_mask)["logits"]
    assert torch.allclose(fast, masked, atol=1e-5)


def test_kv_cache_decode_parity():
    cfg = tiny_config()
    model = FramerModel(cfg).eval()
    ids = random_ids(cfg, batch=1, length=24)
    with torch.no_grad():
        full = model.forward_lm(ids)["logits"]
        past, steps = None, []
        for t in range(ids.shape[1]):
            o = model.forward_lm(ids[:, t:t + 1], past_kvs=past, use_cache=True)
            past = o["past_kvs"]
            steps.append(o["logits"])
        inc = torch.cat(steps, dim=1)
    assert torch.allclose(full, inc, atol=1e-4)


def test_chunked_prefill_parity():
    cfg = tiny_config()
    model = FramerModel(cfg).eval()
    ids = random_ids(cfg, batch=1, length=20)
    with torch.no_grad():
        full = model.forward_lm(ids)["logits"]
        o = model.forward_lm(ids[:, :12], use_cache=True)
        past, chunks = o["past_kvs"], [o["logits"]]
        for t in range(12, ids.shape[1]):
            o = model.forward_lm(ids[:, t:t + 1], past_kvs=past, use_cache=True)
            past = o["past_kvs"]
            chunks.append(o["logits"])
        inc = torch.cat(chunks, dim=1)
    assert torch.allclose(full, inc, atol=1e-4)


def test_rope_scaling_runs_beyond_train_length():
    cfg = tiny_config(max_seq_len=64, rope_scaling_factor=4.0, rope_scaling_type="linear")
    model = FramerModel(cfg).eval()
    long_ids = random_ids(cfg, batch=1, length=200)  # > max_seq_len
    with torch.no_grad():
        logits = model.forward_lm(long_ids)["logits"]
    assert logits.shape == (1, 200, cfg.vocab_size) and torch.isfinite(logits).all()

"""Generation tests: KV-cache text generation returns text; greedy is stable."""

import torch

from conftest import tiny_config
from model.framer import FramerModel
from model.generate import FramerGenerator
from model.tokenizer import FramerTokenizer


def _generator():
    tok = FramerTokenizer(vocab_size=300)
    tok.train(["hello world", "foo bar baz", "the cat sat"], target_vocab_size=300)
    cfg = tiny_config(vocab_size=tok.vocab_size, max_seq_len=64)
    model = FramerModel(cfg)
    return FramerGenerator(model, tok, device="cpu")


def test_generate_text_returns_string():
    gen = _generator()
    out = gen.generate_text("hello", max_new_tokens=8, temperature=0.8)
    assert isinstance(out, str)


def test_greedy_generation_is_deterministic():
    gen = _generator()
    torch.manual_seed(0)
    a = gen.generate_text("hello", max_new_tokens=10, temperature=1e-6, top_k=1, top_p=1.0)
    torch.manual_seed(0)
    b = gen.generate_text("hello", max_new_tokens=10, temperature=1e-6, top_k=1, top_p=1.0)
    assert a == b


def test_generate_code_wraps_prompt():
    gen = _generator()
    out = gen.generate_code("add two numbers", max_new_tokens=8)
    assert isinstance(out, str)

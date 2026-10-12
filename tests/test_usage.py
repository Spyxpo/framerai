"""Token usage reported for metering.

Billing reads what the generator ran, not a recount: the prompt after the
window trimmed it, any modality prefix, and every sampled token. These tests
pin that the counts are exact and reach the worker's reply.
"""

from conftest import tiny_config
from model.framer import FramerModel
from model.generate import FramerGenerator
from model.serve import _usage, handle
from model.tokenizer import FramerTokenizer


def _generator(**overrides):
    tok = FramerTokenizer(vocab_size=300)
    tok.train(["the quick brown fox jumps", "hello world foo bar"], target_vocab_size=300)
    config = tiny_config(vocab_size=300, **overrides)
    return FramerGenerator(FramerModel(config).eval(), tok, device="cpu")


def test_counts_start_at_zero():
    assert _usage(_generator()) == {"usage": {"input_tokens": 0, "output_tokens": 0}}


def test_input_counts_the_prompt_with_its_markers():
    gen = _generator()
    ids = gen.tokenizer.encode("hello world", add_special=False)
    gen.generate_text(ids, max_new_tokens=3, temperature=0)
    assert gen.last_input_tokens == len(ids) + 2  # start and end markers


def test_output_counts_every_sampled_token():
    gen = _generator()
    gen.generate_text("hello", max_new_tokens=5, temperature=0)
    # The end marker is not content and is not billed.
    if gen.last_finish_reason == "length":
        assert gen.last_output_tokens == 5
    else:
        assert gen.last_output_tokens < 5


def test_trimmed_prompts_are_billed_for_what_ran():
    gen = _generator(max_seq_len=32)
    ids = gen.tokenizer.encode("the quick brown fox jumps " * 20, add_special=False)
    gen.generate_text(ids, max_new_tokens=4, temperature=0)
    assert gen.last_prompt_tokens_dropped > 0
    assert gen.last_input_tokens == len(ids) - gen.last_prompt_tokens_dropped + 2


def test_streaming_reports_the_same_counts():
    gen = _generator()
    ids = gen.tokenizer.encode("hello world", add_special=False)
    gen.generate_text(ids, max_new_tokens=4, temperature=0)
    blocking = (gen.last_input_tokens, gen.last_output_tokens)
    list(gen.generate_stream(ids, max_new_tokens=4, temperature=0))
    assert (gen.last_input_tokens, gen.last_output_tokens) == blocking


def test_chat_reply_carries_usage(tmp_path):
    gen = _generator()
    result = handle(gen, "chat", {"prompt": "hello", "max_new_tokens": 3, "temperature": 0,
                                  "out_dir": str(tmp_path)})
    assert result["usage"]["input_tokens"] > 0
    assert result["usage"]["output_tokens"] == gen.last_output_tokens

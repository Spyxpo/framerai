"""A reply is what the model said, not the prompt it was given.

generate_text returns prompt and continuation together, and the chat, code,
understand and document ops passed that straight to the client, so every reply
began with the formatted conversation so far. Streaming was unaffected, which
is why the two paths disagreed about the same answer.
"""

from conftest import tiny_config
from model.framer import FramerModel
from model.generate import FramerGenerator
from model.serve import handle
from model.tokenizer import FramerTokenizer


def _generator():
    tok = FramerTokenizer(vocab_size=300)
    tok.train(["hello world foo bar", "def add(a, b): return a + b"], target_vocab_size=300)
    return FramerGenerator(FramerModel(tiny_config(vocab_size=300, max_seq_len=256)).eval(), tok, "cpu")


def test_chat_reply_carries_no_prompt(tmp_path):
    gen = _generator()
    res = handle(gen, "chat", {
        "messages": [
            {"role": "user", "content": "first question"},
            {"role": "assistant", "content": "first answer"},
            {"role": "user", "content": "second question"},
        ],
        "max_new_tokens": 6, "temperature": 1.0, "seed": 3, "out_dir": str(tmp_path),
    })
    assert "first question" not in res["content"]
    assert not res["content"].startswith("<user>")
    assert res["content"] == gen.last_completion


def test_blocking_and_streaming_agree(tmp_path):
    gen = _generator()
    params = {"prompt": "hello", "max_new_tokens": 6, "temperature": 0, "out_dir": str(tmp_path)}
    blocking = handle(gen, "chat", dict(params))["content"]
    streamed = [c for c in handle(gen, "chat", {**params, "stream": True})]
    assert streamed[-1]["result"]["content"] == blocking


def test_code_reply_carries_no_instruction(tmp_path):
    gen = _generator()
    res = handle(gen, "code", {"prompt": "add two numbers", "max_new_tokens": 6, "out_dir": str(tmp_path)})
    assert "Write python code" not in res["content"]


def test_a_stop_string_ends_the_reply_there():
    gen = _generator()
    gen.generate_text("hello", max_new_tokens=8, temperature=0)
    full = gen.last_completion
    if len(full) > 2:
        gen.generate_text("hello", max_new_tokens=8, temperature=0, stop=[full[1:3]])
        assert gen.last_completion == full[: full.index(full[1:3])]

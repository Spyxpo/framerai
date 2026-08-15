"""Tests for the inference worker's cognition ops.

The worker is the only path the backend has to the model, so the contract that
matters here is: without ``--mind`` nothing about it changes, and with a mind
attached chat carries a trace and the introspection ops answer. These call
``handle`` directly rather than spawning the worker, so no checkpoint is needed.
"""

import pytest
import torch

from conftest import tiny_config
from model.cognition import CognitionConfig, Mind
from model.framer import FramerModel
from model.generate import FramerGenerator
from model.serve import handle
from model.tokenizer import FramerTokenizer


def _generator() -> FramerGenerator:
    tokenizer = FramerTokenizer(vocab_size=300)
    tokenizer.train(["hello world vocoder diffusion"], target_vocab_size=300)
    config = tiny_config(vocab_size=tokenizer.vocab_size, max_seq_len=64)
    return FramerGenerator(FramerModel(config), tokenizer, device="cpu")


@pytest.fixture(scope="module")
def generator():
    return _generator()


@pytest.fixture
def mind(generator):
    return Mind.from_generator(
        generator, CognitionConfig(d_embed=32, episodic_capacity=32, retrieval_k=2)
    )


def test_chat_without_a_mind_behaves_exactly_as_before(generator):
    result = handle(generator, "chat", {"prompt": "hi", "max_new_tokens": 4})
    assert isinstance(result["content"], str)
    assert "trace" not in result


def test_chat_with_a_mind_returns_the_trace_alongside_the_reply(generator, mind):
    result = handle(generator, "chat", {"prompt": "what is a vocoder", "max_new_tokens": 4}, mind)

    assert isinstance(result["content"], str)
    trace = result["trace"]
    assert trace["language"]["code"] == "en"
    assert trace["sampling"]["temperature"] > 0
    assert mind.tick >= 1


def test_mind_ops_refuse_clearly_when_the_layer_is_off(generator):
    for op in ("wonder", "reflect", "introspect", "feedback", "see", "hear", "watch", "live"):
        with pytest.raises(ValueError, match="--mind"):
            handle(generator, op, {})


def test_wonder_reflect_and_introspect_answer(generator, mind):
    handle(generator, "chat", {"prompt": "what is a vocoder", "max_new_tokens": 4}, mind)

    assert handle(generator, "wonder", {}, mind)["content"].endswith("?")

    reflection = handle(generator, "reflect", {}, mind)
    assert reflection["rehearsed"] > 0
    assert reflection["reflection"]

    snapshot = handle(generator, "introspect", {}, mind)
    assert snapshot["tick"] > 0
    assert "languages" in snapshot and "senses" in snapshot


def test_feedback_records_the_reward_and_reports_the_new_state(generator, mind):
    handle(generator, "chat", {"prompt": "what is a vocoder", "max_new_tokens": 4}, mind)
    before = mind.affect.valence

    trace = handle(generator, "feedback", {"value": 1.0, "note": "that was right"}, mind)
    assert trace["kind"] == "feedback"
    assert mind.affect.valence > before


def test_an_unknown_op_is_still_rejected(generator, mind):
    with pytest.raises(ValueError, match="Unknown op"):
        handle(generator, "levitate", {}, mind)


def test_seeing_an_image_file_becomes_an_episode(generator, mind, tmp_path):
    from PIL import Image

    path = tmp_path / "frame.png"
    Image.fromarray((torch.rand(32, 32, 3) * 255).byte().numpy()).save(path)

    # A text-only build has no vision tower, and the op must say so rather than
    # silently remembering something it never looked at.
    with pytest.raises(RuntimeError, match="multimodal"):
        handle(generator, "see", {"image_path": str(path), "describe": False}, mind)

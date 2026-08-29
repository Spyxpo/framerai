"""Unit tests for instruction-following evaluation benchmark."""


from model.configs import FramerConfig
from model.eval.benchmarks import evaluate_instruction_following
from model.framer import FramerModel
from model.generate import FramerGenerator
from model.tokenizer import FramerTokenizer


def test_evaluate_instruction_following():
    config = FramerConfig.from_preset("framer-tiny")
    model = FramerModel(config)
    tokenizer = FramerTokenizer(vocab_size=config.vocab_size)
    generator = FramerGenerator(model, tokenizer, device="cpu")

    result = evaluate_instruction_following(generator)
    assert result.benchmark == "instruction-following"
    assert "format_adherence" in result.metrics
    assert "tool_call_validity" in result.metrics
    assert result.samples == 2

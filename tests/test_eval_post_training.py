"""Unit tests for instruction-following evaluation benchmark."""

from model.configs import FramerConfig
from model.eval.benchmarks import evaluate_instruction_following
from model.framer import FramerModel
from model.generate import FramerGenerator
from model.tokenizer import FramerTokenizer


class MockGenerator:
    def __init__(self, output_map: dict[str, str]):
        self.output_map = output_map

    def generate_text(self, prompt: str, **kwargs) -> str:
        return prompt + self.output_map.get(prompt, "")


def test_evaluate_instruction_following_basic():
    config = FramerConfig.from_preset("framer-tiny")
    model = FramerModel(config)
    tokenizer = FramerTokenizer(vocab_size=config.vocab_size)
    generator = FramerGenerator(model, tokenizer, device="cpu")

    result = evaluate_instruction_following(generator)
    assert result.benchmark == "instruction-following"
    assert "format_adherence" in result.metrics
    assert "tool_call_validity" in result.metrics
    assert result.samples == 2


def test_evaluate_instruction_following_empty_output():
    # Empty outputs should receive 0.0 score (not 1.0)
    mock_gen = MockGenerator({
        "<user>Search the web for FramerAI.<assistant>": "",
        "<user>Say hello.<assistant>": "",
    })
    result = evaluate_instruction_following(mock_gen)
    assert result.metrics["format_adherence"] == 0.0
    assert result.metrics["tool_call_validity"] == 0.0


def test_evaluate_instruction_following_valid_responses():
    # Valid tool call and valid text response should receive 1.0 score
    mock_gen = MockGenerator({
        "<user>Search the web for FramerAI.<assistant>": '<tool_call>{"name": "web_search", "arguments": {"query": "FramerAI"}}</tool_call>',
        "<user>Say hello.<assistant>": "Hello! How can I help you today?",
    })
    result = evaluate_instruction_following(mock_gen)
    assert result.metrics["format_adherence"] == 1.0
    assert result.metrics["tool_call_validity"] == 1.0


def test_evaluate_instruction_following_invalid_tool_call():
    # Invalid JSON in tool call should fail
    mock_gen = MockGenerator({
        "<user>Search the web for FramerAI.<assistant>": "<tool_call>{invalid_json}</tool_call>",
        "<user>Say hello.<assistant>": "Hello!",
    })
    result = evaluate_instruction_following(mock_gen)
    assert result.metrics["format_adherence"] == 0.5
    assert result.metrics["tool_call_validity"] == 0.5


def test_evaluate_instruction_following_unexpected_tool_call():
    # Emitting tool call when expect_tool=False should fail
    mock_gen = MockGenerator({
        "<user>Search the web for FramerAI.<assistant>": '<tool_call>{"name": "web_search", "arguments": {}}</tool_call>',
        "<user>Say hello.<assistant>": '<tool_call>{"name": "unwanted", "arguments": {}}</tool_call>',
    })
    result = evaluate_instruction_following(mock_gen)
    assert result.metrics["format_adherence"] == 0.5
    assert result.metrics["tool_call_validity"] == 0.5

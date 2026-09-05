"""Unit tests for ChatTemplate, versioned chat formatting, next-token alignment, and real tool parser integration."""

import torch

from model.tokenizer import ChatTemplate, FramerTokenizer
from model.tools.base import ToolRegistry
from model.tools.loop import parse_tool_call, render_prompt


def test_chat_template_format_messages():
    template = ChatTemplate(version="v1")
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hi there!"},
    ]

    formatted = template.format_messages(messages)
    assert formatted == "<system>You are a helpful assistant.<user>Hello!<assistant>Hi there!"

    with_prompt = template.format_messages(messages[:2], add_generation_prompt=True)
    assert with_prompt == "<system>You are a helpful assistant.<user>Hello!<assistant>"


def test_chat_template_tool_call_formatting_and_real_parser():
    template = ChatTemplate(version="v1")
    call_payload = {"name": "web_search", "arguments": {"query": "Python 3.13"}}
    messages = [
        {"role": "user", "content": "Search for Python docs"},
        {"role": "assistant", "tool_calls": call_payload},
        {"role": "tool", "name": "web_search", "content": "Python documentation results..."},
    ]

    formatted = template.format_messages(messages)
    assert "<tool_call>{\"name\": \"web_search\", \"arguments\": {\"query\": \"Python 3.13\"}}</tool_call>" in formatted
    assert "<tool>[web_search] Python documentation results..." in formatted

    # Test that the output is directly accepted by the REAL parse_tool_call parser
    tool_call = parse_tool_call(formatted)
    assert tool_call is not None
    assert tool_call.name == "web_search"
    assert tool_call.arguments == {"query": "Python 3.13"}


def test_chat_template_next_token_label_shift():
    tokenizer = FramerTokenizer(vocab_size=400)
    template = ChatTemplate(version="v1")

    messages = [
        {"role": "system", "content": "System prompt"},
        {"role": "user", "content": "User prompt"},
        {"role": "assistant", "content": "Assistant answer"},
    ]

    encoded = template.encode_conversation(messages, tokenizer, max_len=128, pad_to_max=False)
    input_ids = encoded["input_ids"]
    labels = encoded["labels"]

    assert len(input_ids) == len(labels)

    ids_list = input_ids.tolist()
    labels_list = labels.tolist()

    sys_ids = tokenizer.encode("<system>System prompt", add_special=False)
    usr_ids = tokenizer.encode("<user>User prompt", add_special=False)
    ast_ids = tokenizer.encode("<assistant>Assistant answer", add_special=False)

    full_sequence = [tokenizer.sos_id] + sys_ids + usr_ids + ast_ids + [tokenizer.eos_id]

    # Verify input_ids equals full_sequence[:-1]
    assert ids_list == full_sequence[:-1]

    # Prefix length (sos + system + user)
    prefix_len = 1 + len(sys_ids) + len(usr_ids)

    # Prompt positions in labels (indices 0 .. prefix_len-2) must be masked with -100
    for i in range(prefix_len - 1):
        assert labels_list[i] == -100, f"Index {i} should be masked"

    # Index (prefix_len - 1) is the last prompt token position (end of <user> turn).
    # Its label MUST be the FIRST token of the assistant turn (ast_ids[0]), proving NEXT-TOKEN prediction!
    assert labels_list[prefix_len - 1] == ast_ids[0], "Prompt end label must predict first assistant token"

    # Subsequent assistant tokens are shifted next-token targets
    for j in range(len(ast_ids) - 1):
        idx = prefix_len + j
        assert labels_list[idx] == ast_ids[j + 1], f"Label at {idx} must be next assistant token {ast_ids[j+1]}"

    # Final label must be EOS id
    assert labels_list[-1] == tokenizer.eos_id, "Final target label must be EOS token id"


def test_reserved_tool_tokens_in_tokenizer():
    tokenizer = FramerTokenizer()
    assert "<tool>" in tokenizer.reserved_tokens
    assert "<tool_call>" in tokenizer.reserved_tokens
    assert tokenizer.reserved_tokens["<tool>"] == 273
    assert tokenizer.reserved_tokens["<tool_call>"] == 274


def test_inference_integration_with_render_prompt():
    registry = ToolRegistry()
    rendered = render_prompt(registry, "Hello world")
    assert rendered.startswith("<system>")
    assert "<user>Hello world" in rendered
    assert rendered.endswith("<assistant>")


def test_serve_path_chat_template_single_application(monkeypatch):
    from conftest import tiny_config
    from model.framer import FramerModel
    from model.generate import FramerGenerator
    from model.serve import handle
    from model.tokenizer import FramerTokenizer
    from model.tools import ToolRegistry
    from model.tools.base import Tool, ToolResult

    class FakeSearch(Tool):
        name = "web_search"
        description = "Stub search."
        parameters = {"query": "string"}

        def run(self, query: str = "", **_):
            return ToolResult.success("result text")

    tokenizer = FramerTokenizer(vocab_size=300)
    tokenizer.train(["hello world rectified flow"], target_vocab_size=300)
    config = tiny_config(vocab_size=tokenizer.vocab_size, max_seq_len=64)
    generator = FramerGenerator(FramerModel(config), tokenizer, device="cpu")
    registry = ToolRegistry([FakeSearch()])

    captured_prompts = []

    def mock_generate_text(prompt, **kwargs):
        captured_prompts.append(prompt)
        return prompt + "I answer directly."

    monkeypatch.setattr(generator, "generate_text", mock_generate_text)

    # 1. Normal chat request with prompt string
    captured_prompts.clear()
    res1 = handle(generator, "chat", {"prompt": "what is rectified flow", "max_new_tokens": 8})
    assert len(captured_prompts) == 1
    p1 = captured_prompts[0]
    assert res1["content"] == p1 + "I answer directly."
    assert "<user><user>" not in p1
    assert "<system><system>" not in p1
    assert "<assistant><assistant>" not in p1
    assert p1.startswith("<user>what is rectified flow")
    assert p1.endswith("<assistant>")

    # 2. Normal chat request with messages list
    captured_prompts.clear()
    res2 = handle(
        generator,
        "chat",
        {"messages": [{"role": "user", "content": "what is rectified flow"}], "max_new_tokens": 8},
    )
    assert len(captured_prompts) == 1
    p2 = captured_prompts[0]
    assert res2["content"] == p2 + "I answer directly."
    assert "<user><user>" not in p2
    assert "<system><system>" not in p2
    assert "<assistant><assistant>" not in p2
    assert p2.startswith("<user>what is rectified flow")
    assert p2.endswith("<assistant>")

    # 3. Tool-enabled chat request with prompt string
    captured_prompts.clear()
    res3 = handle(
        generator,
        "chat",
        {"prompt": "what is rectified flow", "max_new_tokens": 8, "tools": True},
        tools=registry,
    )
    assert res3["content"] == "I answer directly."
    assert len(captured_prompts) == 1
    p3 = captured_prompts[0]
    assert "<user><user>" not in p3
    assert "<system><system>" not in p3
    assert "<assistant><assistant>" not in p3
    assert p3.startswith("<system>")
    assert "Tools:" in p3
    assert "<user>what is rectified flow" in p3
    assert p3.endswith("<assistant>")

    # 4. Tool-enabled chat request with messages list
    captured_prompts.clear()
    res4 = handle(
        generator,
        "chat",
        {
            "messages": [{"role": "user", "content": "what is rectified flow"}],
            "max_new_tokens": 8,
            "tools": True,
        },
        tools=registry,
    )
    assert res4["content"] == "I answer directly."
    assert len(captured_prompts) == 1
    p4 = captured_prompts[0]
    assert "<user><user>" not in p4
    assert "<system><system>" not in p4
    assert "<assistant><assistant>" not in p4
    assert p4.startswith("<system>")
    assert "Tools:" in p4
    assert "<user>what is rectified flow" in p4
    assert p4.endswith("<assistant>")


def test_left_truncation_preserves_assistant_turn():
    """Regression test for Issue #234: Over-long conversation must be left-truncated to keep newest assistant tokens."""
    tokenizer = FramerTokenizer(vocab_size=300)
    template = ChatTemplate("v1")

    # Long user prompt + short assistant answer
    long_user_text = "lorem ipsum " * 50
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": long_user_text},
        {"role": "assistant", "content": "Final assistant response."},
    ]

    max_len = 32
    encoded = template.encode_conversation(messages, tokenizer, max_len=max_len, pad_to_max=True)

    input_ids = encoded["input_ids"]
    labels = encoded["labels"]
    attention_mask = encoded["attention_mask"]

    assert len(input_ids) == max_len
    assert len(labels) == max_len
    assert len(attention_mask) == max_len

    # Assistant target tokens MUST survive in labels
    non_masked = (labels != -100).nonzero(as_tuple=True)[0]
    assert len(non_masked) > 0, "Left truncation must preserve the assistant response tokens at the end"

    # Verify rightmost tokens match the end of the full conversation (left truncation)
    full_encoded = template.encode_conversation(messages, tokenizer, max_len=10000, pad_to_max=False)
    expected_input_ids = full_encoded["input_ids"][-max_len:]
    assert torch.equal(input_ids, expected_input_ids)


def test_attention_mask_returned_and_correct():
    """Verify attention_mask is returned and accurately marks real tokens vs padding."""
    tokenizer = FramerTokenizer(vocab_size=300)
    template = ChatTemplate("v1")

    messages = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ]

    encoded_short = template.encode_conversation(messages, tokenizer, max_len=64, pad_to_max=True)
    assert "attention_mask" in encoded_short

    mask = encoded_short["attention_mask"]
    input_ids = encoded_short["input_ids"]
    assert mask.shape == (64,)

    # Real tokens should have mask == 1, pad tokens should have mask == 0
    num_real = sum(1 for tok in input_ids.tolist() if tok != tokenizer.pad_id)
    assert (mask[:num_real] == 1).all()
    assert (mask[num_real:] == 0).all()

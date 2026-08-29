"""Unit tests for ChatTemplate, versioned chat formatting, next-token alignment, and real tool parser integration."""

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


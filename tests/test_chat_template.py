"""Unit tests for ChatTemplate and versioned chat formatting."""


from model.tokenizer import ChatTemplate, FramerTokenizer


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


def test_chat_template_tool_call_formatting():
    template = ChatTemplate(version="v1")
    messages = [
        {"role": "user", "content": "Search for Python docs"},
        {"role": "assistant", "tool_calls": '{"name": "web_search", "arguments": {"query": "Python"}}'},
        {"role": "tool", "name": "web_search", "content": "Python 3.13 documentation..."},
    ]

    formatted = template.format_messages(messages)
    assert "<tool_call>" in formatted
    assert "<tool>[web_search] Python 3.13 documentation..." in formatted


def test_chat_template_loss_masking():
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

    # SOS token is masked
    assert labels_list[0] == -100

    # Calculate token sequence lengths
    sys_ids = tokenizer.encode("<system>System prompt", add_special=False)
    usr_ids = tokenizer.encode("<user>User prompt", add_special=False)
    ast_ids = tokenizer.encode("<assistant>Assistant answer", add_special=False)

    # Prefix (SOS + sys + usr) should all be masked (-100)
    prefix_len = 1 + len(sys_ids) + len(usr_ids)
    for i in range(prefix_len):
        assert labels_list[i] == -100, f"Index {i} should be masked"

    # Assistant turn tokens should match input_ids
    for j in range(len(ast_ids)):
        idx = prefix_len + j
        assert ids_list[idx] == ast_ids[j]
        assert labels_list[idx] == ast_ids[j], f"Assistant token at {idx} should be target label"


def test_reserved_tool_tokens_in_tokenizer():
    tokenizer = FramerTokenizer()
    assert "<tool>" in tokenizer.reserved_tokens
    assert "<tool_call>" in tokenizer.reserved_tokens
    assert tokenizer.reserved_tokens["<tool>"] == 273
    assert tokenizer.reserved_tokens["<tool_call>"] == 274

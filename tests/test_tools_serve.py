import io
import json

import pytest

from conftest import tiny_config
from model.framer import FramerModel
from model.generate import FramerGenerator
from model.serve import handle, make_stdio_approver
from model.tokenizer import FramerTokenizer
from model.tools import ToolRegistry, ToolResult
from model.tools.base import Tool
from model.tools.cli import ShellPolicy


class FakeSearch(Tool):
    name = "web_search"
    description = "Stub search."
    parameters = {"query": "string"}

    def __init__(self):
        self.queries = []

    def run(self, query: str = "", max_results: int = 5, **_):
        self.queries.append(query)
        return ToolResult.success(
            "Rectified flow\nhttps://example.com/flow",
            query=query,
            results=[{"title": "Rectified flow", "url": "https://example.com/flow"}],
        )


class FakeFetch(Tool):
    name = "web_fetch"
    description = "Stub fetch."
    parameters = {"url": "string"}

    def run(self, url: str = "", max_chars: int | None = None, **_):
        if "missing" in url:
            return ToolResult.failure(f"{url} had no readable text", url=url)
        return ToolResult.success(f"page at {url}", url=url, title="Page", truncated=False)


@pytest.fixture(scope="module")
def generator() -> FramerGenerator:
    tokenizer = FramerTokenizer(vocab_size=300)
    tokenizer.train(["hello world rectified flow"], target_vocab_size=300)
    config = tiny_config(vocab_size=tokenizer.vocab_size, max_seq_len=64)
    return FramerGenerator(FramerModel(config), tokenizer, device="cpu")


@pytest.fixture
def registry():
    return ToolRegistry([FakeSearch(), FakeFetch()])


def test_chat_without_tools_behaves_exactly_as_before(generator):
    result = handle(generator, "chat", {"prompt": "hi", "max_new_tokens": 4})
    assert isinstance(result["content"], str)
    assert "tools" not in result


def test_chat_ignores_a_tools_request_when_none_are_registered(generator):
    result = handle(generator, "chat", {"prompt": "hi", "max_new_tokens": 4, "tools": ["web"]})
    assert "tools" not in result


def test_chat_with_tools_runs_the_loop_and_returns_a_trace(generator, registry, monkeypatch):
    replies = iter(
        [
            '<tool_call>{"name": "web_search", "arguments": {"query": "rectified flow"}}</tool_call>',
            "It is a straight path. https://example.com/flow",
        ]
    )
    monkeypatch.setattr(generator, "generate_text", lambda prompt, **_: prompt + next(replies))

    result = handle(
        generator,
        "chat",
        {"prompt": "what is rectified flow", "max_new_tokens": 8, "tools": True},
        tools=registry,
    )

    assert result["content"].startswith("It is a straight path")
    assert result["tools"]["used"] == ["web_search"]
    assert result["tools"]["stopped"] == "answered"
    assert registry.get("web_search").queries == ["rectified flow"]


def test_chat_accepts_a_toolset_name_as_well_as_a_tool_name(generator, registry, monkeypatch):
    monkeypatch.setattr(generator, "generate_text", lambda prompt, **_: prompt + "answer")

    for requested in (["web"], ["web_search"], "web", True):
        result = handle(
            generator,
            "chat",
            {"prompt": "q", "max_new_tokens": 4, "tools": requested},
            tools=registry,
        )
        assert result["tools"]["stopped"] == "answered"


def test_search_op_returns_the_results(generator, registry):
    result = handle(generator, "search", {"query": "rectified flow"}, tools=registry)
    assert "https://example.com/flow" in result["content"]
    assert result["results"][0]["title"] == "Rectified flow"


def test_search_op_falls_back_to_the_prompt_field(generator, registry):
    handle(generator, "search", {"prompt": "from prompt"}, tools=registry)
    assert registry.get("web_search").queries == ["from prompt"]


def test_fetch_op_raises_on_a_failed_fetch(generator, registry):
    with pytest.raises(ValueError, match="no readable text"):
        handle(generator, "fetch", {"url": "https://example.com/missing"}, tools=registry)


def test_web_ops_refuse_clearly_when_no_tools_are_registered(generator):
    for op in ("search", "fetch"):
        with pytest.raises(ValueError, match="--tools web"):
            handle(generator, op, {"query": "q", "url": "https://example.com"})


# --- stdio approval tests -------------------------------------------------


def test_make_stdio_approver_accepted(monkeypatch):
    response_stream = io.StringIO()

    def fake_print(obj):
        if obj.get("type") == "approval_request":
            resp = {
                "type": "approval_response",
                "approval_id": obj["approval_id"],
                "approved": True,
            }
            response_stream.write(json.dumps(resp) + "\n")
            response_stream.seek(0)

    monkeypatch.setattr("model.serve._print", fake_print)
    monkeypatch.setattr("sys.stdin", response_stream)

    approver = make_stdio_approver("/sandbox", timeout_sec=1.0)
    assert approver("cat notes.txt", ["cat", "notes.txt"]) is True


def test_make_stdio_approver_denied(monkeypatch):
    response_stream = io.StringIO()

    def fake_print(obj):
        if obj.get("type") == "approval_request":
            resp = {
                "type": "approval_response",
                "approval_id": obj["approval_id"],
                "approved": False,
            }
            response_stream.write(json.dumps(resp) + "\n")
            response_stream.seek(0)

    monkeypatch.setattr("model.serve._print", fake_print)
    monkeypatch.setattr("sys.stdin", response_stream)

    approver = make_stdio_approver("/sandbox", timeout_sec=1.0)
    assert approver("cat notes.txt", ["cat", "notes.txt"]) is False


def test_make_stdio_approver_timeout(monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO())

    approver = make_stdio_approver("/sandbox", timeout_sec=0.01)
    assert approver("ls", ["ls"]) is False


def test_make_stdio_approver_malformed_json(monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO("{not valid json\n"))

    approver = make_stdio_approver("/sandbox", timeout_sec=1.0)
    assert approver("ls", ["ls"]) is False


def test_make_stdio_approver_eof_input(monkeypatch):
    monkeypatch.setattr("sys.stdin", io.StringIO(""))

    approver = make_stdio_approver("/sandbox", timeout_sec=1.0)
    assert approver("ls", ["ls"]) is False


def test_make_stdio_approver_wrong_approval_id(monkeypatch):
    resp = (
        json.dumps({
            "type": "approval_response",
            "approval_id": "wrong-uuid-1234",
            "approved": True,
        })
        + "\n"
    )
    monkeypatch.setattr("sys.stdin", io.StringIO(resp))

    approver = make_stdio_approver("/sandbox", timeout_sec=1.0)
    assert approver("ls", ["ls"]) is False


def test_deny_list_precedence_over_approval(tmp_path):
    seen = []

    def mock_approver(command, argv):
        seen.append(command)
        return True

    policy = ShellPolicy(mode="ask", root=str(tmp_path), approve=mock_approver)
    decision = policy.decide(["rm", "-rf", "/"])

    assert not decision
    assert "refused" in decision.reason
    assert seen == []

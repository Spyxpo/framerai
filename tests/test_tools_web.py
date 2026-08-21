"""Tool tests: the registry, the tool-calling loop, and the internet tools.

Nothing here touches the network. The client's transport is injected, so the
parsing, redirect unwrapping, address filtering, truncation, and loop control
are all exercised against fixed bytes.
"""

import pytest

from model.tools import (
    ToolError,
    ToolRegistry,
    ToolResult,
    build_registry,
    expand_toolsets,
    parse_tool_call,
    run_tool_loop,
)
from model.tools.base import Tool
from model.tools.loop import ToolCallError
from model.tools.web import (
    SearchClient,
    WebFetchTool,
    WebSearchTool,
    check_url,
    html_to_text,
    unwrap_redirect,
)

SEARCH_HTML = """
<html><body>
  <div class="result">
    <a rel="nofollow" class="result__a"
       href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fflow&amp;rut=abc">Rectified flow</a>
    <a class="result__snippet">A straight path between noise and data.</a>
  </div>
  <div class="result">
    <a rel="nofollow" class="result__a" href="https://example.org/second">Second hit</a>
    <a class="result__snippet">Another snippet.</a>
  </div>
  <div class="result">
    <a rel="nofollow" class="result__a" href="https://example.net/third">Third hit</a>
  </div>
</body></html>
"""

PAGE_HTML = """
<html><head><title>  Flow   matching </title>
<style>body { color: red; }</style>
<script>window.tracked = true;</script></head>
<body><h1>Heading</h1><p>First paragraph.</p><p>Second paragraph.</p>
<noscript>hidden</noscript></body></html>
"""


class EchoTool(Tool):
    name = "echo"
    description = "Echo the text back."
    parameters = {"text": "string"}

    def run(self, text: str = "", **_):
        return ToolResult.success(text.upper(), text=text)


class StrictTool(Tool):
    """Takes exactly one argument, so a wrong one is a TypeError."""

    name = "strict"
    description = "Double a number."
    parameters = {"value": "integer"}

    def run(self, value: int):
        return ToolResult.success(str(value * 2))


class BoomTool(Tool):
    name = "boom"
    description = "Always refuses."
    parameters = {}

    def run(self, **_):
        raise ToolError("refused on purpose")


def _client(payload):
    """A client whose transport returns fixed bytes and records its calls."""
    calls = []

    def transport(url, data, timeout, max_bytes):
        calls.append({"url": url, "data": data, "timeout": timeout, "max_bytes": max_bytes})
        body = payload(url) if callable(payload) else payload
        return body.encode() if isinstance(body, str) else body

    client = SearchClient(transport=transport)
    return client, calls


def _scripted(replies):
    """A generate() that returns the scripted replies in order, prompt echoed."""
    seen = iter(replies)

    def generate(prompt):
        return prompt + next(seen)

    return generate


# --- registry -------------------------------------------------------------


def test_registry_runs_a_tool():
    registry = ToolRegistry([EchoTool()])
    result = registry.run("echo", {"text": "hi"})
    assert result.ok and result.content == "HI"


def test_registry_reports_unknown_tools_instead_of_raising():
    result = ToolRegistry([EchoTool()]).run("nope", {})
    assert not result.ok
    assert "unknown tool" in result.content and "echo" in result.content


def test_registry_turns_bad_arguments_into_a_failed_result():
    registry = ToolRegistry([StrictTool()])
    assert registry.run("strict", {"value": 3}).content == "6"

    result = registry.run("strict", {"wrong": 1})
    assert not result.ok and "bad arguments" in result.content


def test_registry_rejects_non_object_arguments():
    result = ToolRegistry([EchoTool()]).run("echo", ["hi"])
    assert not result.ok and "must be an object" in result.content


def test_registry_turns_tool_errors_into_a_failed_result():
    result = ToolRegistry([BoomTool()]).run("boom", {})
    assert not result.ok and result.content == "refused on purpose"


def test_registry_rejects_duplicate_names():
    registry = ToolRegistry([EchoTool()])
    with pytest.raises(ValueError):
        registry.register(EchoTool())


def test_subset_keeps_known_names_and_drops_the_rest():
    registry = ToolRegistry([EchoTool(), BoomTool()])
    subset = registry.subset(["echo", "absent"])
    assert subset.names() == ["echo"]


def test_build_registry_expands_the_web_toolset():
    registry = build_registry("web")
    assert registry.names() == ["web_search", "web_fetch"]
    assert build_registry(None).names() == []
    assert expand_toolsets(["web"]) == ["web_search", "web_fetch"]


def test_build_registry_rejects_an_unknown_toolset():
    with pytest.raises(ValueError):
        build_registry(["telepathy"])


# --- parsing --------------------------------------------------------------


def test_parse_tool_call_reads_name_and_arguments():
    call = parse_tool_call('<tool_call>{"name": "echo", "arguments": {"text": "hi"}}</tool_call>')
    assert call.name == "echo" and call.arguments == {"text": "hi"}


def test_parse_tool_call_defaults_missing_arguments():
    call = parse_tool_call('<tool_call>{"name": "echo"}</tool_call>')
    assert call.arguments == {}


def test_parse_tool_call_returns_none_for_plain_prose():
    assert parse_tool_call("Rectified flow is a straight path.") is None


@pytest.mark.parametrize(
    "text",
    [
        '<tool_call>{"name": "echo"',
        "<tool_call>not json at all</tool_call>",
        "<tool_call>[1, 2]</tool_call>",
        '<tool_call>{"arguments": {}}</tool_call>',
        '<tool_call>{"name": "echo", "arguments": 3}</tool_call>',
    ],
)
def test_parse_tool_call_rejects_broken_calls(text):
    with pytest.raises(ToolCallError):
        parse_tool_call(text)


# --- the loop -------------------------------------------------------------


def test_loop_without_tools_is_one_plain_generation():
    reply, trace = run_tool_loop(_scripted(["plain answer"]), ToolRegistry(), "hello")
    assert reply == "plain answer"
    assert trace.stopped == "no_tools" and trace.steps == []


def test_loop_runs_a_tool_then_answers():
    registry = ToolRegistry([EchoTool()])
    generate = _scripted(
        ['<tool_call>{"name": "echo", "arguments": {"text": "hi"}}</tool_call>', "it said HI"]
    )
    reply, trace = run_tool_loop(generate, registry, "what did it say?")

    assert reply == "it said HI"
    assert trace.stopped == "answered"
    assert [step.name for step in trace.steps] == ["echo"]
    assert trace.steps[0].result.content == "HI"
    assert trace.to_dict()["used"] == ["echo"]


def test_loop_feeds_a_malformed_call_back_instead_of_failing():
    registry = ToolRegistry([EchoTool()])
    reply, trace = run_tool_loop(
        _scripted(["<tool_call>{oops}</tool_call>", "sorry, answer"]), registry, "q"
    )

    assert reply == "sorry, answer"
    assert trace.steps[0].name == "(malformed)"
    assert not trace.steps[0].result.ok


def test_loop_stops_at_max_steps_and_strips_the_dangling_call():
    registry = ToolRegistry([EchoTool()])
    call = '<tool_call>{"name": "echo", "arguments": {"text": "again"}}</tool_call>'
    reply, trace = run_tool_loop(_scripted([call] * 3), registry, "q", max_steps=3)

    assert trace.stopped == "max_steps"
    assert len(trace.steps) == 3
    assert "<tool_call>" not in reply


def test_loop_context_carries_successful_results_only():
    registry = ToolRegistry([EchoTool(), BoomTool()])
    generate = _scripted(
        [
            '<tool_call>{"name": "boom"}</tool_call>',
            '<tool_call>{"name": "echo", "arguments": {"text": "ok"}}</tool_call>',
            "done",
        ]
    )
    _, trace = run_tool_loop(generate, registry, "q")

    assert trace.context() == "[echo] OK"


def test_loop_prompt_lists_every_tool():
    registry = ToolRegistry([EchoTool(), BoomTool()])
    generate_calls = []

    def generate(prompt):
        generate_calls.append(prompt)
        return prompt + "answer"

    run_tool_loop(generate, registry, "question")
    assert "echo(text: string)" in generate_calls[0]
    assert "boom()" in generate_calls[0]
    assert "question" in generate_calls[0]


# --- the search client -----------------------------------------------------------


def test_unwrap_redirect_recovers_the_target_url():
    href = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fexample.com%2Fa%20b&rut=x"
    assert unwrap_redirect(href) == "https://example.com/a b"
    assert unwrap_redirect("https://example.org/plain") == "https://example.org/plain"
    assert unwrap_redirect("") == ""


def test_search_parses_titles_urls_and_snippets():
    client, calls = _client(SEARCH_HTML)
    results = client.search("rectified flow")

    assert [r.url for r in results] == [
        "https://example.com/flow",
        "https://example.org/second",
        "https://example.net/third",
    ]
    assert results[0].title == "Rectified flow"
    assert results[0].snippet == "A straight path between noise and data."
    assert results[2].snippet == ""
    assert calls[0]["data"] == b"q=rectified+flow&kl=wt-wt"


def test_search_honours_max_results():
    client, _ = _client(SEARCH_HTML)
    assert len(client.search("q", max_results=2)) == 2


def test_search_rejects_an_empty_query():
    client, _ = _client(SEARCH_HTML)
    with pytest.raises(ToolError):
        client.search("   ")


def test_instant_answer_reads_the_abstract():
    client, _ = _client('{"AbstractText": "A  short   summary", "Answer": ""}')
    assert client.instant_answer("q") == "A short summary"


def test_instant_answer_is_empty_when_the_payload_is_not_json():
    client, _ = _client("<html>rate limited</html>")
    assert client.instant_answer("q") == ""


def test_html_to_text_drops_scripts_styles_and_markup():
    title, text = html_to_text(PAGE_HTML)
    assert title == "Flow matching"
    assert "First paragraph." in text and "Second paragraph." in text
    assert "window.tracked" not in text and "color: red" not in text


# --- the tools ------------------------------------------------------------


def test_web_search_tool_renders_results_with_urls():
    client, _ = _client(
        lambda url: SEARCH_HTML if "html.duckduckgo" in url else '{"AbstractText": ""}'
    )
    result = WebSearchTool(client).run(query="rectified flow")

    assert result.ok
    assert "https://example.com/flow" in result.content
    assert len(result.data["results"]) == 3


def test_web_search_tool_fails_cleanly_with_no_results():
    client, _ = _client("<html><body>nothing here</body></html>")
    result = WebSearchTool(client).run(query="asdfghjkl")
    assert not result.ok and result.data["results"] == []


def test_web_fetch_tool_truncates_to_the_budget(monkeypatch):
    monkeypatch.setattr("model.tools.web.check_url", lambda url: url)
    client, _ = _client(PAGE_HTML)
    result = WebFetchTool(client).run(url="https://example.com/a", max_chars=20)

    assert result.ok and result.data["truncated"] is True
    assert result.content.endswith("[truncated]")
    assert result.data["chars"] == 20


def test_web_fetch_tool_needs_a_url():
    client, _ = _client(PAGE_HTML)
    assert not WebFetchTool(client).run(url="").ok


def test_web_fetch_tool_reports_an_unreachable_host():
    def transport(*_args):
        raise ToolError("could not reach https://example.com: offline")

    client = SearchClient(transport=transport)
    result = ToolRegistry([WebFetchTool(client)]).run("web_fetch", {"url": "https://example.com"})
    assert not result.ok and "offline" in result.content


@pytest.mark.parametrize(
    "url",
    ["ftp://example.com/x", "file:///etc/passwd", "https:///nohost", "http://127.0.0.1:8080/admin"],
)
def test_check_url_refuses_non_public_targets(url):
    with pytest.raises(ToolError):
        check_url(url)


def test_check_url_refuses_a_host_resolving_to_a_private_address(monkeypatch):
    monkeypatch.setattr("socket.getaddrinfo", lambda *_a, **_k: [(2, 1, 6, "", ("10.0.0.5", 0))])
    with pytest.raises(ToolError, match="10.0.0.5"):
        check_url("https://internal.example.com/")


def test_check_url_allows_a_public_address(monkeypatch):
    monkeypatch.setattr(
        "socket.getaddrinfo", lambda *_a, **_k: [(2, 1, 6, "", ("93.184.216.34", 0))]
    )
    assert check_url("https://example.com/page") == "https://example.com/page"

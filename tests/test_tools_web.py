"""Tool tests: the registry, the tool-calling loop, and the internet tools.

Nothing here touches the network. The client's transport is injected, so the
parsing, redirect unwrapping, address filtering, truncation, and loop control
are all exercised against fixed bytes.
"""

import io
import socket

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
    _DNS_PIN_CACHE,
    SearchClient,
    WebFetchTool,
    WebSearchTool,
    _build_safe_opener,
    _make_pinned_create,
    _PinnedHTTPConnection,
    _PinnedHTTPSHandler,
    _urllib_transport,
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


class _MockResponseSocket:
    """A mock socket that feeds scripted HTTP response bytes and tracks calls."""

    def __init__(self, response_bytes: bytes):
        self._raw = response_bytes
        self.sent: list[bytes] = []
        self.closed = False

    def setsockopt(self, *args, **kwargs):
        pass

    def getsockopt(self, *args, **kwargs):
        return socket.SOCK_STREAM

    def sendall(self, data: bytes):
        self.sent.append(data)

    def makefile(self, mode="r", *args, **kwargs):
        return io.BytesIO(self._raw)

    def close(self):
        self.closed = True


# --- Issue #237: redirect revalidation, address pinning, and SSRF tests ---


def test_safe_normal_request_urllib_transport(monkeypatch):
    """A. Safe normal request: standard valid public URL succeeds and retrieves page."""
    _DNS_PIN_CACHE.clear()
    monkeypatch.setattr(
        "socket.getaddrinfo", lambda *_a, **_k: [(2, 1, 6, "", ("93.184.216.34", 0))]
    )

    body = b"<html><head><title>Test Page</title></head><body>Hello world</body></html>"
    resp = b"HTTP/1.1 200 OK\r\nContent-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body

    connected: list[tuple[str, int]] = []

    def fake_connect(ip, port, timeout=None, source_address=None):
        connected.append((ip, port))
        return _MockResponseSocket(resp)

    monkeypatch.setattr("model.tools.web._connect_socket", fake_connect)

    data = _urllib_transport("http://example.com/test", None, 5.0, 1000)
    assert data == body
    assert connected == [("93.184.216.34", 80)]

    # Also test through WebFetchTool with default client
    client = SearchClient()
    result = WebFetchTool(client).run(url="http://example.com/test")
    assert result.ok
    assert "Test Page" in result.content
    assert "Hello world" in result.content


@pytest.mark.parametrize(
    "url",
    [
        "http://127.0.0.1:8080/admin",
        "http://169.254.169.254/latest/meta-data/",
        "http://10.0.0.1/",
        "http://172.16.0.1/",
        "http://192.168.1.1/",
        "ftp://example.com/x",
        "file:///etc/passwd",
        "http:///nohost",
        "http://[::1]/admin",
        "http://[fe80::1]/admin",
    ],
)
def test_blocked_initial_url_urllib_transport(url):
    """B. Blocked initial URL: private/loopback/link-local/scheme targets rejected."""
    _DNS_PIN_CACHE.clear()
    with pytest.raises(ToolError):
        _urllib_transport(url, None, 5.0, 1000)


def test_blocked_redirect_to_loopback(monkeypatch):
    """C. Blocked redirect: public URL redirecting to 127.0.0.1 must be rejected."""
    _DNS_PIN_CACHE.clear()

    def fake_gai(host, *args, **kwargs):
        if host == "example.com":
            return [(2, 1, 6, "", ("93.184.216.34", 0))]
        return [(2, 1, 6, "", ("127.0.0.1", 0))]

    monkeypatch.setattr("socket.getaddrinfo", fake_gai)

    connected: list[tuple[str, int]] = []

    def fake_connect(ip, port, timeout=None, source_address=None):
        connected.append((ip, port))
        resp = b"HTTP/1.1 302 Found\r\nLocation: http://127.0.0.1:8080/admin\r\n\r\n"
        return _MockResponseSocket(resp)

    monkeypatch.setattr("model.tools.web._connect_socket", fake_connect)

    with pytest.raises(ToolError, match="127.0.0.1"):
        _urllib_transport("http://example.com/start", None, 5.0, 1000)

    # 127.0.0.1 was NEVER connected to
    assert connected == [("93.184.216.34", 80)]


def test_blocked_redirect_to_cloud_metadata(monkeypatch):
    """C. Blocked redirect: redirect to 169.254.169.254 must be rejected."""
    _DNS_PIN_CACHE.clear()

    def fake_gai(host, *args, **kwargs):
        if host == "example.com":
            return [(2, 1, 6, "", ("93.184.216.34", 0))]
        return [(2, 1, 6, "", ("169.254.169.254", 0))]

    monkeypatch.setattr("socket.getaddrinfo", fake_gai)

    connected: list[tuple[str, int]] = []

    def fake_connect(ip, port, timeout=None, source_address=None):
        connected.append((ip, port))
        resp = b"HTTP/1.1 302 Found\r\nLocation: http://169.254.169.254/latest/meta-data/\r\n\r\n"
        return _MockResponseSocket(resp)

    monkeypatch.setattr("model.tools.web._connect_socket", fake_connect)

    with pytest.raises(ToolError, match="169.254.169.254"):
        _urllib_transport("http://example.com/start", None, 5.0, 1000)

    # Metadata service was NEVER connected to
    assert connected == [("93.184.216.34", 80)]


@pytest.mark.parametrize(
    "bad_location",
    [
        "ftp://example.com/file",
        "file:///etc/passwd",
        "javascript:alert(1)",
        "gopher://example.com/",
    ],
)
def test_blocked_redirect_to_non_http_schemes(bad_location, monkeypatch):
    """C. Blocked redirect: redirects to non-http/https schemes rejected."""
    _DNS_PIN_CACHE.clear()
    monkeypatch.setattr(
        "socket.getaddrinfo", lambda *_a, **_k: [(2, 1, 6, "", ("93.184.216.34", 0))]
    )

    def fake_connect(ip, port, timeout=None, source_address=None):
        resp = f"HTTP/1.1 302 Found\r\nLocation: {bad_location}\r\n\r\n".encode()
        return _MockResponseSocket(resp)

    monkeypatch.setattr("model.tools.web._connect_socket", fake_connect)

    with pytest.raises(ToolError, match="only http and https"):
        _urllib_transport("http://example.com/start", None, 5.0, 1000)


def test_multi_hop_redirect_success(monkeypatch):
    """D. Multi-hop redirect: public -> public -> public succeeds."""
    _DNS_PIN_CACHE.clear()

    hosts_map = {
        "example.com": "93.184.216.34",
        "example.org": "93.184.216.35",
        "example.net": "93.184.216.36",
    }
    monkeypatch.setattr(
        "socket.getaddrinfo",
        lambda host, *args, **kwargs: [(2, 1, 6, "", (hosts_map.get(host, "93.184.216.34"), 0))],
    )

    connected: list[tuple[str, int]] = []

    def fake_connect(ip, port, timeout=None, source_address=None):
        connected.append((ip, port))
        if ip == "93.184.216.34":
            return _MockResponseSocket(b"HTTP/1.1 302 Found\r\nLocation: http://example.org/2\r\n\r\n")
        elif ip == "93.184.216.35":
            return _MockResponseSocket(b"HTTP/1.1 301 Moved\r\nLocation: http://example.net/3\r\n\r\n")
        return _MockResponseSocket(b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\nfinal")

    monkeypatch.setattr("model.tools.web._connect_socket", fake_connect)

    data = _urllib_transport("http://example.com/1", None, 5.0, 1000)
    assert data == b"final"
    assert connected == [
        ("93.184.216.34", 80),
        ("93.184.216.35", 80),
        ("93.184.216.36", 80),
    ]


def test_multi_hop_redirect_blocked_at_unsafe_hop(monkeypatch):
    """D. Multi-hop redirect: public -> public -> private rejected at unsafe hop."""
    _DNS_PIN_CACHE.clear()

    hosts_map = {
        "example.com": "93.184.216.34",
        "example.org": "93.184.216.35",
        "private.local": "10.0.0.1",
    }
    monkeypatch.setattr(
        "socket.getaddrinfo",
        lambda host, *args, **kwargs: [(2, 1, 6, "", (hosts_map.get(host, "127.0.0.1"), 0))],
    )

    connected: list[tuple[str, int]] = []

    def fake_connect(ip, port, timeout=None, source_address=None):
        connected.append((ip, port))
        if ip == "93.184.216.34":
            return _MockResponseSocket(b"HTTP/1.1 302 Found\r\nLocation: http://example.org/2\r\n\r\n")
        elif ip == "93.184.216.35":
            return _MockResponseSocket(b"HTTP/1.1 302 Found\r\nLocation: http://private.local/secret\r\n\r\n")
        return _MockResponseSocket(b"HTTP/1.1 200 OK\r\n\r\nunsafe")

    monkeypatch.setattr("model.tools.web._connect_socket", fake_connect)

    with pytest.raises(ToolError, match="10.0.0.1"):
        _urllib_transport("http://example.com/1", None, 5.0, 1000)

    # Verified: Public 1 and Public 2 were requested, but private.local was NEVER connected
    assert connected == [
        ("93.184.216.34", 80),
        ("93.184.216.35", 80),
    ]


def test_redirect_limit_loop(monkeypatch):
    """E. Redirect limit: redirect loop terminates cleanly."""
    _DNS_PIN_CACHE.clear()
    monkeypatch.setattr(
        "socket.getaddrinfo", lambda *_a, **_k: [(2, 1, 6, "", ("93.184.216.34", 0))]
    )

    def fake_connect(ip, port, timeout=None, source_address=None):
        return _MockResponseSocket(b"HTTP/1.1 302 Found\r\nLocation: /loop\r\n\r\n")

    monkeypatch.setattr("model.tools.web._connect_socket", fake_connect)

    with pytest.raises(ToolError, match="redirect loop detected|too many redirects"):
        _urllib_transport("http://example.com/loop", None, 5.0, 1000)


def test_redirect_limit_excessive_chain(monkeypatch):
    """E. Redirect limit: redirect chain exceeding limit terminates cleanly."""
    _DNS_PIN_CACHE.clear()
    monkeypatch.setattr(
        "socket.getaddrinfo", lambda *_a, **_k: [(2, 1, 6, "", ("93.184.216.34", 0))]
    )

    hop = 0

    def fake_connect(ip, port, timeout=None, source_address=None):
        nonlocal hop
        hop += 1
        return _MockResponseSocket(
            f"HTTP/1.1 302 Found\r\nLocation: /step_{hop}\r\n\r\n".encode()
        )

    monkeypatch.setattr("model.tools.web._connect_socket", fake_connect)

    with pytest.raises(ToolError, match="too many redirects"):
        _urllib_transport("http://example.com/step_0", None, 5.0, 1000)


def test_relative_redirect(monkeypatch):
    """F. Relative redirect: relative Location header resolved correctly against current URL."""
    _DNS_PIN_CACHE.clear()
    monkeypatch.setattr(
        "socket.getaddrinfo", lambda *_a, **_k: [(2, 1, 6, "", ("93.184.216.34", 0))]
    )

    sockets: list[_MockResponseSocket] = []

    def fake_connect(ip, port, timeout=None, source_address=None):
        if not sockets:
            sock = _MockResponseSocket(b"HTTP/1.1 302 Found\r\nLocation: ../sibling/page.html\r\n\r\n")
        else:
            sock = _MockResponseSocket(b"HTTP/1.1 200 OK\r\nContent-Length: 7\r\n\r\nsuccess")
        sockets.append(sock)
        return sock

    monkeypatch.setattr("model.tools.web._connect_socket", fake_connect)

    data = _urllib_transport("http://example.com/dir/sub/index.html", None, 5.0, 1000)
    assert data == b"success"
    # Second request must have requested /dir/sibling/page.html
    second_request = b"".join(sockets[1].sent).decode()
    assert "GET /dir/sibling/page.html HTTP/1.1" in second_request


def test_dns_address_pinning_prevents_rebinding(monkeypatch):
    """G. DNS / Address pinning: socket connection uses address selected during validation.

    Resolver returns a safe public IP on initial validation, but if called a second time
    returns a private IP (DNS rebinding). The actual socket connection connects to the
    pinned public IP, and the resolver is NOT queried during socket connection.
    """
    _DNS_PIN_CACHE.clear()

    gai_queries: list[str] = []

    def fake_gai(host, *args, **kwargs):
        gai_queries.append(host)
        if len(gai_queries) == 1:
            return [(2, 1, 6, "", ("93.184.216.34", 0))]
        # DNS rebinding return
        return [(2, 1, 6, "", ("127.0.0.1", 0))]

    connected: list[tuple[str, int]] = []

    def fake_connect(ip, port, timeout=None, source_address=None):
        connected.append((ip, port))
        return _MockResponseSocket(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nOK")

    monkeypatch.setattr("socket.getaddrinfo", fake_gai)
    monkeypatch.setattr("model.tools.web._connect_socket", fake_connect)

    data = _urllib_transport("http://rebind.example.com/test", None, 5.0, 1000)
    assert data == b"OK"
    # Resolver was queried only once (during validation), NOT a second time during connect
    assert gai_queries == ["rebind.example.com"]
    # Connected directly to the pinned public address, never to 127.0.0.1
    assert connected == [("93.184.216.34", 80)]


def test_https_sni_and_hostname_preserved(monkeypatch):
    """H. HTTPS: TLS hostname/SNI and certificate validation use domain, socket uses pinned IP."""
    _DNS_PIN_CACHE.clear()
    monkeypatch.setattr(
        "socket.getaddrinfo", lambda *_a, **_k: [(2, 1, 6, "", ("93.184.216.34", 0))]
    )

    connected: list[tuple[str, int]] = []
    created_sockets: list[_MockResponseSocket] = []

    def fake_connect(ip, port, timeout=None, source_address=None):
        connected.append((ip, port))
        sock = _MockResponseSocket(b"HTTP/1.1 200 OK\r\nContent-Length: 5\r\n\r\nHTTPS")
        created_sockets.append(sock)
        return sock

    wrapped_hostnames: list[str | None] = []

    class MockSSLContext:
        def wrap_socket(self, sock, server_hostname=None):
            wrapped_hostnames.append(server_hostname)
            return sock

    opener = _build_safe_opener()
    for h in opener.handlers:
        if isinstance(h, _PinnedHTTPSHandler):
            h._context = MockSSLContext()

    monkeypatch.setattr("model.tools.web._connect_socket", fake_connect)
    monkeypatch.setattr("model.tools.web._build_safe_opener", lambda: opener)

    data = _urllib_transport("https://secure.example.com/page", None, 5.0, 1000)
    assert data == b"HTTPS"

    # Socket connected to the pinned IP on port 443
    assert connected == [("93.184.216.34", 443)]
    # wrap_socket received the original domain name for SNI and cert check
    assert wrapped_hostnames == ["secure.example.com"]
    # Request headers contain Host: secure.example.com
    request_headers = b"".join(created_sockets[0].sent).decode()
    assert "Host: secure.example.com" in request_headers


def test_ipv6_public_address_supported_and_pinned(monkeypatch):
    """IPv6: Public IPv6 address is supported and pinned, while IPv6 loopback is blocked."""
    _DNS_PIN_CACHE.clear()
    public_ipv6 = "2606:2800:220:1:248:1893:25c8:1946"

    def fake_gai(host, *args, **kwargs):
        if host == "::1":
            return [(10, 1, 6, "", ("::1", 0, 0, 0))]
        return [(10, 1, 6, "", (public_ipv6, 0, 0, 0))]

    monkeypatch.setattr("socket.getaddrinfo", fake_gai)

    connected: list[tuple[str, int]] = []

    def fake_connect(ip, port, timeout=None, source_address=None):
        connected.append((ip, port))
        return _MockResponseSocket(b"HTTP/1.1 200 OK\r\nContent-Length: 4\r\n\r\nIPv6")

    monkeypatch.setattr("model.tools.web._connect_socket", fake_connect)

    data = _urllib_transport("http://ipv6.example.com/test", None, 5.0, 1000)
    assert data == b"IPv6"
    assert connected == [(public_ipv6, 80)]

    # IPv6 loopback must be rejected
    with pytest.raises(ToolError, match="::1"):
        check_url("http://[::1]/admin")


def test_pinned_connection_fails_closed_without_hostname_resolution(monkeypatch):
    """Pinned connection path must fail closed and never fall back to hostname resolution."""

    def forbid_dns(*args, **kwargs):
        pytest.fail("socket.getaddrinfo was called during pinned connection!")

    monkeypatch.setattr("socket.getaddrinfo", forbid_dns)

    # 1. Unpinned connection with None must fail closed
    pinned_create_none = _make_pinned_create(None)
    with pytest.raises(OSError, match="refusing unpinned connection"):
        pinned_create_none(("example.com", 80))

    # 2. Unpinned connection with empty list must fail closed
    pinned_create_empty = _make_pinned_create([])
    with pytest.raises(OSError, match="refusing unpinned connection"):
        pinned_create_empty(("example.com", 80))

    # 3. Connection with non-IP target must fail closed rather than resolving
    pinned_create_hostname = _make_pinned_create(["example.com"])
    with pytest.raises(OSError, match="invalid pinned IP address"):
        pinned_create_hostname(("example.com", 80))

    # 4. _PinnedHTTPConnection without pinned_ips must also fail closed without DNS lookup
    conn = _PinnedHTTPConnection("example.com", 80, pinned_ips=None)
    with pytest.raises(OSError, match="refusing unpinned connection"):
        conn.connect()


def test_pinned_connection_tries_multiple_validated_addresses(monkeypatch):
    """Multiple validated public IPs are tried in order if earlier ones fail with OSError."""
    connected: list[str] = []

    def fake_connect(ip, port, timeout=None, source_address=None):
        connected.append(ip)
        if ip == "198.51.100.1":
            raise OSError("Connection refused on primary IP")
        return _MockResponseSocket(b"HTTP/1.1 200 OK\r\nContent-Length: 2\r\n\r\nOK")

    def forbid_dns(*args, **kwargs):
        pytest.fail("socket.getaddrinfo was called during connection!")

    monkeypatch.setattr("socket.getaddrinfo", forbid_dns)
    monkeypatch.setattr("model.tools.web._connect_socket", fake_connect)

    pinned_create = _make_pinned_create(["198.51.100.1", "198.51.100.2"])
    sock = pinned_create(("example.com", 80))
    assert sock is not None
    assert connected == ["198.51.100.1", "198.51.100.2"]

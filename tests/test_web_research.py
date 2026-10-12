"""Search providers, recency, and reading pages into cited passages."""

import json
import urllib.parse

import pytest

from model.tools import build_registry
from model.tools.base import ToolError
from model.tools.research import WebResearchTool
from model.tools.web import SearchClient


class Recorder:
    """A transport that answers from a table and records every request."""

    def __init__(self, pages):
        self.pages = pages
        self.requests = []

    def __call__(self, url, data, timeout, max_bytes, headers=None, trusted=False):
        self.requests.append({"url": url, "data": data, "headers": headers or {}, "trusted": trusted})
        for prefix, body in self.pages.items():
            if url.startswith(prefix):
                if isinstance(body, Exception):
                    raise body
                return body.encode() if isinstance(body, str) else body
        raise ToolError(f"unexpected url {url}")


DDG = """
<a class="result__a" href="https://a.example/one">Pump failure report</a>
<a class="result__snippet">snippet one</a>
<a class="result__a" href="https://b.example/two">Committee minutes</a>
<a class="result__snippet">snippet two</a>
"""

PAGE_A = "<html><head><title>Plant report</title></head><body>" + "".join(
    f"<p>Section {i}. Routine maintenance continued without incident.</p>" for i in range(30)
) + "<p>The reactor coolant pump failed on 3 March after a seal fault.</p></body></html>"

PAGE_B = "<html><body><p>The committee approved the budget.</p></body></html>"


# ── Providers ─────────────────────────────────────────────────────────────

def test_duckduckgo_sends_recency():
    transport = Recorder({"https://html.duckduckgo.com": DDG})
    SearchClient(transport=transport, provider="duckduckgo").search("pump", recency="week")
    fields = urllib.parse.parse_qs(transport.requests[0]["data"].decode())
    assert fields["df"] == ["w"]


def test_an_unknown_recency_is_refused():
    client = SearchClient(transport=Recorder({}), provider="duckduckgo")
    with pytest.raises(ToolError, match="recency"):
        client.search("x", recency="fortnight")


def test_searxng_is_trusted_and_parsed():
    payload = {"results": [{"title": "T", "url": "https://x.example", "content": "c"}]}
    transport = Recorder({"http://10.0.0.5:8080/search": json.dumps(payload)})
    client = SearchClient(transport=transport, provider="searxng", searxng_url="http://10.0.0.5:8080/")
    results = client.search("q", recency="month")
    assert [(r.title, r.url, r.snippet) for r in results] == [("T", "https://x.example", "c")]
    request = transport.requests[0]
    assert request["trusted"], "an operator-configured instance may sit on a private address"
    assert "time_range=month" in request["url"]


def test_brave_sends_its_key_and_strips_markup():
    payload = {"web": {"results": [{"title": "T", "url": "https://y.example", "description": "<b>bold</b> text"}]}}
    transport = Recorder({"https://api.search.brave.com": json.dumps(payload)})
    client = SearchClient(transport=transport, provider="brave", brave_api_key="k")
    results = client.search("q", recency="day")
    assert results[0].snippet == "bold text"
    request = transport.requests[0]
    assert request["headers"]["X-Subscription-Token"] == "k"
    assert "freshness=pd" in request["url"]
    assert not request["trusted"]


def test_providers_need_their_configuration(monkeypatch):
    monkeypatch.delenv("MODEL_SEARXNG_URL", raising=False)
    monkeypatch.delenv("MODEL_BRAVE_API_KEY", raising=False)
    with pytest.raises(ValueError, match="MODEL_SEARXNG_URL"):
        SearchClient(provider="searxng")
    with pytest.raises(ValueError, match="MODEL_BRAVE_API_KEY"):
        SearchClient(provider="brave")
    with pytest.raises(ValueError, match="provider"):
        SearchClient(provider="altavista")


def test_the_provider_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("MODEL_SEARCH_PROVIDER", "searxng")
    monkeypatch.setenv("MODEL_SEARXNG_URL", "http://search.local")
    assert SearchClient().provider == "searxng"


def test_malformed_json_is_a_tool_error():
    transport = Recorder({"http://s.local/search": "not json"})
    client = SearchClient(transport=transport, provider="searxng", searxng_url="http://s.local")
    with pytest.raises(ToolError, match="malformed"):
        client.search("q")


# ── Research ──────────────────────────────────────────────────────────────

def _research(pages=None):
    transport = Recorder(pages or {
        "https://html.duckduckgo.com": DDG,
        "https://a.example/one": PAGE_A,
        "https://b.example/two": PAGE_B,
    })
    client = SearchClient(transport=transport, provider="duckduckgo")
    # Fetches go through check_url, which resolves hosts; the fake hosts here
    # do not exist, so the check is bypassed for the test client only.
    client.fetch = _unchecked_fetch(client)
    return WebResearchTool(client), transport


def _unchecked_fetch(client):
    from model.tools.web import html_to_text

    def fetch(url, max_chars=4000):
        title, text = html_to_text(client._get(url))
        return title, text[:max_chars], len(text) > max_chars

    return fetch


def test_research_finds_the_passage_deep_in_a_page():
    tool, _ = _research()
    result = tool.run(query="when did the coolant pump fail", pages=2)
    assert result.ok
    assert "3 March" in result.content
    assert result.content.startswith("[1] Plant report\nhttps://a.example/one")
    assert result.data["sources"][0] == {"n": 1, "title": "Plant report", "url": "https://a.example/one"}
    assert "cite them as [n]" in result.content


def test_research_keeps_a_snippet_when_a_page_cannot_be_read():
    tool, _ = _research({
        "https://html.duckduckgo.com": DDG,
        "https://a.example/one": ToolError("blocked"),
        "https://b.example/two": PAGE_B,
    })
    result = tool.run(query="snippet one", pages=2)
    assert result.ok and "snippet one" in result.content


def test_research_limits_passages_per_source():
    tool, _ = _research()
    tool.passages = 10
    result = tool.run(query="routine maintenance section", pages=1)
    assert result.content.count("\n...\n") <= 2  # at most three excerpts from one page


def test_research_with_no_results_fails_cleanly():
    tool, _ = _research({"https://html.duckduckgo.com": "<html></html>"})
    result = tool.run(query="nothing")
    assert not result.ok and result.data["sources"] == []


def test_the_web_toolset_includes_research():
    assert "web_research" in build_registry("web").names()

"""Internet access, standard library only.

Two tools: ``web_search`` runs a query and returns ranked results, ``web_fetch``
retrieves one page and strips it to text. Both go through :class:`SearchClient`,
whose transport is injectable so the parsing, filtering, and truncation can be
tested without a socket.

The search endpoint is keyless and lives behind that one class, so it stays an
implementation detail: the tool names, the prompt, and the trace never name it,
and swapping it touches this module alone.

No API key, no account, and nothing added to requirements.txt - which is the
point: a clone of this repository can search the web as soon as it can run.
"""

import http.client
import ipaddress
import json
import re
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any

from .base import Tool, ToolError, ToolResult

# The keyless endpoints the default transport talks to. Nothing outside this
# module depends on which provider these point at.
SEARCH_URL = "https://html.duckduckgo.com/html/"
ANSWER_URL = "https://api.duckduckgo.com/"
USER_AGENT = "FramerAI/1.0 (+https://github.com/Spyxpo/framerai)"

DEFAULT_TIMEOUT = 10.0
DEFAULT_MAX_BYTES = 1_000_000
DEFAULT_MAX_CHARS = 4_000
DEFAULT_MAX_RESULTS = 5
DEFAULT_MAX_REDIRECTS = 10
DEFAULT_MAX_REDIRECT_REPEATS = 4
_DNS_PIN_CACHE_TTL = 5.0

_SKIP_TAGS = {"script", "style", "noscript", "head", "template", "svg"}
_WHITESPACE = re.compile(r"[ \t\r\f\v]+")
_BLANK_LINES = re.compile(r"\n{3,}")

Transport = Callable[[str, bytes | None, float, int], bytes]


@dataclass
class SearchResult:
    """One search hit."""

    title: str
    url: str
    snippet: str = ""

    def to_dict(self) -> dict[str, str]:
        return {"title": self.title, "url": self.url, "snippet": self.snippet}

    def render(self) -> str:
        line = f"{self.title}\n{self.url}"
        return f"{line}\n{self.snippet}" if self.snippet else line


def unwrap_redirect(href: str) -> str:
    """Turn a wrapped ``/l/?uddg=`` result link back into the real URL."""
    if not href:
        return ""
    if href.startswith("//"):
        href = "https:" + href
    parsed = urllib.parse.urlsplit(href)
    if parsed.path.startswith("/l/"):
        target = urllib.parse.parse_qs(parsed.query).get("uddg", [""])[0]
        if target:
            return urllib.parse.unquote(target)
    return href


_DNS_PIN_CACHE: dict[str, tuple[float, list[str]]] = {}


def _resolve_and_validate(
    url: str, use_cache: bool = True
) -> tuple[urllib.parse.SplitResult, list[str]]:
    """Reject anything that is not a public http(s) URL and return approved IP addresses."""
    parsed = urllib.parse.urlsplit(url)
    if parsed.scheme not in ("http", "https"):
        raise ToolError(f"only http and https URLs can be fetched, got {parsed.scheme or 'none'!r}")
    if not parsed.hostname:
        raise ToolError(f"{url!r} has no host")

    now = time.monotonic()
    if use_cache and url in _DNS_PIN_CACHE:
        ts, cached_ips = _DNS_PIN_CACHE[url]
        if now - ts < _DNS_PIN_CACHE_TTL:
            return parsed, cached_ips

    try:
        infos = socket.getaddrinfo(parsed.hostname, None)
    except socket.gaierror as exc:
        raise ToolError(f"could not resolve {parsed.hostname!r}: {exc}") from exc

    if not infos:
        raise ToolError(f"could not resolve {parsed.hostname!r}: no addresses returned")

    safe_ips: list[str] = []
    for info in infos:
        ip_str = info[4][0]
        address = ipaddress.ip_address(ip_str)
        if (
            address.is_private
            or address.is_loopback
            or address.is_link_local
            or address.is_reserved
            or address.is_multicast
            or address.is_unspecified
        ):
            raise ToolError(f"refusing to fetch {parsed.hostname!r}: resolves to {address}")
        if ip_str not in safe_ips:
            safe_ips.append(ip_str)

    _DNS_PIN_CACHE[url] = (now, safe_ips)
    if len(_DNS_PIN_CACHE) > 256:
        expired = [k for k, (t, _) in _DNS_PIN_CACHE.items() if now - t >= _DNS_PIN_CACHE_TTL]
        for k in expired:
            _DNS_PIN_CACHE.pop(k, None)

    return parsed, safe_ips


def check_url(url: str) -> str:
    """Reject anything that is not a public http(s) URL.

    A page the model chose to fetch must not be usable to reach the host's own
    network, so the hostname is resolved and private, loopback, link-local, and
    reserved addresses are refused before any request is made.
    """
    _resolve_and_validate(url)
    return url


def _connect_socket(
    ip_str: str,
    port: int,
    timeout: float = socket._GLOBAL_DEFAULT_TIMEOUT,
    source_address: tuple[str, int] | None = None,
) -> socket.socket:
    """Create and connect a TCP socket directly to an IP without hostname resolution."""
    addr = ipaddress.ip_address(ip_str)
    family = socket.AF_INET6 if addr.version == 6 else socket.AF_INET
    sock = socket.socket(family, socket.SOCK_STREAM)
    try:
        if timeout is not socket._GLOBAL_DEFAULT_TIMEOUT:
            sock.settimeout(timeout)
        if source_address:
            sock.bind(source_address)
        sa: tuple[Any, ...] = (ip_str, port, 0, 0) if addr.version == 6 else (ip_str, port)
        sock.connect(sa)
        return sock
    except Exception:
        sock.close()
        raise


def _make_pinned_create(pinned_ips: list[str] | None) -> Any:
    def _pinned_create(
        address: tuple[str, int],
        timeout: float = socket._GLOBAL_DEFAULT_TIMEOUT,
        source_address: tuple[str, int] | None = None,
    ) -> socket.socket:
        host, port = address
        if not pinned_ips:
            raise OSError(f"refusing unpinned connection to {host!r}: no validated IP addresses")
        exceptions: list[OSError] = []
        for target in pinned_ips:
            try:
                return _connect_socket(target, port, timeout, source_address)
            except ValueError as exc:
                exceptions.append(
                    OSError(f"invalid pinned IP address {target!r} for {host!r}: {exc}")
                )
            except OSError as err:
                exceptions.append(err)
        if exceptions:
            raise exceptions[0]
        raise OSError(f"could not connect to any validated address for {host}")

    return _pinned_create


class _PinnedHTTPConnection(http.client.HTTPConnection):
    """HTTP connection that ties socket creation to validated IP addresses."""

    def __init__(self, *args: Any, pinned_ips: list[str] | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.pinned_ips = pinned_ips
        self._create_connection = _make_pinned_create(pinned_ips)


class _PinnedHTTPSConnection(http.client.HTTPSConnection):
    """HTTPS connection that ties socket creation to validated IP addresses.

    The raw TCP socket is established to the pinned IP, while TLS SNI,
    certificate hostname verification, and HTTP Host headers all remain
    bound to the original target hostname.
    """

    def __init__(self, *args: Any, pinned_ips: list[str] | None = None, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.pinned_ips = pinned_ips
        self._create_connection = _make_pinned_create(pinned_ips)


class _PinnedHTTPHandler(urllib.request.HTTPHandler):
    """HTTP handler passing validated pinned IPs to _PinnedHTTPConnection."""

    def http_open(self, req: urllib.request.Request) -> http.client.HTTPResponse:
        def conn_factory(*args: Any, **kwargs: Any) -> _PinnedHTTPConnection:
            pinned_ips = getattr(req, "_pinned_ips", None)
            return _PinnedHTTPConnection(*args, pinned_ips=pinned_ips, **kwargs)

        return self.do_open(conn_factory, req)


class _PinnedHTTPSHandler(urllib.request.HTTPSHandler):
    """HTTPS handler passing validated pinned IPs to _PinnedHTTPSConnection."""

    def https_open(self, req: urllib.request.Request) -> http.client.HTTPResponse:
        def conn_factory(*args: Any, **kwargs: Any) -> _PinnedHTTPSConnection:
            pinned_ips = getattr(req, "_pinned_ips", None)
            return _PinnedHTTPSConnection(*args, pinned_ips=pinned_ips, **kwargs)

        return self.do_open(conn_factory, req, context=self._context)


class _SafeRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Redirect handler that validates every redirect target against SSRF rules."""

    max_redirections = DEFAULT_MAX_REDIRECTS
    max_repeats = DEFAULT_MAX_REDIRECT_REPEATS

    def http_error_302(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
    ) -> Any:
        location = headers.get("location") or headers.get("uri")
        if not location:
            return None

        # Resolve relative redirect against the current request URL
        target_url = urllib.parse.urljoin(req.full_url, location.strip())

        # Check redirect limits
        redirect_count = getattr(req, "redirect_count", 0) + 1
        if redirect_count > self.max_redirections:
            raise ToolError(f"too many redirects: exceeded limit of {self.max_redirections}")

        visited = dict(getattr(req, "redirect_dict", {}))
        if visited.get(target_url, 0) >= self.max_repeats:
            raise ToolError(f"redirect loop detected for {target_url}")
        visited[target_url] = visited.get(target_url, 0) + 1

        # Security check: validate the redirect target BEFORE following it
        try:
            _, safe_ips = _resolve_and_validate(target_url, use_cache=False)
        finally:
            if fp:
                fp.close()

        new_req = self.redirect_request(req, fp, code, msg, headers, target_url)
        if new_req is None:
            return None

        new_req._pinned_ips = safe_ips
        new_req.redirect_count = redirect_count
        new_req.redirect_dict = visited

        return self.parent.open(new_req, timeout=req.timeout)

    http_error_301 = http_error_303 = http_error_307 = http_error_308 = http_error_302


def _build_safe_opener() -> urllib.request.OpenerDirector:
    """Build an OpenerDirector with strict SSRF validation and address pinning."""
    opener = urllib.request.OpenerDirector()
    opener.add_handler(urllib.request.HTTPDefaultErrorHandler())
    opener.add_handler(_SafeRedirectHandler())
    opener.add_handler(_PinnedHTTPHandler())
    opener.add_handler(_PinnedHTTPSHandler())
    opener.add_handler(urllib.request.HTTPErrorProcessor())
    return opener


def _urllib_transport(url: str, data: bytes | None, timeout: float, max_bytes: int) -> bytes:
    """The real transport: one request, capped by time and by bytes."""
    _, safe_ips = _resolve_and_validate(url)
    request = urllib.request.Request(
        url,
        data=data,
        headers={
            "User-Agent": USER_AGENT,
            "Accept": "text/html,application/json;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9",
        },
        method="POST" if data else "GET",
    )
    request._pinned_ips = safe_ips
    request.redirect_count = 0
    request.redirect_dict = {url: 1}

    opener = _build_safe_opener()
    try:
        with opener.open(request, timeout=timeout) as response:
            return response.read(max_bytes)
    except ToolError:
        raise
    except urllib.error.HTTPError as exc:
        raise ToolError(f"{url} returned HTTP {exc.code}") from exc
    except (urllib.error.URLError, TimeoutError) as exc:
        # Offline is a supported state, not a crash: the model answers without
        # the tool and the trace says the tool failed.
        raise ToolError(f"could not reach {url}: {exc}") from exc


class _ResultParser(HTMLParser):
    """Pull titles, URLs, and snippets out of the results page."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.results: list[SearchResult] = []
        self._mode: str | None = None
        self._title: list[str] = []
        self._snippet: list[str] = []
        self._url = ""

    def handle_starttag(self, tag, attrs):
        if tag != "a":
            return
        attributes = dict(attrs)
        classes = (attributes.get("class") or "").split()
        if "result__a" in classes:
            self._flush()
            self._mode = "title"
            self._url = unwrap_redirect(attributes.get("href") or "")
        elif "result__snippet" in classes:
            self._mode = "snippet"

    def handle_endtag(self, tag):
        if tag == "a":
            self._mode = None

    def handle_data(self, data):
        if self._mode == "title":
            self._title.append(data)
        elif self._mode == "snippet":
            self._snippet.append(data)

    def _flush(self):
        title = _collapse("".join(self._title))
        if title and self._url:
            self.results.append(
                SearchResult(title=title, url=self._url, snippet=_collapse("".join(self._snippet)))
            )
        self._title, self._snippet, self._url = [], [], ""

    def close(self):
        super().close()
        self._flush()


class _TextExtractor(HTMLParser):
    """Everything a reader would see, and nothing a browser would run."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP_TAGS:
            self._skip += 1
        if tag == "title":
            self._in_title = True
        if tag in ("p", "br", "div", "li", "tr", "h1", "h2", "h3", "h4", "h5", "h6"):
            self.parts.append("\n")

    def handle_endtag(self, tag):
        if tag in _SKIP_TAGS and self._skip:
            self._skip -= 1
        if tag == "title":
            self._in_title = False

    def handle_data(self, data):
        if self._in_title and not self.title:
            self.title = _collapse(data)
        if not self._skip:
            self.parts.append(data)


def _collapse(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()


def html_to_text(html: str) -> tuple[str, str]:
    """Return ``(title, text)`` for a page, with markup and scripts removed."""
    parser = _TextExtractor()
    parser.feed(html)
    parser.close()
    lines = [_collapse(line) for line in "".join(parser.parts).splitlines()]
    text = _BLANK_LINES.sub("\n\n", "\n".join(line for line in lines if line))
    return parser.title, text.strip()


class SearchClient:
    """Search and page retrieval against the keyless search endpoints."""

    def __init__(
        self,
        timeout: float = DEFAULT_TIMEOUT,
        max_bytes: int = DEFAULT_MAX_BYTES,
        transport: Transport | None = None,
    ):
        self.timeout = timeout
        self.max_bytes = max_bytes
        self.transport = transport or _urllib_transport

    def _get(self, url: str, data: bytes | None = None) -> str:
        raw = self.transport(url, data, self.timeout, self.max_bytes)
        if isinstance(raw, bytes):
            return raw.decode("utf-8", errors="replace")
        return str(raw)

    def search(self, query: str, max_results: int = DEFAULT_MAX_RESULTS) -> list[SearchResult]:
        """Ranked results for a query, in the provider's own order."""
        query = (query or "").strip()
        if not query:
            raise ToolError("search needs a non-empty query")

        body = urllib.parse.urlencode({"q": query, "kl": "wt-wt"}).encode()
        parser = _ResultParser()
        parser.feed(self._get(SEARCH_URL, body))
        parser.close()
        return parser.results[: max(1, max_results)]

    def instant_answer(self, query: str) -> str:
        """The provider's own summary for a query, empty when there is none."""
        params = urllib.parse.urlencode(
            {"q": query, "format": "json", "no_html": "1", "skip_disambig": "1"}
        )
        try:
            payload = json.loads(self._get(f"{ANSWER_URL}?{params}"))
        except (json.JSONDecodeError, ToolError):
            return ""
        if not isinstance(payload, dict):
            return ""
        return _collapse(payload.get("AbstractText") or payload.get("Answer") or "")

    def fetch(self, url: str, max_chars: int = DEFAULT_MAX_CHARS) -> tuple[str, str, bool]:
        """Return ``(title, text, truncated)`` for one page."""
        check_url(url)
        title, text = html_to_text(self._get(url))
        if len(text) > max_chars:
            return title, text[:max_chars], True
        return title, text, False


class WebSearchTool(Tool):
    """Search the web."""

    name = "web_search"
    description = "Search the internet and return ranked results with URLs."
    parameters = {"query": "string", "max_results": f"integer, default {DEFAULT_MAX_RESULTS}"}

    def __init__(self, client: SearchClient | None = None, max_results: int = DEFAULT_MAX_RESULTS):
        self.client = client or SearchClient()
        self.max_results = max_results

    def run(self, query: str = "", max_results: int | None = None, **_: Any) -> ToolResult:
        limit = int(max_results or self.max_results)
        results = self.client.search(query, max_results=limit)
        if not results:
            return ToolResult.failure(f"no results for {query!r}", query=query, results=[])

        summary = self.client.instant_answer(query)
        body = "\n\n".join(result.render() for result in results)
        content = f"{summary}\n\n{body}" if summary else body
        return ToolResult.success(
            content,
            query=query,
            summary=summary,
            results=[result.to_dict() for result in results],
        )


class WebFetchTool(Tool):
    """Read one page."""

    name = "web_fetch"
    description = "Fetch a URL and return its readable text, truncated to a budget."
    parameters = {"url": "string", "max_chars": f"integer, default {DEFAULT_MAX_CHARS}"}

    def __init__(self, client: SearchClient | None = None, max_chars: int = DEFAULT_MAX_CHARS):
        self.client = client or SearchClient()
        self.max_chars = max_chars

    def run(self, url: str = "", max_chars: int | None = None, **_: Any) -> ToolResult:
        if not url:
            return ToolResult.failure("fetch needs a url")
        limit = int(max_chars or self.max_chars)
        title, text, truncated = self.client.fetch(url, max_chars=limit)
        if not text:
            return ToolResult.failure(f"{url} had no readable text", url=url)
        header = f"{title}\n{url}" if title else url
        suffix = "\n\n[truncated]" if truncated else ""
        return ToolResult.success(
            f"{header}\n\n{text}{suffix}",
            url=url,
            title=title,
            truncated=truncated,
            chars=len(text),
        )


def web_tools(client: SearchClient | None = None, **kwargs: Any) -> list[Tool]:
    """Both web tools sharing one client, so they share one timeout policy."""
    client = client or SearchClient(**kwargs)
    return [WebSearchTool(client), WebFetchTool(client)]

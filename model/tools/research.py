"""Search, read, and keep what answers the question, numbered for citation.

``web_search`` returns titles and one-line snippets, which is rarely enough to
answer from, and ``web_fetch`` returns one page from the top, which is rarely
where the answer is. A model with a small window cannot afford either habit:
reading three whole pages fills it with navigation and boilerplate.

``web_research`` does the reading for it. It searches, fetches the leading
pages in parallel, splits them into passages, ranks every passage from every
page against the question with BM25, and returns the best few grouped under
numbered sources. The model answers from evidence that fits, and cites ``[n]``
back to a URL the trace records, so a reader can check the claim.
"""

from concurrent.futures import ThreadPoolExecutor
from typing import Any

from ..retrieval import bm25_scores, split_passages
from .base import Tool, ToolError, ToolResult
from .web import SearchClient

DEFAULT_PAGES = 3
DEFAULT_PASSAGES = 6
PASSAGE_CHARS = 600
PAGE_CHARS = 20_000
PER_SOURCE = 3


class WebResearchTool(Tool):
    """Search, read the leading pages, and return cited passages."""

    name = "web_research"
    description = (
        "Search the internet, read the top pages, and return the passages most relevant "
        "to the question under numbered sources. Cite them as [n]."
    )
    parameters = {
        "query": "string",
        "pages": f"integer, default {DEFAULT_PAGES}",
        "recency": "optional: day, week, month or year",
    }

    def __init__(self, client: SearchClient | None = None, pages: int = DEFAULT_PAGES,
                 passages: int = DEFAULT_PASSAGES):
        self.client = client or SearchClient()
        self.pages = pages
        self.passages = passages

    def _read(self, result):
        try:
            title, text, _ = self.client.fetch(result.url, max_chars=PAGE_CHARS)
        except ToolError:
            title, text = "", ""
        # A page that cannot be read still has the snippet the search returned.
        return title or result.title, text or result.snippet

    def run(self, query: str = "", pages: int | None = None, recency: str | None = None,
            **_: Any) -> ToolResult:
        pages = max(1, min(int(pages or self.pages), 8))
        results = self.client.search(query, max_results=pages, recency=recency or None)
        if not results:
            return ToolResult.failure(f"no results for {query!r}", query=query, sources=[])

        with ThreadPoolExecutor(max_workers=len(results)) as pool:
            read = list(pool.map(self._read, results))

        candidates = []  # (source index, passage)
        for index, (_, text) in enumerate(read):
            for passage in split_passages(text, size=PASSAGE_CHARS):
                if passage.strip():
                    candidates.append((index, passage.strip()))
        if not candidates:
            return ToolResult.failure(f"nothing readable for {query!r}", query=query, sources=[])

        scores = bm25_scores([p for _, p in candidates], query)
        ranked = sorted(range(len(candidates)), key=lambda i: scores[i], reverse=True)

        chosen, per_source = [], {}
        for i in ranked:
            source = candidates[i][0]
            if per_source.get(source, 0) >= PER_SOURCE:
                continue
            chosen.append(i)
            per_source[source] = per_source.get(source, 0) + 1
            if len(chosen) >= self.passages:
                break

        # Number sources in the order they first matter, and keep each
        # source's passages in the order they appear on its page.
        order = []
        for i in chosen:
            if candidates[i][0] not in order:
                order.append(candidates[i][0])
        numbers = {source: n for n, source in enumerate(order, start=1)}

        blocks, sources = [], []
        for source in order:
            title = read[source][0]
            url = results[source].url
            excerpts = [candidates[i][1] for i in sorted(chosen) if candidates[i][0] == source]
            blocks.append(f"[{numbers[source]}] {title}\n{url}\n" + "\n...\n".join(excerpts))
            sources.append({"n": numbers[source], "title": title, "url": url})

        content = "\n\n".join(blocks) + "\n\nAnswer from these sources and cite them as [n]."
        return ToolResult.success(content, query=query, sources=sources)

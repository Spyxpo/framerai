"""Choosing what of a long document reaches the window.

A document longer than the window used to be cut at the budget, so a question
about its last chapter was answered from its first. Here the document is split
into passages, each is scored against the question with BM25, and the best are
kept until the budget is spent. They go back in document order, with a marker
where text was left out, so the model reads a coherent excerpt and can say that
it is one.

BM25 needs no model and no index built ahead of time, which keeps this usable
on every preset and every request. It is lexical: a question that shares no
words with the passage it needs will not find it, and in that case the opening
of the document is kept, as before.
"""

import math
import re
from collections import Counter

PASSAGE_CHARS = 1200
OMISSION = "[... omitted: not relevant to the question ...]"

_WORD = re.compile(r"[A-Za-z0-9_]+")
_STOP = frozenset(
    "a an and are as at be but by for from has have how i in is it its of on or that the "
    "this to was were what when where which who why will with you your does do did can".split()
)


def _terms(text: str) -> list[str]:
    return [w for w in (m.lower() for m in _WORD.findall(text)) if w not in _STOP and len(w) > 1]


def split_passages(text: str, size: int = PASSAGE_CHARS) -> list[str]:
    """Split on paragraph breaks, packing paragraphs up to ``size`` characters.

    A paragraph longer than ``size`` is cut on its own, so no passage exceeds it
    by more than one paragraph's tail.
    """
    passages, current = [], ""
    for para in re.split(r"\n\s*\n", text):
        while len(para) > size:
            if current:
                passages.append(current)
                current = ""
            passages.append(para[:size])
            para = para[size:]
        if current and len(current) + len(para) + 2 > size:
            passages.append(current)
            current = para
        else:
            current = f"{current}\n\n{para}" if current else para
    if current:
        passages.append(current)
    return passages


def bm25_scores(passages: list[str], query: str, k1: float = 1.5, b: float = 0.75) -> list[float]:
    """BM25 relevance of each passage to ``query``."""
    q_terms = set(_terms(query))
    docs = [_terms(p) for p in passages]
    if not q_terms or not docs:
        return [0.0] * len(passages)
    n = len(docs)
    avg = sum(len(d) for d in docs) / n or 1.0
    df = Counter(t for d in docs for t in set(d) if t in q_terms)
    scores = []
    for d in docs:
        tf = Counter(d)
        score = 0.0
        for t in q_terms:
            if not tf[t]:
                continue
            idf = math.log(1 + (n - df[t] + 0.5) / (df[t] + 0.5))
            score += idf * tf[t] * (k1 + 1) / (tf[t] + k1 * (1 - b + b * len(d) / avg))
        scores.append(score)
    return scores


def select_relevant(text: str, query: str, budget: int, keep_head: int = 1) -> tuple[str, bool]:
    """The most relevant passages of ``text`` that fit ``budget`` characters.

    Returns ``(excerpt, selected)``; ``selected`` is False when the text already
    fit or the question matched nothing, and the excerpt is then the opening of
    the text. The first ``keep_head`` passages are always kept, because a
    document's opening usually says what it is.
    """
    if len(text) <= budget:
        return text, False
    # Several passages have to fit, or choosing among them means nothing.
    passages = split_passages(text, size=min(PASSAGE_CHARS, max(budget // 4, 64)))
    scores = bm25_scores(passages, query)
    if not any(scores):
        return text[:budget], False

    chosen, used = set(), 0
    for i in range(min(keep_head, len(passages))):
        if used + len(passages[i]) <= budget:
            chosen.add(i)
            used += len(passages[i]) + 2
    ranked = sorted(range(len(passages)), key=lambda i: scores[i], reverse=True)
    for i in ranked:
        if scores[i] <= 0:
            break
        cost = len(passages[i]) + 2 + len(OMISSION) + 2
        if i not in chosen and used + cost <= budget:
            chosen.add(i)
            used += cost

    parts, last = [], -1
    for i in sorted(chosen):
        if i != last + 1:
            parts.append(OMISSION)
        parts.append(passages[i])
        last = i
    if last != len(passages) - 1:
        parts.append(OMISSION)
    return "\n\n".join(parts), True

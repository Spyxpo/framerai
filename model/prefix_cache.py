"""Reuse of a prompt's KV cache across requests.

A conversation resends every earlier turn on each new one, and a system prompt
or an attached document is the same tokens every time. Prefilling those again
is the bulk of what a long chat costs, and it is work whose answer is already
known. This keeps the caches of recent prompts and, for a new prompt, starts
from the longest prefix it shares with one of them, so only the new tokens run
through the backbone.

The reused positions are reported as cached input, which is what lets the
pricing card charge them at a fraction of a fresh token: the discount follows
work that was actually skipped.

Only text prompts are cached. A prompt carrying an image or audio prefix has
positions that are not tokens, and two such prompts with identical ids can
differ in what those positions hold.
"""

from collections import OrderedDict

from .modules.kv_cache import fork_cache

DEFAULT_MAX_ENTRIES = 16
DEFAULT_MAX_TOKENS = 65536


def _common_prefix(a: tuple, b: list) -> int:
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


class PrefixCache:
    """A bounded, least-recently-used store of prompt caches."""

    def __init__(self, max_entries: int = DEFAULT_MAX_ENTRIES, max_tokens: int = DEFAULT_MAX_TOKENS,
                 min_match: int = 16):
        if max_entries < 1:
            raise ValueError(f"max_entries must be positive, got {max_entries}")
        self.max_entries = int(max_entries)
        self.max_tokens = int(max_tokens)
        # Below this a match saves less than the copy costs.
        self.min_match = int(min_match)
        self._entries = OrderedDict()
        self.hits = 0
        self.misses = 0

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def tokens_held(self) -> int:
        return sum(len(key) for key in self._entries)

    def lookup(self, tokens: list):
        """The longest reusable prefix of ``tokens`` and a private copy of its cache.

        Returns ``(0, None)`` on a miss. At least one token is always left to
        run, because generation needs the logits of the final position.
        """
        best_key, best = None, 0
        for key in self._entries:
            match = _common_prefix(key, tokens)
            if match > best:
                best_key, best = key, match
        best = min(best, len(tokens) - 1)
        if best_key is None or best < self.min_match:
            self.misses += 1
            return 0, None
        self._entries.move_to_end(best_key)
        self.hits += 1
        return best, fork_cache(self._entries[best_key], best)

    def store(self, tokens: list, past_kvs: list):
        """Keep a copy of a prompt's cache, evicting the oldest past the bounds."""
        if len(tokens) < self.min_match or len(tokens) > self.max_tokens:
            return
        key = tuple(tokens)
        # A prompt that extends a held one replaces it: the longer entry
        # serves every request the shorter one could.
        for held in list(self._entries):
            if len(held) <= len(key) and key[: len(held)] == held:
                del self._entries[held]
        self._entries[key] = fork_cache(past_kvs, len(key))
        while len(self._entries) > self.max_entries or self.tokens_held > self.max_tokens:
            self._entries.popitem(last=False)

    def clear(self):
        self._entries.clear()

"""Long-context retrieval: does the window hold anything, and where.

The three largest presets declare a 1,048,576-token context and nothing
measured whether the model can retrieve from it. Perplexity does not answer
this: most tokens in a long document are locally predictable, so it stays low
while retrieval fails completely, reporting success where the capability is
absent.

Three suites, each a forced choice between the right answer and distractors
scored by the model's own likelihood. A choice is scored rather than generated
because generation would confound retrieval with instruction following, and
because a forced choice has a known chance rate: with four options an untrained
model sits at 0.25, so a number above it means something.

- ``single_fact``   one fact at a controlled depth, swept across the window, so
  position-dependence is visible rather than averaged away.
- ``multi_hop``     two facts that must be combined, which a single lookup
  cannot answer.
- ``aggregation``   a question over the whole context, which no retrieval
  shortcut can answer.

The material is generated from a seed, so the suites need no external data and
the same run reproduces. Lengths come from the model's own ``max_seq_len``, so
``framer-tiny`` exercises the same code at 1024 tokens that a flagship does at
a million.
"""

import random

import torch
import torch.nn.functional as F

# Filler that is grammatical, repetitive, and carries no fact, so anything the
# model retrieves came from a needle rather than from the haystack.
FILLER = [
    "The maintenance log records routine activity for the period under review.",
    "Operations continued without incident during the reporting window.",
    "No deviation from the published schedule was observed on this shift.",
    "The duty officer confirmed the readings fell inside the expected band.",
    "Consumption held steady against the forecast for the same period.",
    "Inspection found the seals intact and the fittings correctly torqued.",
]

SUBJECTS = ["the north gate", "the archive room", "the west pump", "the relay hut",
            "the survey office", "the cold store", "the signal mast", "the boat shed"]

CHOICES = 4
DEFAULT_DEPTHS = (0.0, 0.25, 0.5, 0.75, 1.0)

# A case is filler plus a needle, a question and an option, and only the filler
# is asked for by length. This is the room the rest needs, so a requested bucket
# can be clamped to something that fits the window instead of walking off the
# end of it.
CASE_OVERHEAD_TOKENS = 96


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def _score_continuation(model, prefix_ids, continuation_ids, device, chunk: int = 4096) -> float:
    """Mean log-probability of a continuation, given a prefix.

    The prefix is walked through the KV cache in chunks rather than run in one
    forward, which is what makes a bucket of a million tokens possible at all:
    a single forward would materialise a logits row per position before scoring
    a single one of them. The score is per token so options of different
    lengths compare fairly.
    """
    if not continuation_ids:
        raise ValueError("a continuation needs at least one token")

    past, logits = None, None
    for start in range(0, len(prefix_ids), chunk):
        piece = torch.tensor([prefix_ids[start:start + chunk]], device=device)
        out = model.forward_lm(piece, past_kvs=past, use_cache=True)
        past, logits = out["past_kvs"], out["logits"][:, -1, :]

    total = 0.0
    for token in continuation_ids:
        total += float(F.log_softmax(logits[0], dim=-1)[token])
        out = model.forward_lm(
            torch.tensor([[token]], device=device), past_kvs=past, use_cache=True
        )
        past, logits = out["past_kvs"], out["logits"][:, -1, :]
    return total / len(continuation_ids)


def _pick(model, tokenizer, prefix: str, options: list, device, chunk: int = 4096) -> int:
    """Index of the option the model finds most likely after ``prefix``.

    A window too small to hold even a minimal case is reported as such. The
    harness records the reason and moves on, which is the right outcome: a
    suite that cannot run is not the same as one that scored badly, and the
    alternative here is a position error from deep inside attention.
    """
    prefix_ids = tokenizer.encode(prefix, add_special=False)
    longest = max(len(tokenizer.encode(option, add_special=False)) for option in options)
    window = int(getattr(model.config, "max_seq_len", 0) or 0)
    if window and len(prefix_ids) + longest > window:
        raise ValueError(
            f"a {window}-token context window cannot hold a retrieval case, "
            f"which needs {len(prefix_ids) + longest} tokens at the shortest "
            f"length this suite builds. Use a preset with a larger window."
        )

    scores = [
        _score_continuation(
            model, prefix_ids, tokenizer.encode(option, add_special=False), device, chunk
        )
        for option in options
    ]
    return max(range(len(scores)), key=scores.__getitem__)


# ---------------------------------------------------------------------------
# Building the material
# ---------------------------------------------------------------------------

def _filler_to_length(tokenizer, target_tokens: int, rng) -> list:
    """Filler sentences totalling at most ``target_tokens`` tokens.

    Measured rather than estimated, and it stops below the target rather than
    stepping over it, because overshooting is what pushes a case past the
    window it was meant to fit inside.
    """
    costs = {sentence: len(tokenizer.encode(sentence, add_special=False)) + 1 for sentence in FILLER}
    sentences, total = [], 0
    while True:
        sentence = rng.choice(FILLER)
        if total + costs[sentence] > target_tokens:
            break
        sentences.append(sentence)
        total += costs[sentence]
    return sentences


def usable_length(model, requested: int, overhead: int = CASE_OVERHEAD_TOKENS) -> int:
    """Clamp a requested filler length to what the model's window can hold."""
    window = int(getattr(model.config, "max_seq_len", 0) or 0)
    if not window:
        return requested
    return max(16, min(requested, window - overhead))


def build_single_fact(tokenizer, length_tokens: int, depth: float, seed: int = 0) -> dict:
    """A haystack with one fact buried at ``depth`` through it."""
    rng = random.Random(seed)
    subject = rng.choice(SUBJECTS)
    values = rng.sample(range(1000, 9999), CHOICES)
    answer = values[0]

    needle = f"The access code for {subject} is {answer}."
    sentences = _filler_to_length(tokenizer, length_tokens, rng)
    at = min(len(sentences), max(0, round(depth * len(sentences))))
    sentences.insert(at, needle)

    return {
        "prefix": " ".join(sentences) + f"\n\nThe access code for {subject} is",
        "options": [f" {value}." for value in values],
        "answer": 0,
    }


def build_multi_hop(tokenizer, length_tokens: int, seed: int = 0) -> dict:
    """Two facts, placed apart, that only answer the question together."""
    rng = random.Random(seed + 1)
    subject = rng.choice(SUBJECTS)
    code = rng.randrange(1000, 9999)
    holders = rng.sample(["Avery", "Blake", "Corin", "Devi", "Emeka", "Farid"], CHOICES)

    sentences = _filler_to_length(tokenizer, length_tokens, rng)
    # A quarter and three quarters in, so neither fact is at an edge and the
    # model has to hold the first while it reads to the second.
    sentences.insert(max(0, len(sentences) // 4), f"The access code for {subject} is {code}.")
    sentences.insert(
        min(len(sentences), 3 * len(sentences) // 4),
        f"Code {code} is held by {holders[0]}.",
    )

    return {
        "prefix": " ".join(sentences) + f"\n\nThe access code for {subject} is held by",
        "options": [f" {holder}." for holder in holders],
        "answer": 0,
    }


def build_aggregation(tokenizer, length_tokens: int, seed: int = 0) -> dict:
    """A count over the whole context, which no single lookup answers."""
    rng = random.Random(seed + 2)
    sentences = _filler_to_length(tokenizer, length_tokens, rng)
    occurrences = rng.randrange(3, 7)

    positions = sorted(rng.sample(range(len(sentences)), min(occurrences, len(sentences))))
    for offset, position in enumerate(positions):
        sentences.insert(position + offset, "An alarm was raised at the west pump.")
    true_count = len(positions)

    wrong = [n for n in range(1, 10) if n != true_count]
    options = [true_count] + rng.sample(wrong, CHOICES - 1)

    return {
        "prefix": " ".join(sentences) + "\n\nThe number of alarms raised at the west pump was",
        "options": [f" {count}." for count in options],
        "answer": 0,
    }


def length_buckets(max_seq_len: int, smallest: int = 256, count: int = 4) -> list:
    """Powers of four up to the window, so a sweep spans it without crowding.

    The largest bucket leaves room for the question and the options, which is
    why it is a fraction of the window rather than the window itself.
    """
    usable = max(smallest, int(max_seq_len * 0.75))
    buckets, length = [], smallest
    while length <= usable and len(buckets) < count:
        buckets.append(length)
        length *= 4
    if usable not in buckets and usable > (buckets[-1] if buckets else 0):
        buckets.append(usable)
    return buckets


# ---------------------------------------------------------------------------
# Suites
# ---------------------------------------------------------------------------

def single_fact_accuracy(model, tokenizer, device="cpu", lengths=None, depths=DEFAULT_DEPTHS,
                         seed: int = 0, chunk: int = 4096) -> dict:
    """Retrieval accuracy per length, and per depth within each length."""
    lengths = lengths or length_buckets(model.config.max_seq_len)
    per_length, per_depth = {}, {}

    for length in lengths:
        hits = 0
        budget = usable_length(model, length)
        for index, depth in enumerate(depths):
            case = build_single_fact(tokenizer, budget, depth, seed + index)
            correct = _pick(model, tokenizer, case["prefix"], case["options"], device, chunk) == case["answer"]
            hits += int(correct)
            per_depth.setdefault(f"depth_{depth:g}", []).append(int(correct))
        per_length[f"len_{length}"] = hits / len(depths)

    values = dict(per_length)
    # Depth is reported separately because a window that only holds its ends is
    # a different failure from one that holds nothing, and averaging hides it.
    for depth, results in per_depth.items():
        values[depth] = sum(results) / len(results)
    values["accuracy"] = sum(per_length.values()) / len(per_length)
    values["chance"] = 1.0 / CHOICES
    return values


def multi_hop_accuracy(model, tokenizer, device="cpu", lengths=None, seed: int = 0,
                       chunk: int = 4096) -> dict:
    lengths = lengths or length_buckets(model.config.max_seq_len)
    values, hits = {}, 0
    for index, length in enumerate(lengths):
        case = build_multi_hop(tokenizer, usable_length(model, length), seed + index)
        correct = _pick(model, tokenizer, case["prefix"], case["options"], device, chunk) == case["answer"]
        values[f"len_{length}"] = float(correct)
        hits += int(correct)
    values["accuracy"] = hits / len(lengths)
    values["chance"] = 1.0 / CHOICES
    return values


def aggregation_accuracy(model, tokenizer, device="cpu", lengths=None, seed: int = 0,
                         chunk: int = 4096) -> dict:
    lengths = lengths or length_buckets(model.config.max_seq_len)
    values, hits = {}, 0
    for index, length in enumerate(lengths):
        case = build_aggregation(tokenizer, usable_length(model, length), seed + index)
        correct = _pick(model, tokenizer, case["prefix"], case["options"], device, chunk) == case["answer"]
        values[f"len_{length}"] = float(correct)
        hits += int(correct)
    values["accuracy"] = hits / len(lengths)
    values["chance"] = 1.0 / CHOICES
    return values

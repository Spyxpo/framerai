"""Tests for the cognition layer's memory stores.

Covers the properties that separate a memory from a database: traces decay with
disuse, recall strengthens what it returns, the weakest trace is what gets
evicted under pressure, retrieval blends similarity with recency and salience,
and repeated episodes collapse into concepts.

CPU-only, seeded, no checkpoint required.
"""

import pytest
import torch

from model.cognition import (
    CognitionConfig,
    EpisodicMemory,
    Experience,
    SemanticMemory,
    WorkingMemory,
    hash_embed,
)
from model.cognition.memory import default_salience, half_life_decay


def _config(**overrides) -> CognitionConfig:
    base = dict(d_embed=32, episodic_capacity=8, working_memory_size=3, semantic_capacity=4)
    base.update(overrides)
    return CognitionConfig(**base).validate()


def _write(memory, text, tick, **kwargs):
    exp = Experience(tick=tick, text=text, **kwargs)
    return memory.write(hash_embed(text, memory.d_embed), exp)


# ---------------------------------------------------------------------------
# episodic memory
# ---------------------------------------------------------------------------

def test_write_and_retrieve_returns_the_relevant_episode():
    memory = EpisodicMemory(_config())
    _write(memory, "rectified flow trains straight paths", tick=1)
    _write(memory, "coral reefs bleach in warm water", tick=2)

    hits = memory.retrieve(hash_embed("rectified flow paths", 32), now=3, k=1)
    assert len(hits) == 1
    assert "rectified flow" in hits[0].experience.text


def test_retrieval_strengthens_the_recalled_trace():
    memory = EpisodicMemory(_config())
    exp = _write(memory, "the vocoder inverts mel spectrograms", tick=1)
    before = exp.strength

    memory.retrieve(hash_embed("vocoder mel", 32), now=2, k=1)
    assert exp.strength > before
    assert exp.uses == 1
    assert exp.last_used == 2


def test_unused_traces_decay_by_half_life():
    config = _config(strength_half_life=100.0)
    memory = EpisodicMemory(config)
    exp = _write(memory, "a fact nobody ever asks about", tick=0)

    assert memory.trace_strength(exp, now=100) == pytest.approx(0.5, rel=1e-6)
    assert memory.trace_strength(exp, now=200) == pytest.approx(0.25, rel=1e-6)


def test_eviction_drops_the_weakest_trace_not_the_oldest():
    """A striking old memory outlives a bland recent one when capacity binds."""
    memory = EpisodicMemory(_config(episodic_capacity=3, retrieval_k=2))
    striking = _write(memory, "the day everything changed", tick=1, novelty=1.0, surprise=1.0)
    _write(memory, "filler one", tick=2)
    _write(memory, "filler two", tick=3)

    _write(memory, "filler three", tick=4)  # forces an eviction
    kept = [e.text for e in memory.all()]
    assert striking.text in kept
    assert len(memory) == 3


def test_salience_rises_with_novelty_surprise_and_reward_magnitude():
    assert default_salience(0.0, 0.0, 0.0) == 0.0
    assert default_salience(1.0, 1.0, 1.0) == 1.0
    # Punishment is as memorable as reward: magnitude, not sign.
    assert default_salience(0.0, 0.0, -0.8) == default_salience(0.0, 0.0, 0.8)


def test_recency_and_salience_can_outrank_a_closer_match():
    """Retrieval is not pure nearest-neighbour, or nothing would ever fade."""
    config = _config(recency_half_life=5.0, w_similarity=1.0, w_recency=1.0, w_salience=1.0)
    memory = EpisodicMemory(config)
    _write(memory, "diffusion sampling steps", tick=1)  # older, plainer
    recent = _write(memory, "diffusion sampling", tick=100, novelty=1.0, surprise=1.0)

    hits = memory.retrieve(hash_embed("diffusion sampling steps", 32), now=101, k=1)
    assert hits[0].experience is recent


def test_retrieval_floor_rejects_unrelated_cues():
    memory = EpisodicMemory(_config(retrieval_floor=0.9))
    _write(memory, "residual vector quantization", tick=1)
    assert memory.retrieve(hash_embed("marine biology", 32), now=2) == []


def test_decay_all_forgets_weak_traces_and_reindexes():
    memory = EpisodicMemory(_config(strength_half_life=1.0))
    _write(memory, "forgettable", tick=0)
    kept = _write(memory, "worth keeping", tick=0, novelty=1.0, surprise=1.0)

    forgotten = memory.decay_all(factor=0.5, now=50)
    assert forgotten == 1
    assert [e.text for e in memory.all()] == [kept.text]
    # Slots are compacted, so the surviving embedding still matches its episode.
    assert memory.all()[0].slot == 0
    hits = memory.retrieve(hash_embed("worth keeping", 32), now=51, k=1)
    assert hits and hits[0].similarity > 0.9


def test_prioritised_sample_is_bounded_and_seeded():
    memory = EpisodicMemory(_config())
    for i in range(5):
        _write(memory, f"episode {i}", tick=i, novelty=i / 5)

    generator = torch.Generator().manual_seed(0)
    batch = memory.sample(3, now=6, alpha=0.7, generator=generator)
    assert len(batch) == 3
    assert len({id(e) for e in batch}) == 3  # sampled without replacement


def test_write_rejects_a_wrong_width_embedding():
    memory = EpisodicMemory(_config())
    with pytest.raises(ValueError, match="32-d embedding"):
        memory.write(torch.zeros(16), Experience(tick=1, text="wrong shape"))


def test_state_dict_round_trip_preserves_recall():
    memory = EpisodicMemory(_config())
    _write(memory, "spacetime diffusion transformer", tick=1, novelty=0.5)
    state = memory.state_dict()

    restored = EpisodicMemory(_config()).load_state_dict(state)
    hits = restored.retrieve(hash_embed("spacetime transformer", 32), now=2, k=1)
    assert hits and hits[0].experience.text == "spacetime diffusion transformer"


# ---------------------------------------------------------------------------
# semantic memory
# ---------------------------------------------------------------------------

def test_similar_episodes_collapse_into_one_concept():
    config = _config(concept_threshold=0.5)
    semantic = SemanticMemory(config)
    text = "rectified flow trains straight paths"

    _, first_is_new = semantic.integrate(hash_embed(text, 32), Experience(tick=1, text=text))
    concept, second_is_new = semantic.integrate(
        hash_embed(text + " between noise and data", 32), Experience(tick=2, text=text)
    )
    assert first_is_new and not second_is_new
    assert concept.visits == 2
    assert len(semantic) == 1


def test_dissimilar_episodes_start_separate_concepts():
    semantic = SemanticMemory(_config(concept_threshold=0.9))
    semantic.integrate(hash_embed("audio codec", 32), Experience(tick=1, text="audio codec"))
    semantic.integrate(hash_embed("coral reef", 32), Experience(tick=2, text="coral reef"))
    assert len(semantic) == 2


def test_concept_exemplar_is_the_most_striking_member():
    semantic = SemanticMemory(_config(concept_threshold=0.3))
    text = "the vocoder inverts mel spectrograms"
    semantic.integrate(
        hash_embed(text, 32), Experience(tick=1, text="bland version", salience=0.1)
    )
    concept, _ = semantic.integrate(
        hash_embed(text, 32), Experience(tick=2, text="striking version", salience=0.9)
    )
    assert concept.exemplar == "striking version"


def test_semantic_recall_ranks_by_similarity():
    semantic = SemanticMemory(_config(concept_threshold=0.99))
    semantic.integrate(hash_embed("audio codec", 32), Experience(tick=1, text="audio", topic="audio"))
    semantic.integrate(hash_embed("coral reef", 32), Experience(tick=2, text="reef", topic="reef"))

    ranked = semantic.recall(hash_embed("audio codec", 32), k=2)
    assert ranked[0][0].label == "audio"
    assert ranked[0][1] > ranked[1][1]


# ---------------------------------------------------------------------------
# working memory
# ---------------------------------------------------------------------------

def test_working_memory_is_bounded_and_keeps_the_salient():
    working = WorkingMemory(_config(working_memory_size=2))
    working.push(Experience(tick=1, text="striking", salience=0.9))
    working.push(Experience(tick=2, text="bland", salience=0.0))
    working.push(Experience(tick=3, text="newest", salience=0.1))

    texts = [e.text for e in working.items()]
    assert len(texts) == 2
    assert "striking" in texts and "newest" in texts


def test_working_memory_render_respects_its_budget():
    working = WorkingMemory(_config())
    for i in range(3):
        working.push(Experience(tick=i, text="x" * 50))
    assert len(working.render(budget=60)) <= 60


def test_half_life_decay_rejects_a_non_positive_half_life():
    with pytest.raises(ValueError):
        half_life_decay(1.0, 0.0)

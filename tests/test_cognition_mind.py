"""Tests for the Mind tick loop, consolidation, and persistence.

These cover the properties that make the layer a mind rather than a cache: a cue
recalls the past and not itself, repetition habituates, effort accumulates until
sleep consolidates episodes into concepts, the mind's own output is remembered as
something it did, and the whole state survives a restart.

The generator-backed tests build a tiny text-only model on CPU, so they need no
checkpoint and run in seconds.
"""

import pytest
import torch

from conftest import tiny_config
from model.cognition import CognitionConfig, Mind
from model.cognition.encoder import ExperienceEncoder, hash_embed
from model.framer import FramerModel
from model.generate import FramerGenerator
from model.tokenizer import FramerTokenizer

_CORPUS = [
    "rectified flow trains straight paths between noise and data",
    "the vocoder inverts mel spectrograms into waveforms",
    "hello world the quick brown fox",
]


def _mind(**overrides) -> Mind:
    base = dict(
        d_embed=32, episodic_capacity=32, retrieval_k=3, working_memory_size=4,
        semantic_capacity=8, novelty_hidden=32, replay_batch=4,
    )
    base.update(overrides)
    return Mind(CognitionConfig(**base))


def _generator() -> FramerGenerator:
    tokenizer = FramerTokenizer(vocab_size=300)
    tokenizer.train(_CORPUS, target_vocab_size=300)
    config = tiny_config(vocab_size=tokenizer.vocab_size, max_seq_len=64)
    return FramerGenerator(FramerModel(config), tokenizer, device="cpu")


# ---------------------------------------------------------------------------
# the tick loop
# ---------------------------------------------------------------------------

def test_perceive_records_an_episode_and_returns_a_trace():
    mind = _mind()
    trace = mind.perceive("rectified flow trains straight paths")

    assert mind.tick == 1
    assert len(mind.episodic) == 1
    assert trace.kind == "perception"
    assert trace.topic in trace.text
    assert 0.0 <= trace.novelty <= 1.0
    assert trace.feeling


def test_recall_happens_before_storage_so_a_cue_never_returns_itself():
    mind = _mind()
    mind.perceive("the vocoder inverts mel spectrograms")
    trace = mind.perceive("the vocoder inverts mel spectrograms")

    assert len(trace.recalled) == 1
    assert trace.recalled[0]["tick"] == 1


def test_a_first_experience_recalls_nothing():
    assert _mind().perceive("something entirely new").recalled == []


def test_repetition_habituates_and_a_new_subject_reawakens():
    mind = _mind(sleep_threshold=1.0)
    first = mind.perceive("residual vector quantization of audio")
    for _ in range(8):
        mind.perceive("residual vector quantization of audio")
    repeated = mind.perceive("residual vector quantization of audio")
    fresh = mind.perceive("coral reefs bleach when the water warms")

    assert repeated.novelty < first.novelty
    assert fresh.novelty > repeated.novelty


def test_topic_inference_picks_a_content_word():
    assert Mind.infer_topic("what is the vocoder for") == "vocoder"
    assert Mind.infer_topic("a b c") == "general"


def test_explicit_topics_keep_learning_progress_separate():
    mind = _mind(sleep_threshold=1.0)
    for i in range(8):
        mind.perceive(f"audio fact {i}", topic="audio")
    mind.perceive("one image fact", topic="vision")

    topics = set(mind.curiosity.progress.topics())
    assert {"audio", "vision"} <= topics
    # The same events are also filed by sense, on their own axis.
    assert "sense:text" in topics
    assert mind.self_model.competence["audio"] > 0.0


def test_reward_moves_valence_in_the_right_direction():
    good, bad = _mind(), _mind()
    good.perceive("a claim", reward=1.0)
    bad.perceive("a claim", reward=-1.0)
    assert good.affect.valence > bad.affect.valence


def test_feedback_is_stored_as_its_own_kind_of_episode():
    mind = _mind()
    mind.perceive("the answer was about vocoders")
    trace = mind.reward(0.8, note="that was right")

    assert trace.kind == "feedback"
    assert mind.episodic.all()[-1].reward == pytest.approx(0.8)


# ---------------------------------------------------------------------------
# sleep and consolidation
# ---------------------------------------------------------------------------

def test_fatigue_accumulates_until_the_mind_sleeps_on_its_own():
    mind = _mind(fatigue_per_tick=0.2, sleep_threshold=0.6)
    slept = [mind.perceive(f"fact number {i}").slept for i in range(10)]

    reports = [s for s in slept if s]
    assert reports, "the mind never slept despite accumulating fatigue"
    assert mind.affect.fatigue < 0.6
    assert mind.self_model.sleeps == len(reports)


def test_sleep_forms_concepts_and_writes_a_reflection():
    mind = _mind(concept_threshold=0.5)
    for i in range(6):
        mind.perceive(f"rectified flow variant {i}", topic="flow")

    report = mind.rest()
    assert report["rehearsed"] > 0
    assert len(mind.semantic) > 0
    assert report["reflection"] in [n["text"] for n in mind.self_model.narrative]


def test_sleep_rehearsal_lowers_prediction_error_on_replayed_material():
    """Replay is not bookkeeping: rehearsed episodes get easier to predict."""
    mind = _mind(replay_batch=8)
    text = "the spacetime transformer factorises attention"
    for _ in range(3):
        mind.perceive(text)

    embedding = mind.encoder.encode_text(text)
    before = mind.curiosity.novelty.error_of(embedding)
    for _ in range(5):
        mind.rest()
    assert mind.curiosity.novelty.error_of(embedding) < before


def test_train_step_receives_the_replayed_episodes():
    mind = _mind(replay_batch=3)
    for i in range(6):
        mind.perceive(f"episode {i}")

    seen = {}

    def train_step(batch):
        seen["batch"] = batch
        return {"loss": 0.5}

    report = mind.rest(train_step=train_step)
    assert report["training"] == {"loss": 0.5}
    assert 0 < len(seen["batch"]) <= 3
    assert all(hasattr(e, "text") for e in seen["batch"])


def test_sleep_resets_fatigue_and_settles_arousal():
    mind = _mind()
    for i in range(5):
        mind.perceive(f"tiring fact {i}")
    mind.affect.fatigue = 0.9

    mind.rest()
    assert mind.affect.fatigue == 0.0


# ---------------------------------------------------------------------------
# curiosity in the loop
# ---------------------------------------------------------------------------

def test_wonder_asks_a_question_and_raises_a_goal():
    mind = _mind(sleep_threshold=1.0)
    for i in range(4):
        mind.perceive(f"audio codec detail {i}", topic="audio")

    question = mind.wonder()
    assert question.endswith("?")
    assert mind.episodic.all()[-1].kind == "question"
    assert any(g.origin == "self" for g in mind.self_model.goals)


def test_wonder_works_with_no_history():
    assert _mind().wonder().endswith("?")


# ---------------------------------------------------------------------------
# context building
# ---------------------------------------------------------------------------

def test_context_includes_identity_memory_and_feeling_within_budget():
    mind = _mind()
    mind.perceive("the vocoder inverts mel spectrograms")
    trace = mind.perceive("the vocoder inverts mel spectrograms")

    context = mind.context(trace, budget=600)
    assert "FramerAI" in context
    assert "I remember:" in context
    assert "I feel" in context
    assert len(context) <= 600


def test_context_stays_inside_a_tight_budget():
    mind = _mind()
    for i in range(6):
        mind.perceive("a fairly long sentence about diffusion samplers " * 3 + str(i))
    assert len(mind.context(budget=200)) <= 200


# ---------------------------------------------------------------------------
# generation
# ---------------------------------------------------------------------------

def test_converse_returns_the_continuation_not_the_echoed_prompt():
    mind = Mind.from_generator(
        _generator(), CognitionConfig(d_embed=32, episodic_capacity=16, retrieval_k=2)
    )
    prompt = "what is a vocoder"
    reply, trace = mind.converse(prompt, max_new_tokens=8)

    assert reply is not None
    assert not reply.startswith(prompt)
    assert trace.response == reply
    assert trace.sampling["temperature"] > 0


def test_converse_remembers_its_own_reply_as_an_action():
    mind = Mind.from_generator(
        _generator(), CognitionConfig(d_embed=32, episodic_capacity=16, retrieval_k=2)
    )
    mind.converse("what is a vocoder", max_new_tokens=8)

    kinds = [e.kind for e in mind.episodic.all()]
    assert "perception" in kinds and "action" in kinds
    assert mind.self_model.exchanges == 1


def test_converse_without_a_generator_still_runs_the_loop():
    mind = _mind()
    reply, trace = mind.converse("what is a vocoder", max_new_tokens=4)

    assert reply is None
    assert trace.sampling
    assert len(mind.episodic) == 1


def test_generator_backed_mind_encodes_through_the_model():
    mind = Mind.from_generator(_generator(), CognitionConfig(d_embed=32))
    assert mind.encoder.grounded
    vector = mind.encoder.encode_text("rectified flow")
    assert vector.shape == (32,)
    assert torch.isfinite(vector).all()
    assert float(vector.norm()) == pytest.approx(1.0, abs=1e-5)


def test_encoding_a_long_input_is_truncated_to_the_context_window():
    mind = Mind.from_generator(_generator(), CognitionConfig(d_embed=32))
    vector = mind.encoder.encode_text("vocoder mel spectrogram " * 200)
    assert vector.shape == (32,)
    assert torch.isfinite(vector).all()


def test_image_encoding_requires_a_multimodal_model():
    mind = Mind.from_generator(_generator(), CognitionConfig(d_embed=32))
    with pytest.raises(RuntimeError, match="multimodal"):
        mind.perceive_image(torch.zeros(3, 32, 32))


def test_hash_encoder_is_stable_and_similar_for_similar_text():
    a = hash_embed("rectified flow trains straight paths", 64)
    b = hash_embed("rectified flow trains straight paths", 64)
    c = hash_embed("rectified flow trains straight paths between noise and data", 64)
    d = hash_embed("coral reefs bleach in warm water", 64)

    assert torch.equal(a, b)
    assert float(a @ c) > float(a @ d)


def test_standalone_encoder_reports_itself_as_ungrounded():
    assert not ExperienceEncoder(16).grounded


# ---------------------------------------------------------------------------
# introspection and persistence
# ---------------------------------------------------------------------------

def test_introspect_reports_the_whole_state():
    mind = _mind()
    for i in range(3):
        mind.perceive(f"audio codec detail {i}", topic="audio")

    snapshot = mind.introspect()
    for key in ("tick", "affect", "memory", "concepts", "interests", "goals", "frontier"):
        assert key in snapshot
    assert snapshot["tick"] == 3
    assert snapshot["memory"]["episodes"] == 3


def test_saved_mind_reloads_with_its_history_intact(tmp_path):
    mind = _mind()
    for i in range(5):
        mind.perceive(f"rectified flow detail {i}", topic="flow")
    mind.wonder()
    path = str(tmp_path / "mind.pt")
    mind.save(path)

    restored = Mind.load(path)
    assert restored.tick == mind.tick
    assert len(restored.episodic) == len(mind.episodic)
    assert restored.affect.to_dict() == mind.affect.to_dict()
    assert restored.self_model.goals[0].text == mind.self_model.goals[0].text

    hits = restored.recall("rectified flow detail 2")
    assert hits and "rectified flow" in hits[0].experience.text


def test_a_reloaded_mind_keeps_its_habituation(tmp_path):
    mind = _mind(sleep_threshold=1.0)
    text = "residual vector quantization of audio"
    for _ in range(8):
        mind.perceive(text)
    path = str(tmp_path / "mind.pt")
    mind.save(path)

    restored = Mind.load(path)
    familiar = restored.perceive(text).novelty
    novel = restored.perceive("an unrelated claim about coral reefs").novelty
    assert novel > familiar


def test_config_validation_rejects_impossible_settings():
    with pytest.raises(ValueError, match="d_embed"):
        CognitionConfig(d_embed=0).validate()
    with pytest.raises(ValueError, match="concept_threshold"):
        CognitionConfig(concept_threshold=2.0).validate()
    with pytest.raises(ValueError, match="curiosity_temperature"):
        CognitionConfig(curiosity_temperature=0.0).validate()

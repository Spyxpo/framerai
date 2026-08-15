"""Tests for curiosity and affect.

The claims being checked are behavioural, not aesthetic: novelty falls with
repeated exposure and jumps for genuinely new input; learning progress is a
slope, so a topic whose error is flat scores zero however large that error is;
the exploration frontier prefers where progress is happening; affect decays
toward its setpoints, responds to appraisal with the right signs, and actually
moves the decoder.

CPU-only, seeded, no checkpoint required.
"""

import pytest
import torch

from model.cognition import (
    AffectState,
    Appraisal,
    CognitionConfig,
    CuriosityEngine,
    LearningProgress,
    NoveltyEstimator,
    hash_embed,
)
from model.cognition.affect import SETPOINTS


def _config(**overrides) -> CognitionConfig:
    base = dict(d_embed=32, novelty_hidden=32, learning_progress_window=8)
    base.update(overrides)
    return CognitionConfig(**base).validate()


# ---------------------------------------------------------------------------
# novelty (RND) - habituation
# ---------------------------------------------------------------------------

def test_novelty_falls_with_repeated_exposure():
    estimator = NoveltyEstimator(_config())
    vec = hash_embed("residual vector quantization", 32)

    first, _ = estimator.observe(vec)
    for _ in range(20):
        estimator.observe(vec)
    last, _ = estimator.observe(vec)
    assert last < first


def test_novelty_is_high_for_something_genuinely_new():
    estimator = NoveltyEstimator(_config())
    familiar = hash_embed("rectified flow objective", 32)
    for _ in range(20):
        estimator.observe(familiar)

    novel, _ = estimator.observe(hash_embed("coral reefs bleach in warm water", 32))
    seen_again, _ = estimator.observe(familiar)
    assert novel > seen_again


def test_familiarity_probe_does_not_train_the_predictor():
    estimator = NoveltyEstimator(_config())
    vec = hash_embed("spacetime diffusion transformer", 32)
    estimator.observe(vec)

    before = [p.clone() for p in estimator.predictor.parameters()]
    estimator.familiarity_of(vec)
    after = list(estimator.predictor.parameters())
    assert all(torch.equal(b, a) for b, a in zip(before, after, strict=True))


def test_target_network_stays_frozen():
    estimator = NoveltyEstimator(_config())
    before = [p.clone() for p in estimator.target.parameters()]
    for _ in range(5):
        estimator.observe(hash_embed("anything at all", 32))
    assert all(
        torch.equal(b, a) for b, a in zip(before, estimator.target.parameters(), strict=True)
    )
    assert not any(p.requires_grad for p in estimator.target.parameters())


# ---------------------------------------------------------------------------
# learning progress
# ---------------------------------------------------------------------------

def test_learning_progress_measures_the_slope_not_the_level():
    progress = LearningProgress(_config())
    for error in [1.0, 0.9, 0.8, 0.7, 0.3, 0.2, 0.15, 0.1]:
        progress.record("improving", error)
    # Large but flat error: nothing is being learned here.
    for _ in range(8):
        progress.record("stuck", 5.0)

    assert progress.progress("improving") > 0.5
    assert progress.progress("stuck") == 0.0


def test_learning_progress_is_zero_before_enough_history():
    progress = LearningProgress(_config())
    progress.record("new", 1.0)
    assert progress.progress("new") == 0.0


def test_competence_rises_as_error_falls():
    progress = LearningProgress(_config())
    for _ in range(4):
        progress.record("easy", 0.01)
        progress.record("hard", 10.0)
    assert progress.competence("easy") > progress.competence("hard")
    assert progress.competence("never seen") == 0.0


def test_expected_error_is_none_until_a_topic_is_seen():
    progress = LearningProgress(_config())
    assert progress.expected_error("unseen") is None
    progress.record("seen", 0.5)
    assert progress.expected_error("seen") == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# the engine
# ---------------------------------------------------------------------------

def test_appraisal_reports_novelty_surprise_and_progress():
    engine = CuriosityEngine(_config())
    appraisal = engine.appraise(hash_embed("audio codec", 32), topic="audio")

    assert isinstance(appraisal, Appraisal)
    assert 0.0 <= appraisal.novelty <= 1.0
    assert 0.0 <= appraisal.surprise <= 1.0
    assert appraisal.learning_progress == 0.0  # no history yet


def test_frontier_prefers_the_topic_that_is_moving():
    engine = CuriosityEngine(_config(curiosity_temperature=0.05))
    for error in [1.0, 0.95, 0.9, 0.85, 0.2, 0.15, 0.1, 0.05]:
        engine.progress.record("learnable", error)
    for _ in range(8):
        engine.progress.record("mastered", 0.0)

    picks = [engine.frontier(["learnable", "mastered"]) for _ in range(10)]
    assert picks.count("learnable") > picks.count("mastered")


def test_frontier_is_none_with_nothing_experienced():
    assert CuriosityEngine(_config()).frontier() is None


def test_frontier_never_offers_a_bookkeeping_axis_as_a_subject():
    """"sense:text" is how the mind rates itself, not something to go and learn."""
    engine = CuriosityEngine(_config())
    for _ in range(6):
        engine.progress.record("vocoders", 1.0)
        engine.progress.record("sense:text", 1.0)
        engine.progress.record("lang:en", 1.0)

    assert engine.progress.subjects() == ["vocoders"]
    assert {engine.frontier() for _ in range(10)} == {"vocoders"}
    # Explicitly asking for one is still allowed - the filter is on the default.
    assert engine.frontier(["lang:en"]) == "lang:en"


def test_question_names_the_topic():
    engine = CuriosityEngine(_config())
    assert "vocoders" in engine.question("vocoders")


def test_curiosity_state_round_trip_preserves_habituation():
    engine = CuriosityEngine(_config())
    vec = hash_embed("mel spectrogram inversion", 32)
    for _ in range(10):
        engine.appraise(vec, topic="audio")
    state = engine.state_dict()

    restored = CuriosityEngine(_config()).load_state_dict(state)
    assert restored.ticks == engine.ticks
    assert restored.novelty.familiarity_of(vec) == pytest.approx(
        engine.novelty.familiarity_of(vec), abs=1e-6
    )
    assert restored.progress.expected_error("audio") == pytest.approx(
        engine.progress.expected_error("audio")
    )


# ---------------------------------------------------------------------------
# affect
# ---------------------------------------------------------------------------

def test_affect_decays_toward_its_setpoints():
    affect = AffectState(valence=1.0, arousal=1.0, fatigue=1.0)
    for _ in range(200):
        affect.decay(0.1)
    assert affect.valence == pytest.approx(SETPOINTS["valence"], abs=1e-3)
    assert affect.arousal == pytest.approx(SETPOINTS["arousal"], abs=1e-3)
    assert affect.fatigue == pytest.approx(0.0, abs=1e-3)


def test_surprise_raises_arousal_and_lowers_confidence():
    affect = AffectState()
    before = (affect.arousal, affect.confidence)
    affect.update(Appraisal(surprise=1.0), gain=0.5, decay=0.0)
    assert affect.arousal > before[0]
    assert affect.confidence < before[1]


def test_reward_raises_valence_and_punishment_lowers_it():
    good, bad = AffectState(), AffectState()
    good.update(Appraisal(reward=1.0), gain=0.5, decay=0.0)
    bad.update(Appraisal(reward=-1.0), gain=0.5, decay=0.0)
    assert good.valence > 0 > bad.valence


def test_curiosity_is_satiated_when_nothing_is_new():
    affect = AffectState(curiosity=0.8)
    for _ in range(10):
        affect.update(Appraisal(), gain=0.35, decay=0.0)
    assert affect.curiosity < 0.8


def test_effort_accumulates_as_fatigue_and_only_rest_clears_it():
    affect = AffectState()
    for _ in range(10):
        affect.update(Appraisal(cost=0.05), gain=0.35, decay=0.0)
    assert affect.fatigue > 0.3

    affect.rest()
    assert affect.fatigue == 0.0


def test_every_dimension_stays_in_bounds_under_extremes():
    affect = AffectState()
    for _ in range(100):
        affect.update(
            Appraisal(novelty=1.0, surprise=1.0, learning_progress=1.0, reward=5.0, cost=1.0),
            gain=1.0, decay=0.0,
        )
    assert -1.0 <= affect.valence <= 1.0
    for name in ("arousal", "confidence", "curiosity", "fatigue"):
        assert 0.0 <= getattr(affect, name) <= 1.0


def test_affect_widens_or_narrows_sampling():
    excited = AffectState(arousal=0.95, curiosity=0.95, confidence=0.1).modulate(
        0.7, top_p=0.9, top_k=50
    )
    settled = AffectState(arousal=0.05, curiosity=0.1, confidence=0.95, fatigue=0.9).modulate(
        0.7, top_p=0.9, top_k=50
    )
    assert excited["temperature"] > 0.7 > settled["temperature"]
    assert excited["top_p"] > settled["top_p"]
    assert excited["top_k"] > settled["top_k"]
    assert settled["top_k"] >= 1


def test_modulation_stays_inside_decoder_limits():
    for state in (AffectState(arousal=1.0, curiosity=1.0), AffectState(confidence=1.0, fatigue=1.0)):
        out = state.modulate(1.9, top_p=0.99, top_k=1, temperature_span=5.0, top_p_span=5.0)
        assert 0.05 <= out["temperature"] <= 2.0
        assert 0.1 <= out["top_p"] <= 1.0
        assert out["top_k"] >= 1


def test_describe_reads_the_state():
    assert "curious" in AffectState(curiosity=0.9).describe()
    assert "tired" in AffectState(fatigue=0.9).describe()
    assert AffectState(valence=0.05, arousal=0.25, confidence=0.5, curiosity=0.5).describe() == "even"


def test_affect_dict_round_trip():
    affect = AffectState(valence=-0.3, arousal=0.8, confidence=0.2, curiosity=0.9, fatigue=0.4)
    assert AffectState.from_dict(affect.to_dict()).to_dict() == affect.to_dict()

"""Tests for live perception: streaming senses, the attention gate, and video.

The properties that matter for a mind that is always looking: a static scene must
not flood memory with duplicates, a change must get through, a totally frozen
sense must still produce the occasional check-in, an idle mind must turn to its
own questions, and a background session must not corrupt state while the
foreground is mid-conversation.

Camera and microphone hardware are never touched. ``CallableSource`` feeds the
same code path the real sources use, and the multimodal tests build a tiny model
on CPU so the grounded encoding path is exercised for real.
"""

import threading

import pytest
import torch

from conftest import tiny_config, tiny_multimodal_config
from model.cognition import (
    CallableSource,
    ChangeGate,
    CognitionConfig,
    LiveSession,
    Mind,
    frame_to_tensor,
)
from model.framer import FramerModel
from model.generate import FramerGenerator
from model.tokenizer import FramerTokenizer


def _mind(**overrides) -> Mind:
    base = dict(
        d_embed=32, episodic_capacity=64, retrieval_k=3, novelty_hidden=32,
        replay_batch=4, sleep_threshold=0.95,
    )
    base.update(overrides)
    return Mind(CognitionConfig(**base))


def _multimodal_generator() -> FramerGenerator:
    tokenizer = FramerTokenizer(vocab_size=300)
    tokenizer.train(["a room with a window", "hello world"], target_vocab_size=300)
    config = tiny_multimodal_config(
        vocab_size=tokenizer.vocab_size, max_seq_len=64, d_model=64,
        n_heads=8, n_kv_heads=2, n_layers=2, d_ff=128, dropout=0.0,
    )
    return FramerGenerator(FramerModel(config), tokenizer, device="cpu")


def _still(value: float, size: int = 16) -> torch.Tensor:
    return torch.full((3, size, size), value)


# ---------------------------------------------------------------------------
# the attention gate
# ---------------------------------------------------------------------------

def test_first_reading_is_always_attended():
    gate = ChangeGate(threshold=0.5)
    attend, change = gate.admit(torch.nn.functional.normalize(torch.randn(8), dim=0))
    assert attend and change == 1.0


def test_an_unchanged_scene_is_gated_out():
    gate = ChangeGate(threshold=0.1, max_skip=100)
    vec = torch.nn.functional.normalize(torch.randn(8), dim=0)
    gate.admit(vec)
    assert gate.admit(vec.clone()) == (False, pytest.approx(0.0, abs=1e-6))


def test_a_changed_scene_gets_through():
    gate = ChangeGate(threshold=0.1, max_skip=100)
    gate.admit(torch.tensor([1.0, 0.0]))
    attend, change = gate.admit(torch.tensor([0.0, 1.0]))
    assert attend and change == pytest.approx(1.0)


def test_a_frozen_sense_still_checks_in_after_max_skip():
    gate = ChangeGate(threshold=0.5, max_skip=3)
    vec = torch.tensor([1.0, 0.0])
    admits = [gate.admit(vec)[0] for _ in range(6)]
    assert admits[0] is True
    assert admits[1:4] == [False, False, False]
    assert admits[4] is True  # forced check-in


def test_gate_rejects_a_threshold_outside_cosine_distance():
    with pytest.raises(ValueError):
        ChangeGate(threshold=3.0)


# ---------------------------------------------------------------------------
# frame conversion
# ---------------------------------------------------------------------------

def test_frame_to_tensor_normalises_uint8_to_the_model_range():
    frame = torch.full((16, 16, 3), 255, dtype=torch.uint8)
    tensor = frame_to_tensor(frame)
    assert tensor.shape == (3, 16, 16)
    assert float(tensor.max()) == pytest.approx(1.0)
    assert float(frame_to_tensor(torch.zeros(16, 16, 3, dtype=torch.uint8)).min()) == -1.0


def test_frame_to_tensor_resizes_when_asked():
    assert frame_to_tensor(torch.zeros(32, 32, 3), size=16).shape == (3, 16, 16)


def test_frame_to_tensor_rejects_a_non_image():
    with pytest.raises(ValueError, match="H, W, 3"):
        frame_to_tensor(torch.zeros(16, 16))


# ---------------------------------------------------------------------------
# streaming into the mind
# ---------------------------------------------------------------------------

def test_a_static_scene_does_not_flood_memory():
    mind = _mind()
    session = LiveSession(
        mind, [CallableSource(lambda: _still(0.5), "vision", "cam")],
        fps=0, change_threshold=0.05, max_skip=1000, wonder_when_idle=False,
    )
    events = session.run(ticks=20, sleep_fn=lambda _: None)

    assert len(events) == 20
    assert sum(e.attended for e in events) == 1
    assert len(mind.episodic) == 1


def test_a_change_in_the_scene_is_attended_and_remembered():
    mind = _mind()
    state = {"n": 0}

    def camera():
        state["n"] += 1
        return _still(0.0) if state["n"] <= 3 else _still(0.9)

    session = LiveSession(
        mind, [CallableSource(camera, "vision", "cam")],
        fps=0, change_threshold=0.05, max_skip=1000, wonder_when_idle=False,
    )
    events = session.run(ticks=6, sleep_fn=lambda _: None)

    assert [e.attended for e in events] == [True, False, False, True, False, False]
    assert len(mind.episodic) == 2
    assert {e.modality for e in mind.episodic.all()} == {"vision"}


def test_both_senses_run_in_one_session():
    mind = _mind()
    state = {"n": 0}

    def camera():
        state["n"] += 1
        return torch.rand(3, 16, 16, generator=torch.Generator().manual_seed(state["n"]))

    def microphone():
        return torch.sin(torch.linspace(0, state["n"] + 1, 400))

    session = LiveSession(
        mind,
        [CallableSource(camera, "vision", "cam"), CallableSource(microphone, "hearing", "mic")],
        fps=0, change_threshold=0.01, wonder_when_idle=False,
    )
    session.run(ticks=4, sleep_fn=lambda _: None)

    assert mind.self_model.modalities["vision"] > 0
    assert mind.self_model.modalities["hearing"] > 0
    assert session.summary()["polls"] == 8


def test_a_source_with_nothing_available_is_skipped_not_recorded():
    mind = _mind()
    session = LiveSession(
        mind, [CallableSource(lambda: None, "hearing", "mic")], fps=0, wonder_when_idle=False
    )
    events = session.run(ticks=5, sleep_fn=lambda _: None)

    assert events == []
    assert len(mind.episodic) == 0
    assert session.summary()["polls"] == 5


def test_an_idle_session_turns_to_its_own_questions():
    mind = _mind()
    session = LiveSession(
        mind, [CallableSource(lambda: _still(0.4), "vision", "cam")],
        fps=0, change_threshold=0.5, max_skip=1000, wonder_when_idle=True, idle_ticks=3,
    )
    session.run(ticks=10, sleep_fn=lambda _: None)

    kinds = [e.kind for e in mind.episodic.all()]
    assert "question" in kinds


def test_run_needs_a_bound():
    with pytest.raises(ValueError, match="seconds or ticks"):
        LiveSession(_mind()).run()


def test_run_stops_on_its_time_budget():
    mind = _mind()
    clock = {"t": 0.0}
    session = LiveSession(
        mind, [CallableSource(lambda: torch.rand(3, 8, 8), "vision", "cam")],
        fps=10, wonder_when_idle=False,
    )

    def sleep_fn(seconds):
        clock["t"] += seconds

    session.run(seconds=1.0, sleep_fn=sleep_fn, clock=lambda: clock["t"])
    assert 9 <= session.polls <= 12  # 10 fps for one second, give or take one poll


def test_summary_reports_attention_rate_and_sources():
    mind = _mind()
    session = LiveSession(
        mind, [CallableSource(lambda: _still(0.5), "vision", "cam")],
        fps=0, change_threshold=0.05, max_skip=1000, wonder_when_idle=False,
    )
    session.run(ticks=4, sleep_fn=lambda _: None)

    summary = session.summary()
    assert summary["attended"] == 1
    assert summary["attention_rate"] == 0.25
    assert summary["sources"] == [{"name": "cam", "modality": "vision"}]
    assert summary["grounded"] is False  # no multimodal model attached
    assert len(session.recent(2)) == 2


def test_background_session_runs_and_stops_cleanly():
    mind = _mind()
    session = LiveSession(
        mind, [CallableSource(lambda: torch.rand(3, 8, 8), "vision", "cam")],
        fps=200, change_threshold=0.0, wonder_when_idle=False,
    )
    session.start()
    assert session.running
    for _ in range(200):
        if session.polls > 3:
            break
    session.stop()

    assert not session.running
    assert session.polls > 0
    assert len(mind.episodic) == mind.tick  # every attended poll produced one episode


def test_the_mind_lock_serialises_background_perception_with_conversing():
    """A live sense and a foreground exchange must not interleave mid-tick."""
    mind = _mind()
    session = LiveSession(
        mind, [CallableSource(lambda: torch.rand(3, 8, 8), "vision", "cam")],
        fps=500, change_threshold=0.0, wonder_when_idle=False,
    )
    session.start()
    errors = []

    def talk():
        try:
            for _ in range(20):
                mind.converse("what is happening", max_new_tokens=1)
        except Exception as exc:  # noqa: BLE001 - the point of the test is that none escape
            errors.append(exc)

    threads = [threading.Thread(target=talk) for _ in range(2)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    session.stop()

    assert not errors
    # Ticks and stored episodes stay in step, which a torn update would break.
    assert mind.tick == len(mind.episodic)


# ---------------------------------------------------------------------------
# grounded senses: real vision and audio towers
# ---------------------------------------------------------------------------

def test_a_multimodal_model_encodes_sight_and_sound_through_its_towers():
    mind = Mind.from_generator(_multimodal_generator(), CognitionConfig(d_embed=32))
    assert mind.encoder.multimodal

    trace = mind.perceive_image(torch.rand(3, 32, 32), caption="a bright room")
    assert trace.modality == "vision"
    assert trace.text == "a bright room"

    heard = mind.perceive_audio(torch.rand(16000) * 2 - 1, caption="someone speaking")
    assert heard.modality == "hearing"
    assert len(mind.episodic) == 2


def test_a_clip_is_remembered_as_one_event_not_n_stills():
    mind = Mind.from_generator(_multimodal_generator(), CognitionConfig(d_embed=32))
    frames = torch.rand(8, 3, 32, 32)

    trace = mind.perceive_video(frames, caption="the door opens", keyframes=4)
    assert trace.modality == "vision"
    assert len(mind.episodic) == 1
    assert mind.episodic.all()[0].text == "the door opens"


def test_video_rejects_frames_of_the_wrong_shape():
    mind = Mind.from_generator(_multimodal_generator(), CognitionConfig(d_embed=32))
    with pytest.raises(ValueError, match=r"\(T, C, H, W\)"):
        mind.perceive_video(torch.rand(3, 32, 32))


def test_describing_an_image_writes_the_description_into_memory():
    mind = Mind.from_generator(_multimodal_generator(), CognitionConfig(d_embed=32))
    trace = mind.perceive_image(torch.rand(3, 32, 32), describe=True)

    assert len(mind.episodic) == 1
    assert trace.text  # an untrained model says nonsense, but it says it into memory
    assert trace.modality == "vision"


def test_a_live_session_on_a_multimodal_model_reports_itself_grounded():
    mind = Mind.from_generator(_multimodal_generator(), CognitionConfig(d_embed=32))
    session = LiveSession(
        mind, [CallableSource(lambda: torch.rand(3, 32, 32), "vision", "cam")],
        fps=0, change_threshold=0.01, wonder_when_idle=False,
    )
    session.run(ticks=3, sleep_fn=lambda _: None)

    assert session.grounded is True
    assert len(mind.episodic) > 0


def test_text_only_models_still_stream_but_say_they_are_not_grounded():
    tokenizer = FramerTokenizer(vocab_size=300)
    tokenizer.train(["hello world"], target_vocab_size=300)
    generator = FramerGenerator(
        FramerModel(tiny_config(vocab_size=tokenizer.vocab_size, max_seq_len=64)),
        tokenizer, device="cpu",
    )
    mind = Mind.from_generator(generator, CognitionConfig(d_embed=32))
    session = LiveSession(
        mind, [CallableSource(lambda: torch.rand(3, 16, 16), "vision", "cam")],
        fps=0, change_threshold=0.01, wonder_when_idle=False,
    )
    session.run(ticks=3, sleep_fn=lambda _: None)

    assert session.grounded is False
    assert len(mind.episodic) > 0


def test_closing_a_session_closes_its_sources():
    closed = []

    class Recording(CallableSource):
        def close(self):
            closed.append(self.name)

    session = LiveSession(_mind(), [Recording(lambda: None, "vision", "cam")], fps=0)
    session.close()
    assert closed == ["cam"]

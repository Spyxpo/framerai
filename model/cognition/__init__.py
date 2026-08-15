"""Cognition layer: memory, curiosity, affect, self-model, and consolidation.

A FramerAI checkpoint answers each prompt from nothing but its weights and the
current context. This package wraps it in the machinery a mind needs to have a
history: experiences it keeps and forgets, an intrinsic drive that decides what
is worth looking at, an affective state that changes how it answers, a model of
itself that experience updates, and an offline pass that turns episodes into
concepts.

Typical use::

    from model.cognition import Mind
    from model.generate import FramerGenerator

    gen = FramerGenerator.from_checkpoint("model.pt", "tokenizer.json")
    mind = Mind.from_generator(gen)

    reply, trace = mind.converse("what is a rectified flow?")
    print(trace.feeling, trace.novelty, trace.recalled)
    print(mind.wonder())          # a question it asked itself
    mind.save("mind.pt")          # continuity across restarts

The docstring in :mod:`model.cognition.mind` states plainly what is and is not
being claimed here.
"""

from .affect import AffectState, Appraisal
from .config import EXPERIENCE_KINDS, CognitionConfig
from .consolidation import Consolidator
from .curiosity import CuriosityEngine, LearningProgress, NoveltyEstimator
from .encoder import ExperienceEncoder, hash_embed
from .language import Language, detect_language, detect_script, language_name
from .memory import (
    Concept,
    EpisodicMemory,
    Experience,
    MemoryHit,
    SemanticMemory,
    WorkingMemory,
)
from .mind import Mind, MindTrace
from .perception import (
    CallableSource,
    CameraSource,
    ChangeGate,
    LiveSession,
    MicrophoneSource,
    SensoryEvent,
    SensorySource,
    frame_to_tensor,
)
from .self_model import Goal, SelfModel

__all__ = [
    "EXPERIENCE_KINDS",
    "AffectState",
    "Appraisal",
    "CallableSource",
    "CameraSource",
    "ChangeGate",
    "CognitionConfig",
    "Concept",
    "Consolidator",
    "CuriosityEngine",
    "EpisodicMemory",
    "Experience",
    "ExperienceEncoder",
    "Goal",
    "Language",
    "LearningProgress",
    "LiveSession",
    "MemoryHit",
    "MicrophoneSource",
    "Mind",
    "MindTrace",
    "NoveltyEstimator",
    "SelfModel",
    "SemanticMemory",
    "SensoryEvent",
    "SensorySource",
    "WorkingMemory",
    "detect_language",
    "detect_script",
    "frame_to_tensor",
    "hash_embed",
    "language_name",
]

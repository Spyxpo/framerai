"""Configuration for the cognition layer.

Every knob the mind uses lives here, with no torch dependency, so the
configuration can be validated, serialised, and compared without building
anything. Defaults are sized for a laptop: the whole cognitive state of a
long-running mind is a few megabytes.
"""

from dataclasses import asdict, dataclass, fields

# What an experience can be. The kind is stored on every episode and is what
# separates "something happened to me" from "I did this" from "I thought about
# it afterwards", which consolidation and reflection both need to tell apart.
EXPERIENCE_KINDS = ("perception", "action", "reflection", "question", "feedback")


@dataclass
class CognitionConfig:
    """Configuration for :class:`model.cognition.mind.Mind`."""

    # Dimension of the vector every experience is projected into. Independent of
    # the model's d_model so the mind keeps working when the backbone is swapped.
    d_embed: int = 256
    seed: int = 42

    # ---- working memory ------------------------------------------------
    # The handful of items currently "in mind". Small on purpose: an unbounded
    # scratchpad is a log, not a working memory, and stops forcing selection.
    working_memory_size: int = 8

    # ---- episodic memory -----------------------------------------------
    episodic_capacity: int = 4096
    retrieval_k: int = 5
    # Retrieval scores blend similarity, how recent the episode is, and how much
    # it mattered when it happened. Similarity alone recalls the relevant but
    # dead; recency alone recalls the last thing regardless of relevance.
    w_similarity: float = 1.0
    w_recency: float = 0.35
    w_salience: float = 0.45
    # Ticks over which an untouched memory loses half its strength, and the
    # strength a successful recall adds back. Together these give the spacing
    # effect: memories that keep getting used survive, the rest fade.
    strength_half_life: float = 800.0
    recency_half_life: float = 200.0
    retrieval_boost: float = 0.25
    # Floor a memory must clear to be retrieved at all, so a query with nothing
    # relevant behind it returns nothing instead of noise.
    retrieval_floor: float = 0.15

    # ---- semantic memory -----------------------------------------------
    # Episodes that keep recurring collapse into concepts: a centroid, a count,
    # and the most central episode as its exemplar.
    semantic_capacity: int = 256
    concept_threshold: float = 0.62  # cosine similarity needed to join a concept
    concept_recall_k: int = 3

    # ---- curiosity ------------------------------------------------------
    # Random network distillation: novelty is how badly a small predictor
    # matches a frozen random target on this embedding. Training the predictor
    # on what it sees is habituation - the familiar stops being interesting.
    novelty_hidden: int = 128
    novelty_lr: float = 1e-3
    novelty_error_ema: float = 0.99
    # Learning progress (Oudeyer/Kaplan): the drive is not "what is unknown" but
    # "where am I currently getting better", which avoids both the boring and
    # the hopeless.
    learning_progress_window: int = 12
    curiosity_temperature: float = 0.5
    w_novelty: float = 0.6
    w_learning_progress: float = 0.4

    # ---- affect ---------------------------------------------------------
    # Affect is a five-dimensional homeostat, not a label. It decays toward its
    # setpoints and is pushed by appraisal, and it feeds back into sampling.
    affect_decay: float = 0.08
    affect_gain: float = 0.35
    # How far affect is allowed to move the decoder. At 0.0 the mind still feels
    # things, it just stops acting differently because of them.
    temperature_span: float = 0.45
    top_p_span: float = 0.08

    # ---- drives and consolidation ---------------------------------------
    fatigue_per_tick: float = 0.02
    sleep_threshold: float = 0.85
    replay_batch: int = 16
    replay_alpha: float = 0.7  # prioritisation exponent over salience
    consolidation_decay: float = 0.9  # strength scaling applied during sleep

    # ---- prompting -------------------------------------------------------
    # Recalled context is injected into the prompt, so it has to stay bounded or
    # it eats the context window it is supposed to help with.
    context_char_budget: int = 1200
    narrative_capacity: int = 64

    def validate(self) -> "CognitionConfig":
        """Check invariants that would otherwise fail deep inside a tick."""
        positive = (
            "d_embed", "working_memory_size", "episodic_capacity", "retrieval_k",
            "semantic_capacity", "concept_recall_k", "novelty_hidden",
            "learning_progress_window", "replay_batch", "narrative_capacity",
            "context_char_budget",
        )
        for name in positive:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")

        for name in ("strength_half_life", "recency_half_life"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be > 0, got {getattr(self, name)}")

        unit_range = (
            "concept_threshold", "affect_decay", "affect_gain", "fatigue_per_tick",
            "sleep_threshold", "replay_alpha", "consolidation_decay", "retrieval_floor",
            "novelty_error_ema",
        )
        for name in unit_range:
            value = getattr(self, name)
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be in [0, 1], got {value}")

        if self.retrieval_k > self.episodic_capacity:
            raise ValueError("retrieval_k cannot exceed episodic_capacity")
        if self.curiosity_temperature <= 0:
            raise ValueError("curiosity_temperature must be > 0")
        if self.w_novelty + self.w_learning_progress <= 0:
            raise ValueError("curiosity needs at least one non-zero drive weight")
        return self

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "CognitionConfig":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})

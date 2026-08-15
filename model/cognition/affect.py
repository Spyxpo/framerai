"""Affective state: the part of the mind that has a weather.

This is not a claim about feeling. It is a five-dimensional homeostat that is
pushed by appraisal of what just happened, decays back toward its setpoints when
nothing does, and - crucially - changes behaviour: a mind that is aroused and
curious samples wider, a confident one samples tighter. Affect that does not
reach the decoder is decoration; this one does.

Dimensions follow the core-affect tradition (valence/arousal) plus the three
signals a learning agent actually needs to regulate: confidence, curiosity, and
fatigue.
"""

from dataclasses import asdict, dataclass, fields

# Where each dimension drifts back to with nothing happening. Curiosity rests
# above zero on purpose: a mind with no input should get restless, not flat.
SETPOINTS = {
    "valence": 0.05,
    "arousal": 0.25,
    "confidence": 0.5,
    "curiosity": 0.55,
    "fatigue": 0.0,
}

# (lower, upper) clamp per dimension. Valence is signed; the rest are unipolar.
BOUNDS = {
    "valence": (-1.0, 1.0),
    "arousal": (0.0, 1.0),
    "confidence": (0.0, 1.0),
    "curiosity": (0.0, 1.0),
    "fatigue": (0.0, 1.0),
}


def _clamp(name: str, value: float) -> float:
    low, high = BOUNDS[name]
    return max(low, min(high, value))


@dataclass
class Appraisal:
    """What the mind made of an event, before it becomes a feeling.

    ``novelty`` is unfamiliarity, ``surprise`` is expectation violated (they
    differ: a familiar thing behaving wrongly is surprising but not novel),
    ``learning_progress`` is how fast error is falling in this area, ``reward``
    is external feedback, and ``cost`` is effort spent.
    """

    novelty: float = 0.0
    surprise: float = 0.0
    learning_progress: float = 0.0
    reward: float = 0.0
    cost: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class AffectState:
    """Current affect, and how it modulates behaviour."""

    valence: float = SETPOINTS["valence"]
    arousal: float = SETPOINTS["arousal"]
    confidence: float = SETPOINTS["confidence"]
    curiosity: float = SETPOINTS["curiosity"]
    fatigue: float = SETPOINTS["fatigue"]

    # ------------------------------------------------------------------
    # dynamics
    # ------------------------------------------------------------------

    def decay(self, rate: float) -> "AffectState":
        """Relax every dimension toward its setpoint. Homeostasis, in one line."""
        for name, target in SETPOINTS.items():
            current = getattr(self, name)
            setattr(self, name, _clamp(name, current + (target - current) * rate))
        return self

    def update(self, appraisal: Appraisal, gain: float, decay: float) -> "AffectState":
        """Decay first, then apply this event's appraisal.

        The sign structure is the whole model, so it is worth stating plainly:
        novelty and surprise raise arousal; reward and learning progress raise
        valence while surprise lowers it; confidence tracks reward minus
        surprise (being wrong is what should shake it); curiosity is fed by
        learning progress and novelty and is *satiated* by their absence; effort
        accumulates as fatigue and nothing but rest removes it.
        """
        self.decay(decay)

        a = appraisal
        self.arousal = _clamp("arousal", self.arousal + gain * (0.6 * a.novelty + 0.7 * a.surprise))
        self.valence = _clamp(
            "valence",
            self.valence + gain * (a.reward + 0.5 * a.learning_progress - 0.4 * a.surprise),
        )
        self.confidence = _clamp(
            "confidence",
            self.confidence + gain * (0.6 * a.reward + 0.3 * a.learning_progress - 0.5 * a.surprise),
        )
        self.curiosity = _clamp(
            "curiosity",
            self.curiosity + gain * (0.7 * a.learning_progress + 0.5 * a.novelty - 0.25),
        )
        self.fatigue = _clamp("fatigue", self.fatigue + a.cost)
        return self

    def rest(self) -> "AffectState":
        """What sleep does to affect: fatigue clears, mood settles upward."""
        self.fatigue = 0.0
        self.arousal = _clamp("arousal", SETPOINTS["arousal"])
        self.valence = _clamp("valence", self.valence * 0.5 + 0.1)
        self.curiosity = _clamp("curiosity", max(self.curiosity, SETPOINTS["curiosity"]))
        return self

    # ------------------------------------------------------------------
    # effect on behaviour
    # ------------------------------------------------------------------

    def modulate(
        self,
        temperature: float,
        top_p: float = 1.0,
        top_k: int = 0,
        temperature_span: float = 0.45,
        top_p_span: float = 0.08,
    ) -> dict:
        """Bend decoding parameters by the current state.

        Aroused and curious widens the distribution; confident narrows it; tired
        narrows it further. The result is clamped to sane decoder ranges, so an
        extreme state degrades output diversity rather than breaking sampling.
        """
        drive = (
            0.5 * (self.arousal - SETPOINTS["arousal"])
            + 0.6 * (self.curiosity - SETPOINTS["curiosity"])
            - 0.5 * (self.confidence - SETPOINTS["confidence"])
            - 0.3 * self.fatigue
        )
        scale = 1.0 + temperature_span * max(-1.0, min(1.0, drive))
        out = {"temperature": round(max(0.05, min(2.0, temperature * scale)), 4)}

        if top_p:
            shifted = top_p + top_p_span * max(-1.0, min(1.0, drive))
            out["top_p"] = round(max(0.1, min(1.0, shifted)), 4)
        if top_k:
            widened = int(round(top_k * scale))
            out["top_k"] = max(1, widened)
        return out

    def describe(self) -> str:
        """A short first-person reading of the state, for prompts and traces."""
        parts = []
        if self.valence > 0.25:
            parts.append("in a good mood")
        elif self.valence < -0.25:
            parts.append("unsettled")
        if self.arousal > 0.6:
            parts.append("alert")
        elif self.arousal < 0.15:
            parts.append("subdued")
        if self.curiosity > 0.65:
            parts.append("curious")
        if self.confidence > 0.7:
            parts.append("sure of this")
        elif self.confidence < 0.3:
            parts.append("unsure")
        if self.fatigue > 0.6:
            parts.append("tired")
        return ", ".join(parts) if parts else "even"

    def to_dict(self) -> dict:
        return {k: round(v, 4) for k, v in asdict(self).items()}

    @classmethod
    def from_dict(cls, data: dict) -> "AffectState":
        known = {f.name for f in fields(cls)}
        return cls(**{k: float(v) for k, v in data.items() if k in known})

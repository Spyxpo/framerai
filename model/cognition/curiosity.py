"""Curiosity: what the mind goes looking for when nothing is asked of it.

Two signals, because either one alone misbehaves.

*Novelty* is random network distillation: a small predictor chases a frozen
random target on the same embedding, and how badly it misses is how unfamiliar
the input is. Training the predictor on everything it sees is habituation, so
the same input stops being interesting after a few exposures - which is exactly
what a novelty-only agent needs and never has when novelty is computed from a
static distance.

*Learning progress* is the derivative, not the level: how fast error is falling
in a given area (Oudeyer and Kaplan's intrinsic motivation). Novelty alone drags
a mind toward noise, because noise is maximally unpredictable forever. Learning
progress pulls it toward what it is currently getting better at, and away from
both the mastered and the hopeless.

The drive is a weighted blend of the two, and the frontier - what to explore
next - is sampled from it.
"""

from collections import deque

import torch
import torch.nn as nn

from .affect import Appraisal
from .config import CognitionConfig


def _mlp(d_in: int, d_hidden: int, d_out: int, generator: torch.Generator) -> nn.Sequential:
    """A small MLP with seeded init, so a mind reloads with the same instincts."""
    net = nn.Sequential(
        nn.Linear(d_in, d_hidden),
        nn.GELU(),
        nn.Linear(d_hidden, d_out),
    )
    with torch.no_grad():
        for module in net:
            if isinstance(module, nn.Linear):
                weight = torch.empty_like(module.weight)
                weight.normal_(0.0, 0.5, generator=generator)
                module.weight.copy_(weight)
                module.bias.zero_()
    return net


class NoveltyEstimator(nn.Module):
    """Random network distillation over experience embeddings.

    ``observe`` returns novelty in [0, 1] (how much more surprising this input is
    than the recent average) and the raw prediction error, which the learning
    progress tracker needs on its own scale.
    """

    def __init__(self, config: CognitionConfig):
        super().__init__()
        generator = torch.Generator().manual_seed(config.seed)
        d, h = config.d_embed, config.novelty_hidden
        self.target = _mlp(d, h, h, generator)
        for param in self.target.parameters():
            param.requires_grad_(False)
        self.predictor = _mlp(d, h, h, torch.Generator().manual_seed(config.seed + 1))
        self.optimizer = torch.optim.Adam(self.predictor.parameters(), lr=config.novelty_lr)
        self.ema = config.novelty_error_ema
        # Running first and second moments of the error, used to turn an
        # unbounded regression loss into a comparable novelty score.
        self.register_buffer("error_mean", torch.zeros(()))
        self.register_buffer("error_sq", torch.zeros(()))
        self.register_buffer("seen", torch.zeros((), dtype=torch.long))

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        vec = embedding.reshape(1, -1).float()
        with torch.no_grad():
            goal = self.target(vec)
        return ((self.predictor(vec) - goal) ** 2).mean()

    def observe(self, embedding: torch.Tensor, learn: bool = True) -> tuple[float, float]:
        """Score an embedding, and (by default) habituate to it."""
        loss = self.forward(embedding)
        raw = float(loss.detach())

        if learn:
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

        with torch.no_grad():
            if int(self.seen) == 0:
                self.error_mean.fill_(raw)
                self.error_sq.fill_(raw * raw)
            else:
                self.error_mean.mul_(self.ema).add_((1 - self.ema) * raw)
                self.error_sq.mul_(self.ema).add_((1 - self.ema) * raw * raw)
            self.seen += 1
            mean = float(self.error_mean)
            var = max(1e-12, float(self.error_sq) - mean * mean)

        # z-score, squashed: "unfamiliar relative to what I have been seeing".
        z = (raw - mean) / (var ** 0.5)
        novelty = float(torch.sigmoid(torch.tensor(z)))
        return novelty, raw

    def error_of(self, embedding: torch.Tensor) -> float:
        """Raw prediction error, with no learning. Falls as an input is rehearsed."""
        return float(self.forward(embedding).detach())

    def familiarity_of(self, embedding: torch.Tensor) -> float:
        """Novelty without learning from it - for probing, not experiencing.

        Relative, like novelty itself: this is familiarity *compared with what
        has been seen lately*, so it moves when the surrounding diet changes even
        if this input does not. Use :meth:`error_of` for the absolute measure.
        """
        raw = self.error_of(embedding)
        mean = float(self.error_mean)
        var = max(1e-12, float(self.error_sq) - mean * mean)
        return 1.0 - float(torch.sigmoid(torch.tensor((raw - mean) / (var ** 0.5))))


class LearningProgress:
    """Per-topic error history, and the slope that makes a topic interesting."""

    def __init__(self, config: CognitionConfig):
        self.window = config.learning_progress_window
        self._errors: dict[str, deque] = {}
        self._expected: dict[str, float] = {}

    def topics(self) -> list[str]:
        return list(self._errors)

    def subjects(self) -> list[str]:
        """Topics that name a subject, not a bookkeeping axis.

        The same event is filed under its subject and under ``lang:xx`` /
        ``sense:xx``. Those axes are for self-assessment - "I am worse at what I
        hear than at what I read" - and must never become things to go and
        explore, or the mind ends up setting itself the goal of understanding
        "sense:text".
        """
        return [t for t in self._errors if ":" not in t]

    def record(self, topic: str, error: float) -> None:
        history = self._errors.setdefault(topic, deque(maxlen=self.window))
        history.append(float(error))
        prior = self._expected.get(topic)
        self._expected[topic] = error if prior is None else 0.8 * prior + 0.2 * error

    def expected_error(self, topic: str) -> float | None:
        """What this topic usually costs to predict. None until it has a history."""
        return self._expected.get(topic)

    def progress(self, topic: str) -> float:
        """Fractional drop in error between the older and newer half of the window."""
        history = self._errors.get(topic)
        if not history or len(history) < 4:
            return 0.0
        values = list(history)
        half = len(values) // 2
        old = sum(values[:half]) / half
        new = sum(values[half:]) / (len(values) - half)
        if old <= 1e-12:
            return 0.0
        return float(max(0.0, min(1.0, (old - new) / old)))

    def competence(self, topic: str) -> float:
        """0 = every prediction here misses, 1 = this area is solved."""
        history = self._errors.get(topic)
        if not history:
            return 0.0
        mean = sum(history) / len(history)
        return float(1.0 / (1.0 + mean))

    def state_dict(self) -> dict:
        return {
            "errors": {k: list(v) for k, v in self._errors.items()},
            "expected": dict(self._expected),
        }

    def load_state_dict(self, state: dict) -> "LearningProgress":
        self._errors = {
            k: deque(v, maxlen=self.window) for k, v in state.get("errors", {}).items()
        }
        self._expected = dict(state.get("expected", {}))
        return self


class CuriosityEngine:
    """Novelty plus learning progress, and the questions they produce."""

    # Frames for a self-directed question. The mind picks one by its own state,
    # so an idle tick still produces something to go and find out.
    QUESTION_FRAMES = (
        "What do I still not understand about {topic}?",
        "What would change my mind about {topic}?",
        "Where does what I know about {topic} break down?",
        "What is the simplest thing about {topic} I have never checked?",
        "How does {topic} connect to what I learned before it?",
    )

    def __init__(self, config: CognitionConfig):
        self.config = config
        self.novelty = NoveltyEstimator(config)
        self.progress = LearningProgress(config)
        self._rng = torch.Generator().manual_seed(config.seed + 2)
        self.ticks = 0
        # Raw prediction error of the most recent appraisal, kept so callers can
        # file the same event under a second axis (language, modality) without
        # paying for a second forward pass.
        self.last_error = 0.0

    def appraise(self, embedding: torch.Tensor, topic: str, learn: bool = True) -> Appraisal:
        """Turn an embedding into the appraisal that drives affect and memory.

        Novelty and surprise are deliberately separate. Novelty asks "have I seen
        anything like this"; surprise asks "given that this is a ``topic`` thing,
        is it behaving as ``topic`` things usually do". A familiar subject acting
        out of character scores low on the first and high on the second.
        """
        expected = self.progress.expected_error(topic)
        novelty, raw = self.novelty.observe(embedding, learn=learn)

        if expected is None:
            surprise = novelty
        else:
            surprise = float(min(1.0, abs(raw - expected) / (expected + 1e-8)))

        self.progress.record(topic, raw)
        lp = self.progress.progress(topic)
        self.last_error = raw
        self.ticks += 1
        return Appraisal(novelty=novelty, surprise=surprise, learning_progress=lp)

    def drive(self, topic: str, novelty: float | None = None) -> float:
        """How much this topic pulls: unfamiliarity blended with learnability."""
        cfg = self.config
        n = novelty if novelty is not None else 1.0 - self.progress.competence(topic)
        raw = cfg.w_novelty * n + cfg.w_learning_progress * self.progress.progress(topic)
        return float(raw / (cfg.w_novelty + cfg.w_learning_progress))

    def frontier(self, topics: list[str] | None = None) -> str | None:
        """Sample the next thing to explore, proportional to its drive.

        Softmax rather than argmax so the mind does not lock onto one subject,
        and seeded so a reloaded mind is the same mind.
        """
        candidates = topics or self.progress.subjects()
        if not candidates:
            return None
        scores = torch.tensor([self.drive(t) for t in candidates])
        probs = torch.softmax(scores / self.config.curiosity_temperature, dim=0)
        return candidates[int(torch.multinomial(probs, 1, generator=self._rng))]

    def question(self, topic: str | None = None) -> str:
        """A question the mind asks itself, chosen deterministically by tick."""
        topic = topic or self.frontier() or "everything I have seen so far"
        frame = self.QUESTION_FRAMES[self.ticks % len(self.QUESTION_FRAMES)]
        return frame.format(topic=topic)

    def state_dict(self) -> dict:
        return {
            "novelty": self.novelty.state_dict(),
            "novelty_optimizer": self.novelty.optimizer.state_dict(),
            "progress": self.progress.state_dict(),
            "ticks": self.ticks,
            "rng": self._rng.get_state(),
        }

    def load_state_dict(self, state: dict) -> "CuriosityEngine":
        self.novelty.load_state_dict(state["novelty"])
        self.novelty.optimizer.load_state_dict(state["novelty_optimizer"])
        self.progress.load_state_dict(state["progress"])
        self.ticks = int(state.get("ticks", 0))
        if "rng" in state:
            self._rng.set_state(state["rng"])
        return self

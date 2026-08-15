"""Memory: what happened (episodic), what it means (semantic), what is in mind now.

Three stores with three different jobs.

``EpisodicMemory`` keeps individual events with their embedding, their affect at
the time, and a strength that decays with disuse and is restored by recall. That
decay is the point: a store that keeps everything at equal weight is a database,
and retrieving from it returns whatever is nearest regardless of whether it ever
mattered.

``SemanticMemory`` is what survives repetition. Episodes that keep landing in the
same region of embedding space collapse into a concept - a centroid, a count, and
the most central episode as its exemplar - so the mind ends up with generalities
it was never told.

``WorkingMemory`` is the small set of items currently in play, deliberately too
small to hold everything, which is what forces selection.
"""

import math
from dataclasses import asdict, dataclass, field, fields

import torch
import torch.nn.functional as F

from .config import CognitionConfig


@dataclass
class Experience:
    """One episode: something that happened, and what it was like."""

    tick: int
    text: str
    kind: str = "perception"
    topic: str = "general"
    # Which modality it arrived in, and which language it was in. Both are
    # remembered because the mind tracks competence separately per language and
    # per sense: being good at English text says nothing about Tamil audio.
    modality: str = "text"
    language: str = "und"
    novelty: float = 0.0
    surprise: float = 0.0
    reward: float = 0.0
    salience: float = 0.0
    affect: dict = field(default_factory=dict)
    # Strength is the memory trace, not the content: it grows on recall and
    # decays with disuse, and it decides what is forgotten under pressure.
    strength: float = 1.0
    uses: int = 0
    last_used: int = 0
    slot: int = -1

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Experience":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class MemoryHit:
    """A retrieved episode with the score components that surfaced it."""

    experience: Experience
    score: float
    similarity: float
    recency: float
    salience: float


def default_salience(novelty: float, surprise: float, reward: float) -> float:
    """How much an event deserves to be remembered.

    Novel, surprising, and rewarding (or punishing - magnitude, not sign) events
    are the ones worth keeping. A flat 1.0 here would make forgetting arbitrary.
    """
    raw = 0.5 * novelty + 0.35 * surprise + 0.45 * abs(reward)
    return float(max(0.0, min(1.0, raw)))


class EpisodicMemory:
    """Bounded store of episodes, retrieved by similarity x recency x salience."""

    def __init__(self, config: CognitionConfig):
        self.config = config
        self.capacity = config.episodic_capacity
        self.d_embed = config.d_embed
        # Embeddings live in one preallocated matrix so retrieval is a single
        # matmul instead of a Python loop over episodes.
        self._embeds = torch.zeros(self.capacity, self.d_embed)
        self._items: list[Experience] = []

    def __len__(self) -> int:
        return len(self._items)

    def all(self) -> list[Experience]:
        return list(self._items)

    # ------------------------------------------------------------------
    # writing
    # ------------------------------------------------------------------

    def write(self, embedding: torch.Tensor, experience: Experience) -> Experience:
        """Store an episode, evicting the weakest trace when full."""
        vec = F.normalize(embedding.detach().reshape(-1).float(), dim=0, eps=1e-8)
        if vec.numel() != self.d_embed:
            raise ValueError(f"expected a {self.d_embed}-d embedding, got {vec.numel()}")

        if not experience.salience:
            experience.salience = default_salience(
                experience.novelty, experience.surprise, experience.reward
            )
        experience.last_used = experience.tick

        if len(self._items) < self.capacity:
            slot = len(self._items)
            experience.slot = slot
            self._items.append(experience)
        else:
            slot = self._weakest_slot(experience.tick)
            experience.slot = slot
            self._items[slot] = experience

        self._embeds[slot] = vec
        return experience

    def _weakest_slot(self, now: int) -> int:
        """Index of the trace with the least to lose."""
        scored = [(self.trace_strength(item, now) * (1.0 + item.salience), i)
                  for i, item in enumerate(self._items)]
        return min(scored)[1]

    # ------------------------------------------------------------------
    # decay and retrieval
    # ------------------------------------------------------------------

    def trace_strength(self, experience: Experience, now: int) -> float:
        """Strength after forgetting: halves every ``strength_half_life`` ticks unused."""
        elapsed = max(0, now - experience.last_used)
        return experience.strength * 0.5 ** (elapsed / self.config.strength_half_life)

    def retrieve(
        self,
        embedding: torch.Tensor,
        now: int,
        k: int | None = None,
        topic: str | None = None,
        reinforce: bool = True,
    ) -> list[MemoryHit]:
        """Recall the k best episodes for this cue.

        Recall is not free of consequence: every returned episode is strengthened
        and stamped as used, which is what makes repeatedly useful memories
        durable and lets the rest fade.
        """
        if not self._items:
            return []

        cfg = self.config
        k = k or cfg.retrieval_k
        query = F.normalize(embedding.detach().reshape(-1).float(), dim=0, eps=1e-8)
        n = len(self._items)
        sims = (self._embeds[:n] @ query).tolist()

        hits: list[MemoryHit] = []
        for i, item in enumerate(self._items):
            if topic and item.topic != topic:
                continue
            recency = 0.5 ** (max(0, now - item.tick) / cfg.recency_half_life)
            strength = self.trace_strength(item, now)
            score = (
                cfg.w_similarity * sims[i]
                + cfg.w_recency * recency
                + cfg.w_salience * item.salience
            ) * (0.5 + 0.5 * min(1.0, strength))
            if score < cfg.retrieval_floor:
                continue
            hits.append(MemoryHit(item, score, sims[i], recency, item.salience))

        hits.sort(key=lambda h: h.score, reverse=True)
        hits = hits[:k]

        if reinforce:
            for hit in hits:
                hit.experience.strength = min(4.0, hit.experience.strength + cfg.retrieval_boost)
                hit.experience.uses += 1
                hit.experience.last_used = now
        return hits

    def embedding_of(self, experience: Experience) -> torch.Tensor:
        return self._embeds[experience.slot].clone()

    def sample(self, n: int, now: int, alpha: float, generator=None) -> list[Experience]:
        """Prioritised sample for replay, weighted by salience and trace strength."""
        if not self._items:
            return []
        weights = torch.tensor([
            (item.salience + self.trace_strength(item, now) + 1e-3) ** alpha
            for item in self._items
        ])
        n = min(n, len(self._items))
        idx = torch.multinomial(weights, n, replacement=False, generator=generator)
        return [self._items[int(i)] for i in idx]

    def decay_all(self, factor: float, now: int) -> int:
        """Apply global decay, dropping traces that fall below usefulness.

        Returns the number of episodes forgotten. Called during sleep: the day's
        weak traces go, the reinforced ones stay.
        """
        survivors: list[Experience] = []
        kept_embeds = []
        for item in self._items:
            item.strength = self.trace_strength(item, now) * factor
            item.last_used = now
            if item.strength < 0.05 and item.salience < 0.2:
                continue
            kept_embeds.append(self._embeds[item.slot].clone())
            survivors.append(item)

        forgotten = len(self._items) - len(survivors)
        for new_slot, (item, vec) in enumerate(zip(survivors, kept_embeds, strict=True)):
            item.slot = new_slot
            self._embeds[new_slot] = vec
        self._items = survivors
        return forgotten

    def stats(self, now: int) -> dict:
        if not self._items:
            return {"episodes": 0, "capacity": self.capacity}
        strengths = [self.trace_strength(i, now) for i in self._items]
        return {
            "episodes": len(self._items),
            "capacity": self.capacity,
            "mean_strength": round(sum(strengths) / len(strengths), 4),
            "mean_salience": round(sum(i.salience for i in self._items) / len(self._items), 4),
            "topics": len({i.topic for i in self._items}),
            "oldest_tick": min(i.tick for i in self._items),
        }

    def state_dict(self) -> dict:
        n = len(self._items)
        return {
            "embeds": self._embeds[:n].clone(),
            "items": [i.to_dict() for i in self._items],
        }

    def load_state_dict(self, state: dict) -> "EpisodicMemory":
        self._items = [Experience.from_dict(d) for d in state["items"]]
        self._embeds = torch.zeros(self.capacity, self.d_embed)
        stored = state["embeds"]
        if stored.numel():
            self._embeds[: stored.shape[0]] = stored
        return self


@dataclass
class Concept:
    """A generality the mind was never told: repeated episodes, collapsed."""

    label: str
    exemplar: str
    visits: int = 1
    first_tick: int = 0
    last_tick: int = 0
    topics: dict = field(default_factory=dict)
    # Salience of the episode currently standing in for the concept, so the
    # exemplar is the most striking member rather than the most recent one.
    exemplar_salience: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Concept":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


class SemanticMemory:
    """Online concept formation over episode embeddings.

    An episode either joins the nearest concept it is close enough to - moving
    that concept's centroid a little, the way a running mean does - or starts a
    new one. No clustering pass, no retraining: the structure accretes.
    """

    def __init__(self, config: CognitionConfig):
        self.config = config
        self.capacity = config.semantic_capacity
        self._centroids = torch.zeros(self.capacity, config.d_embed)
        self._concepts: list[Concept] = []

    def __len__(self) -> int:
        return len(self._concepts)

    def all(self) -> list[Concept]:
        return list(self._concepts)

    def integrate(self, embedding: torch.Tensor, experience: Experience) -> tuple[Concept, bool]:
        """Fold an episode into the concept map. Returns (concept, is_new)."""
        vec = F.normalize(embedding.detach().reshape(-1).float(), dim=0, eps=1e-8)
        n = len(self._concepts)

        if n:
            sims = self._centroids[:n] @ vec
            best = int(torch.argmax(sims))
            if float(sims[best]) >= self.config.concept_threshold:
                concept = self._concepts[best]
                concept.visits += 1
                concept.last_tick = experience.tick
                concept.topics[experience.topic] = concept.topics.get(experience.topic, 0) + 1
                # Running mean: each visit moves the centroid 1/visits of the way.
                blended = self._centroids[best] + (vec - self._centroids[best]) / concept.visits
                self._centroids[best] = F.normalize(blended, dim=0, eps=1e-8)
                if experience.salience > concept.exemplar_salience:
                    concept.exemplar = experience.text
                    concept.exemplar_salience = experience.salience
                return concept, False

        concept = Concept(
            label=experience.topic,
            exemplar=experience.text,
            first_tick=experience.tick,
            last_tick=experience.tick,
            topics={experience.topic: 1},
            exemplar_salience=experience.salience,
        )
        if n < self.capacity:
            slot = n
            self._concepts.append(concept)
        else:
            slot = min(range(n), key=lambda i: (self._concepts[i].visits, self._concepts[i].last_tick))
            self._concepts[slot] = concept
        self._centroids[slot] = vec
        return concept, True

    def recall(self, embedding: torch.Tensor, k: int | None = None) -> list[tuple[Concept, float]]:
        if not self._concepts:
            return []
        k = k or self.config.concept_recall_k
        vec = F.normalize(embedding.detach().reshape(-1).float(), dim=0, eps=1e-8)
        n = len(self._concepts)
        sims = (self._centroids[:n] @ vec).tolist()
        ranked = sorted(zip(self._concepts, sims, strict=True), key=lambda p: p[1], reverse=True)
        return ranked[:k]

    def state_dict(self) -> dict:
        n = len(self._concepts)
        return {
            "centroids": self._centroids[:n].clone(),
            "concepts": [c.to_dict() for c in self._concepts],
        }

    def load_state_dict(self, state: dict) -> "SemanticMemory":
        self._concepts = [Concept.from_dict(d) for d in state["concepts"]]
        self._centroids = torch.zeros(self.capacity, self.config.d_embed)
        stored = state["centroids"]
        if stored.numel():
            self._centroids[: stored.shape[0]] = stored
        return self


class WorkingMemory:
    """The few items currently in mind. Bounded, so it has to choose."""

    def __init__(self, config: CognitionConfig):
        self.capacity = config.working_memory_size
        self._items: list[Experience] = []
        self.focus: str = ""

    def __len__(self) -> int:
        return len(self._items)

    def push(self, experience: Experience) -> None:
        self._items.append(experience)
        if len(self._items) > self.capacity:
            # Drop the least salient item rather than simply the oldest, so
            # something striking survives a few turns of small talk. The newest
            # item is never a candidate - it is what is being attended to.
            victim = min(self._items[:-1], key=lambda e: (e.salience, e.tick))
            self._items.remove(victim)

    def items(self) -> list[Experience]:
        return list(self._items)

    def render(self, budget: int = 400) -> str:
        lines = []
        used = 0
        for item in reversed(self._items):
            line = f"[{item.kind}] {item.text.strip()}"
            if used + len(line) > budget:
                break
            lines.append(line)
            used += len(line)
        return "\n".join(reversed(lines))

    def clear(self) -> None:
        self._items.clear()

    def state_dict(self) -> dict:
        return {"items": [i.to_dict() for i in self._items], "focus": self.focus}

    def load_state_dict(self, state: dict) -> "WorkingMemory":
        self._items = [Experience.from_dict(d) for d in state["items"]][-self.capacity:]
        self.focus = state.get("focus", "")
        return self


def half_life_decay(elapsed: float, half_life: float) -> float:
    """Exponential forgetting curve, exposed for tests and for the trainer."""
    if half_life <= 0:
        raise ValueError("half_life must be > 0")
    return math.pow(0.5, max(0.0, elapsed) / half_life)

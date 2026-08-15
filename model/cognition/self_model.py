"""The model the mind keeps of itself.

Not a personality blurb. This tracks three things that change with experience and
that the mind can be wrong about: what it is competent at, what it keeps being
drawn to, and what it is currently trying to do. Reflection writes a first-person
line into a bounded narrative, which is the only place the mind's history exists
as language rather than as vectors - and it is what gets injected back into the
prompt, so the mind's account of itself feeds its next answer.
"""

from dataclasses import asdict, dataclass, field, fields


@dataclass
class Goal:
    """Something the mind is trying to do, and how far along it is."""

    text: str
    priority: float = 0.5
    progress: float = 0.0
    created_tick: int = 0
    updated_tick: int = 0
    origin: str = "self"  # "self" when curiosity raised it, "user" when asked

    @property
    def done(self) -> bool:
        return self.progress >= 1.0

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "Goal":
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})


@dataclass
class SelfModel:
    """Competence, interests, goals, and a narrative of what has happened."""

    name: str = "FramerAI"
    competence: dict = field(default_factory=dict)
    interests: dict = field(default_factory=dict)
    goals: list = field(default_factory=list)
    narrative: list = field(default_factory=list)
    narrative_capacity: int = 64
    exchanges: int = 0
    sleeps: int = 0
    # How much of the mind's life has arrived in each language and each sense.
    # Counted separately from competence: exposure and skill are not the same
    # thing, and the gap between them is what curiosity should chase.
    languages: dict = field(default_factory=dict)
    unidentified_languages: int = 0
    modalities: dict = field(default_factory=dict)

    # ------------------------------------------------------------------
    # updating
    # ------------------------------------------------------------------

    def observe(self, topic: str, competence: float, drive: float, alpha: float = 0.2) -> None:
        """Fold one encounter into the self-estimate, as an EMA per topic."""
        prior_c = self.competence.get(topic, competence)
        prior_i = self.interests.get(topic, drive)
        self.competence[topic] = round((1 - alpha) * prior_c + alpha * competence, 4)
        self.interests[topic] = round((1 - alpha) * prior_i + alpha * drive, 4)

    def note_language(self, code: str, identified: bool = True) -> None:
        """Record exposure to a language. Unidentified input is counted, not hidden."""
        if identified:
            self.languages[code] = self.languages.get(code, 0) + 1
        else:
            self.unidentified_languages += 1

    def note_modality(self, modality: str) -> None:
        self.modalities[modality] = self.modalities.get(modality, 0) + 1

    def known_languages(self, n: int = 5) -> list[tuple[str, int]]:
        return sorted(self.languages.items(), key=lambda kv: kv[1], reverse=True)[:n]

    def add_goal(self, text: str, tick: int, priority: float = 0.5, origin: str = "self") -> Goal:
        """Add a goal, or raise the priority of one already held."""
        for goal in self.goals:
            if goal.text == text:
                goal.priority = max(goal.priority, priority)
                goal.updated_tick = tick
                return goal
        goal = Goal(
            text=text, priority=priority, created_tick=tick, updated_tick=tick, origin=origin
        )
        self.goals.append(goal)
        return goal

    def advance_goal(self, text: str, delta: float, tick: int) -> Goal | None:
        for goal in self.goals:
            if goal.text == text:
                goal.progress = max(0.0, min(1.0, goal.progress + delta))
                goal.updated_tick = tick
                return goal
        return None

    def drop_completed(self) -> int:
        before = len(self.goals)
        self.goals = [g for g in self.goals if not g.done]
        return before - len(self.goals)

    def top_goals(self, n: int = 3) -> list[Goal]:
        return sorted(
            (g for g in self.goals if not g.done),
            key=lambda g: (g.priority, -g.progress),
            reverse=True,
        )[:n]

    def strongest_interests(self, n: int = 3) -> list[tuple[str, float]]:
        return sorted(self.interests.items(), key=lambda kv: kv[1], reverse=True)[:n]

    def weakest_areas(self, n: int = 3) -> list[tuple[str, float]]:
        return sorted(self.competence.items(), key=lambda kv: kv[1])[:n]

    # ------------------------------------------------------------------
    # language
    # ------------------------------------------------------------------

    def remember_reflection(self, tick: int, text: str) -> str:
        self.narrative.append({"tick": tick, "text": text})
        if len(self.narrative) > self.narrative_capacity:
            del self.narrative[0]
        return text

    def reflect(self, tick: int, affect, memory_stats: dict, concepts: int = 0) -> str:
        """Write one first-person line about the current state of things."""
        interests = ", ".join(t for t, _ in self.strongest_interests(2)) or "nothing in particular"
        weak = ", ".join(t for t, _ in self.weakest_areas(1)) or "nothing yet"
        episodes = memory_stats.get("episodes", 0)
        line = (
            f"After {self.exchanges} exchanges I hold {episodes} episodes and {concepts} concepts. "
            f"I keep returning to {interests}; I am weakest at {weak}. I feel {affect.describe()}."
        )
        return self.remember_reflection(tick, line)

    def summary(self, budget: int = 400) -> str:
        """The identity preamble injected ahead of a prompt."""
        lines = [f"I am {self.name}."]
        interests = self.strongest_interests(3)
        if interests:
            lines.append("Drawn to: " + ", ".join(f"{t} ({v:.2f})" for t, v in interests) + ".")
        goals = self.top_goals(2)
        if goals:
            lines.append(
                "Trying to: " + "; ".join(f"{g.text} ({g.progress:.0%} in)" for g in goals) + "."
            )
        if self.narrative:
            lines.append("Last reflection: " + self.narrative[-1]["text"])
        text = " ".join(lines)
        return text[:budget]

    def to_dict(self) -> dict:
        data = asdict(self)
        data["goals"] = [g.to_dict() if isinstance(g, Goal) else g for g in self.goals]
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "SelfModel":
        known = {f.name for f in fields(cls)}
        payload = {k: v for k, v in data.items() if k in known}
        payload["goals"] = [Goal.from_dict(g) for g in payload.get("goals", [])]
        return cls(**payload)

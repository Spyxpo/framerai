"""Sleep: replay, consolidation, forgetting.

A mind that only ever writes memories accumulates; it does not learn. Sleep is
where the day's episodes get sampled in proportion to how much they mattered,
rehearsed through the novelty predictor (so what was striking becomes familiar),
folded into concepts, and thinned out. Traces that were never used decay away;
the ones that keep being retrieved survive.

The optional ``train_step`` callback is the bridge from experience to weights: it
receives the replayed episodes and may run a real gradient step on the backbone.
Everything else here works with no model at all, which is what keeps consolidation
testable in milliseconds.
"""

import torch

from .config import CognitionConfig


class Consolidator:
    """Prioritised experience replay, concept formation, and forgetting."""

    def __init__(self, config: CognitionConfig):
        self.config = config
        self._rng = torch.Generator().manual_seed(config.seed + 3)

    def should_sleep(self, affect) -> bool:
        """Sleep pressure is fatigue, and fatigue is spent effort."""
        return affect.fatigue >= self.config.sleep_threshold

    def sleep(
        self,
        tick: int,
        episodic,
        semantic,
        curiosity,
        self_model,
        affect,
        train_step=None,
    ) -> dict:
        """Run one consolidation pass. Returns a report of what it did."""
        cfg = self.config
        batch = episodic.sample(cfg.replay_batch, tick, cfg.replay_alpha, generator=self._rng)

        rehearsed = 0
        new_concepts = 0
        for experience in batch:
            embedding = episodic.embedding_of(experience)
            # Rehearsal habituates: replaying an episode lowers its novelty next
            # time, which is why a mind stops being startled by its own history.
            curiosity.novelty.observe(embedding, learn=True)
            _, is_new = semantic.integrate(embedding, experience)
            new_concepts += int(is_new)
            rehearsed += 1

        training = None
        if train_step is not None and batch:
            training = train_step(batch)

        forgotten = episodic.decay_all(cfg.consolidation_decay, tick)

        self_model.sleeps += 1
        self_model.drop_completed()
        stats = episodic.stats(tick)
        reflection = self_model.reflect(tick, affect, stats, concepts=len(semantic))
        affect.rest()

        return {
            "tick": tick,
            "rehearsed": rehearsed,
            "new_concepts": new_concepts,
            "forgotten": forgotten,
            "concepts": len(semantic),
            "episodes": stats.get("episodes", 0),
            "training": training,
            "reflection": reflection,
        }

    def state_dict(self) -> dict:
        return {"rng": self._rng.get_state()}

    def load_state_dict(self, state: dict) -> "Consolidator":
        if "rng" in state:
            self._rng.set_state(state["rng"])
        return self

"""Turning experience into a vector the mind can compare, store, and recall.

Everything downstream - memory retrieval, concept formation, novelty - works on
one fixed-width vector per experience, whatever modality it arrived in. This
module is the only place that knows how those vectors are made.

With a model attached, an experience is encoded by the backbone itself: the LM's
hidden states (or the vision/audio towers) mean-pooled and projected down. The
mind then remembers things the way the model understands them, and its sense of
"similar" moves as the model learns.

With no model, a seeded hashing encoder stands in. It is not semantic, but it is
stable, cheap, and similar-for-similar-text, which is enough to exercise and test
every other part of the mind without a checkpoint.
"""

import hashlib
import re

import torch
import torch.nn.functional as F

_WORD = re.compile(r"[a-z0-9']+")


def _tokens(text: str) -> list[str]:
    """Word unigrams plus character trigrams, so near-identical text stays near."""
    lowered = text.lower()
    words = _WORD.findall(lowered)
    trigrams = [lowered[i:i + 3] for i in range(max(0, len(lowered) - 2))]
    return words + trigrams


def _bucket(token: str, dim: int) -> tuple[int, float]:
    """Stable hash to (index, sign). blake2b, not hash(), which is salted per process."""
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    value = int.from_bytes(digest, "big")
    return value % dim, 1.0 if (value >> 63) & 1 else -1.0


def hash_embed(text: str, dim: int) -> torch.Tensor:
    """Signed hashing-trick embedding. Deterministic across runs and machines."""
    vec = torch.zeros(dim)
    for token in _tokens(text):
        idx, sign = _bucket(token, dim)
        vec[idx] += sign
    return F.normalize(vec, dim=0, eps=1e-8)


class ExperienceEncoder:
    """Encode text, images, and audio into the mind's shared embedding space."""

    def __init__(
        self,
        d_embed: int,
        model=None,
        tokenizer=None,
        device: str = "cpu",
        seed: int = 42,
    ):
        self.d_embed = d_embed
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.seed = seed
        self._projection: torch.Tensor | None = None

    @property
    def grounded(self) -> bool:
        """True when experiences are encoded by the model rather than by hashing."""
        return self.model is not None and self.tokenizer is not None

    def _project(self, hidden: torch.Tensor) -> torch.Tensor:
        """Mean-pool a (B, T, D) hidden state and project it to the mind's width.

        The projection is a fixed seeded Gaussian, not a learned layer: random
        projection preserves relative distances (Johnson-Lindenstrauss) without
        adding a second thing that has to be trained, and it keeps the mind's
        geometry stable across restarts.
        """
        pooled = hidden.mean(dim=1).reshape(-1).float()
        if self._projection is None or self._projection.shape[0] != pooled.numel():
            generator = torch.Generator().manual_seed(self.seed)
            weight = torch.empty(pooled.numel(), self.d_embed)
            weight.normal_(0.0, 1.0 / (self.d_embed ** 0.5), generator=generator)
            self._projection = weight
        return F.normalize(pooled @ self._projection, dim=0, eps=1e-8)

    @torch.no_grad()
    def encode_text(self, text: str) -> torch.Tensor:
        if not self.grounded:
            return hash_embed(text, self.d_embed)

        ids = self.tokenizer.encode(text, add_special=True)
        limit = getattr(self.model.config, "max_seq_len", len(ids))
        if len(ids) > limit:
            # Keep the tail: the most recent part of an experience is the part
            # the rest of it was leading to.
            ids = ids[-limit:]
        if not ids:
            return hash_embed(text, self.d_embed)

        tensor = torch.tensor([ids], device=self.device)
        hidden = self.model.forward_lm(tensor)["hidden"]
        return self._project(hidden.cpu())

    @torch.no_grad()
    def encode_image(self, image: torch.Tensor) -> torch.Tensor:
        """Encode a (C, H, W) or (B, C, H, W) image through the vision tower."""
        if not self.grounded or getattr(self.model, "text_only", True):
            raise RuntimeError("image encoding needs a multimodal model (text_only=False)")
        batch = image if image.dim() == 4 else image.unsqueeze(0)
        embeds = self.model.forward_vision(batch.to(self.device))
        return self._project(embeds.cpu())

    @torch.no_grad()
    def encode_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """Encode a waveform or log-mel input through the audio tower."""
        if not self.grounded or getattr(self.model, "text_only", True):
            raise RuntimeError("audio encoding needs a multimodal model (text_only=False)")
        batch = audio if audio.dim() >= 2 else audio.unsqueeze(0)
        embeds = self.model.forward_audio(batch.to(self.device))
        return self._project(embeds.cpu())

    @property
    def multimodal(self) -> bool:
        """True when the vision and audio towers are actually there to be used."""
        return self.grounded and not getattr(self.model, "text_only", True)

    def encode_signal(self, signal: torch.Tensor) -> torch.Tensor:
        """Encode a raw sensor tensor without a model: pooled, not understood.

        A live camera or microphone has to produce *some* comparable vector even
        on a text-only or untrained build, or the whole streaming path would
        only run against a full multimodal checkpoint. Average-pooling the
        flattened signal is not perception - it carries no semantics - but it
        does track change, which is what the attention gate needs.
        """
        flat = signal.detach().reshape(1, 1, -1).float()
        width = self.d_embed - 4 if self.d_embed > 8 else self.d_embed
        pooled = torch.nn.functional.adaptive_avg_pool1d(flat, width).reshape(-1)

        if width < self.d_embed:
            # Two halves, because either alone is blind to something. The pooled
            # signal minus its mean carries structure - where the light or the
            # sound sits - but cosine similarity is scale-invariant, so on its
            # own a black frame and a grey one point the same way. Four bounded
            # level statistics carry what that discards, which is how "the lights
            # came on" and "the room went quiet" reach the attention gate.
            structure = pooled - pooled.mean()
            structure = structure / structure.norm().clamp_min(1e-8)
            raw = flat.reshape(-1)
            stats = torch.tanh(torch.stack([
                raw.mean(), raw.std(unbiased=False), raw.amin(), raw.amax(),
            ]))
            pooled = torch.cat([structure, stats])

        if float(pooled.norm()) < 1e-6:
            # A black frame or a silent chunk pools to the zero vector, which has
            # no direction and would read as maximally different from everything
            # including itself. Silence is a percept: give it a fixed one.
            pooled = torch.ones(self.d_embed)
        return F.normalize(pooled, dim=0, eps=1e-8)

    def encode_sensor(self, signal: torch.Tensor, modality: str) -> torch.Tensor:
        """Best available encoding for a sensor reading, tower first, pooling second."""
        if self.multimodal:
            if modality == "vision":
                return self.encode_image(signal)
            if modality == "hearing":
                return self.encode_audio(signal)
        return self.encode_signal(signal)

"""Acoustic-token heads, speaker conditioning, and the CTC auxiliary head.

With audio quantized into discrete tokens, generation becomes classification
over codebook entries rather than regression onto a spectrogram - the same shape
of problem the backbone already solves for text.

Also here: the pieces that make audio *controllable* rather than merely
generated. A speaker embedding from a reference clip conditions timbre, and a
CTC head over the audio encoder makes transcription trained rather than hoped
for. Transcription previously worked by prepending audio embeddings and asking
the language model nicely.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AudioTokenEmbedding(nn.Module):
    """Embed a stack of residual codes as one vector per frame.

    Each quantizer gets its own slice of a shared table, so code 5 from
    quantizer 0 and code 5 from quantizer 3 are different entries. The stack is
    summed, which is what makes the residual decomposition additive in embedding
    space as well as in latent space.
    """

    def __init__(self, n_quantizers: int, codebook_size: int, d_model: int):
        super().__init__()
        self.n_quantizers = n_quantizers
        self.codebook_size = codebook_size
        self.embedding = nn.Embedding(n_quantizers * codebook_size, d_model)
        self.register_buffer(
            "offsets", torch.arange(n_quantizers).unsqueeze(-1) * codebook_size
        )

    @torch.no_grad()
    def reset_buffers(self, device=None):
        device = device or self.offsets.device
        self.offsets = (
            torch.arange(self.n_quantizers, device=device).unsqueeze(-1) * self.codebook_size
        )

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        """``codes`` is ``(B, n_q, T)``; returns ``(B, T, d_model)``."""
        n_q = codes.shape[1]
        return self.embedding(codes + self.offsets[:n_q]).sum(dim=1)


class AudioTokenHead(nn.Module):
    """Predict every quantizer's next code from one hidden state."""

    def __init__(self, d_model: int, n_quantizers: int, codebook_size: int):
        super().__init__()
        self.n_quantizers = n_quantizers
        self.codebook_size = codebook_size
        self.proj = nn.Linear(d_model, n_quantizers * codebook_size)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        """``hidden`` is ``(B, T, d_model)``; returns ``(B, n_q, T, codebook)``."""
        B, T, _ = hidden.shape
        logits = self.proj(hidden).view(B, T, self.n_quantizers, self.codebook_size)
        return logits.permute(0, 2, 1, 3)

    def loss(self, hidden: torch.Tensor, codes: torch.Tensor) -> torch.Tensor:
        """Cross-entropy over every quantizer, averaged."""
        logits = self(hidden)[:, : codes.shape[1]]
        return F.cross_entropy(
            logits.reshape(-1, self.codebook_size), codes.reshape(-1)
        )


class SpeakerEncoder(nn.Module):
    """A fixed-length speaker embedding from a variable-length reference clip.

    Mean and standard deviation pooling rather than mean alone: timbre is as much
    about the variation across a clip as its average.
    """

    def __init__(self, in_channels: int, d_model: int = 256, n_layers: int = 3):
        super().__init__()
        layers, channels = [], in_channels
        for _ in range(n_layers):
            layers += [nn.Conv1d(channels, d_model, 5, padding=2), nn.ReLU()]
            channels = d_model
        self.trunk = nn.Sequential(*layers)
        self.proj = nn.Linear(d_model * 2, d_model)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """``features`` is ``(B, C, T)``; returns ``(B, d_model)``."""
        h = self.trunk(features)
        pooled = torch.cat([h.mean(dim=-1), h.std(dim=-1)], dim=-1)
        return F.normalize(self.proj(pooled), dim=-1)


class CTCHead(nn.Module):
    """Connectionist temporal classification over the audio encoder output.

    Alignment-free, so transcription can be trained on audio and text without
    per-frame labels. Index 0 is the blank symbol.
    """

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.proj = nn.Linear(d_model, vocab_size)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return F.log_softmax(self.proj(hidden), dim=-1)

    def loss(
        self,
        hidden: torch.Tensor,
        targets: torch.Tensor,
        input_lengths: torch.Tensor,
        target_lengths: torch.Tensor,
        blank: int = 0,
    ) -> torch.Tensor:
        # CTC wants (T, B, V).
        log_probs = self(hidden).transpose(0, 1)
        return F.ctc_loss(
            log_probs, targets, input_lengths, target_lengths,
            blank=blank, zero_infinity=True,
        )

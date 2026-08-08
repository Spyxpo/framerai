"""Residual vector quantization.

Diffusing a mel spectrogram and inverting it with Griffin-Lim caps audio quality
no matter how well the model is trained: the phase is discarded and then guessed.
Quantizing audio into discrete tokens instead lets the language model predict
acoustics directly, the same way it predicts text, and lets a learned decoder
reconstruct the waveform including phase.

A single codebook large enough for high-fidelity audio would be unusable. Residual
quantization stacks small codebooks, each encoding what the previous ones missed,
so ``n_quantizers`` codebooks of ``codebook_size`` entries span
``codebook_size ** n_quantizers`` effective states while every lookup stays cheap.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class VectorQuantizer(nn.Module):
    """One codebook, updated by exponential moving average.

    EMA rather than a gradient on the codebook: the assignment is
    non-differentiable, and the straight-through estimator that carries gradient
    to the encoder does not carry a useful one to the entries themselves.

    Codes that stop being chosen are restarted from live encoder outputs. Without
    that, a codebook trained from a poor initialization keeps most of its
    capacity permanently unused.
    """

    def __init__(
        self,
        codebook_size: int = 1024,
        codebook_dim: int = 256,
        decay: float = 0.99,
        eps: float = 1e-5,
        restart_threshold: float = 1.0,
    ):
        super().__init__()
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.decay = decay
        self.eps = eps
        self.restart_threshold = restart_threshold

        self.register_buffer("codebook", torch.randn(codebook_size, codebook_dim))
        self.register_buffer("cluster_size", torch.ones(codebook_size))
        self.register_buffer("embed_avg", torch.randn(codebook_size, codebook_dim))

    @torch.no_grad()
    def reset_buffers(self, device=None):
        """Re-seed the codebook. Deterministic init, so a meta build materializes."""
        device = device or self.codebook.device
        generator = torch.Generator(device="cpu").manual_seed(0)
        codebook = torch.randn(
            self.codebook_size, self.codebook_dim, generator=generator
        ).to(device)
        self.codebook = codebook
        self.embed_avg = codebook.clone()
        self.cluster_size = torch.ones(self.codebook_size, device=device)

    def _distances(self, flat: torch.Tensor) -> torch.Tensor:
        # ||x - e||^2 expanded, dropping the constant ||x||^2 term.
        return (
            flat.pow(2).sum(1, keepdim=True)
            - 2 * flat @ self.codebook.t()
            + self.codebook.pow(2).sum(1)
        )

    @torch.no_grad()
    def _update(self, flat: torch.Tensor, onehot: torch.Tensor):
        counts = onehot.sum(0)
        embed_sum = onehot.t() @ flat
        self.cluster_size.mul_(self.decay).add_(counts, alpha=1 - self.decay)
        self.embed_avg.mul_(self.decay).add_(embed_sum, alpha=1 - self.decay)

        total = self.cluster_size.sum()
        smoothed = (
            (self.cluster_size + self.eps) / (total + self.codebook_size * self.eps) * total
        )
        self.codebook.copy_(self.embed_avg / smoothed.unsqueeze(1))

        # Restart dead entries from randomly chosen live encoder outputs.
        dead = self.cluster_size < self.restart_threshold
        if dead.any() and flat.shape[0] > 0:
            picks = torch.randint(0, flat.shape[0], (int(dead.sum()),), device=flat.device)
            self.codebook[dead] = flat[picks]
            self.cluster_size[dead] = 1.0
            self.embed_avg[dead] = flat[picks]

    def forward(self, z: torch.Tensor) -> tuple:
        """``z`` is ``(B, D, T)``. Returns ``(quantized, indices, commit_loss)``."""
        B, D, T = z.shape
        flat = z.transpose(1, 2).reshape(-1, D)

        indices = self._distances(flat).argmin(dim=1)
        quantized = self.codebook[indices].view(B, T, D).transpose(1, 2)

        if self.training:
            onehot = F.one_hot(indices, self.codebook_size).float()
            self._update(flat, onehot)

        # Pull the encoder toward the codebook; the codebook moves by EMA.
        commit_loss = F.mse_loss(z, quantized.detach())
        # Straight-through: gradient flows to the encoder as if quantization
        # were the identity.
        quantized = z + (quantized - z).detach()
        return quantized, indices.view(B, T), commit_loss


class ResidualVQ(nn.Module):
    """A stack of quantizers, each encoding the previous stage's residual."""

    def __init__(
        self,
        n_quantizers: int = 8,
        codebook_size: int = 1024,
        codebook_dim: int = 256,
        decay: float = 0.99,
        quantizer_dropout: float = 0.0,
    ):
        super().__init__()
        self.n_quantizers = n_quantizers
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        # Dropping trailing quantizers during training makes one model serve
        # several bitrates: decoding the first k codes stays valid for any k.
        self.quantizer_dropout = quantizer_dropout
        self.layers = nn.ModuleList([
            VectorQuantizer(codebook_size, codebook_dim, decay)
            for _ in range(n_quantizers)
        ])

    def forward(self, z: torch.Tensor, n_quantizers: int = None) -> tuple:
        """Returns ``(quantized, codes, commit_loss)`` with codes ``(B, n_q, T)``."""
        active = n_quantizers or self.n_quantizers
        if self.training and self.quantizer_dropout > 0 and n_quantizers is None:
            if torch.rand(()) < self.quantizer_dropout:
                active = int(torch.randint(1, self.n_quantizers + 1, ()))

        residual = z
        quantized = torch.zeros_like(z)
        codes, losses = [], []
        for layer in self.layers[:active]:
            stage, index, loss = layer(residual)
            residual = residual - stage.detach()
            quantized = quantized + stage
            codes.append(index)
            losses.append(loss)

        commit_loss = torch.stack(losses).mean() if losses else z.new_zeros(())
        return quantized, torch.stack(codes, dim=1), commit_loss

    @torch.no_grad()
    def encode(self, z: torch.Tensor, n_quantizers: int = None) -> torch.Tensor:
        """Acoustic token ids, ``(B, n_q, T)``."""
        return self.forward(z, n_quantizers)[1]

    def decode(self, codes: torch.Tensor) -> torch.Tensor:
        """Reconstruct the continuous latent from token ids, ``(B, D, T)``."""
        quantized = None
        for i in range(codes.shape[1]):
            stage = self.layers[i].codebook[codes[:, i]].transpose(1, 2)
            quantized = stage if quantized is None else quantized + stage
        return quantized

"""Sparse Mixture-of-Experts feed-forward for FramerAI.

This is the mechanism that lets *total* parameters scale to hundreds of billions
or a trillion while the *active* (per-token) compute stays small: each token is
routed to only ``n_experts_per_tok`` of the ``n_experts`` experts. Optional
always-on shared experts capture computation common to every token.

The dispatch here is correctness-first: experts are iterated and each processes
only the tokens routed to it (a gather / index_add scatter). This is exact and
memory-light; the throughput-optimized grouped-GEMM path is a drop-in future
optimization behind the same interface.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .transformer import FeedForward


class MoEFeedForward(nn.Module):
    """Top-k routed mixture of SwiGLU experts with a load-balancing aux loss."""

    is_moe = True

    def __init__(
        self,
        d_model: int,
        expert_d_ff: int,
        n_experts: int,
        n_experts_per_tok: int = 2,
        n_shared_experts: int = 0,
        dropout: float = 0.1,
        aux_loss_coef: float = 0.01,
        router_z_loss_coef: float = 0.001,
    ):
        super().__init__()
        assert n_experts_per_tok <= n_experts
        self.n_experts = n_experts
        self.top_k = n_experts_per_tok
        self.aux_loss_coef = aux_loss_coef
        self.router_z_loss_coef = router_z_loss_coef

        self.router = nn.Linear(d_model, n_experts, bias=False)
        self.experts = nn.ModuleList(
            [FeedForward(d_model, expert_d_ff, dropout) for _ in range(n_experts)]
        )
        self.shared_experts = nn.ModuleList(
            [FeedForward(d_model, expert_d_ff, dropout) for _ in range(n_shared_experts)]
        )

    def forward(self, x: torch.Tensor):
        B, T, D = x.shape
        x_flat = x.reshape(-1, D)  # (N, D)
        N = x_flat.shape[0]

        router_logits = self.router(x_flat)  # (N, E)
        router_probs = F.softmax(router_logits, dim=-1, dtype=torch.float32)

        topk_probs, topk_idx = router_probs.topk(self.top_k, dim=-1)  # (N, k)
        topk_gates = (topk_probs / (topk_probs.sum(-1, keepdim=True) + 1e-9)).to(x.dtype)

        out = torch.zeros_like(x_flat)
        for e, expert in enumerate(self.experts):
            hit = (topk_idx == e)  # (N, k)
            if not hit.any():
                continue
            token_idx, slot = hit.nonzero(as_tuple=True)
            gates = topk_gates[token_idx, slot].unsqueeze(-1)  # (m, 1)
            expert_out = expert(x_flat[token_idx])  # (m, D)
            out.index_add_(0, token_idx, gates * expert_out)

        for shared in self.shared_experts:
            out = out + shared(x_flat)

        aux = self._aux_loss(router_probs, topk_idx, router_logits, N)
        return out.view(B, T, D), aux

    def _aux_loss(self, router_probs, topk_idx, router_logits, N):
        """Switch/GShard load-balancing loss + router z-loss."""
        # Fraction of (token, slot) assignments landing on each expert.
        assign = torch.zeros(self.n_experts, device=router_probs.device, dtype=torch.float32)
        counts = torch.bincount(topk_idx.reshape(-1), minlength=self.n_experts).float()
        assign = counts / (N * self.top_k)
        # Mean routing probability mass per expert (differentiable).
        prob_mass = router_probs.mean(dim=0)  # (E,)
        balance = self.n_experts * torch.sum(assign * prob_mass)
        z_loss = torch.mean(torch.logsumexp(router_logits, dim=-1) ** 2)
        return self.aux_loss_coef * balance + self.router_z_loss_coef * z_loss


def build_ffn(config, layer_idx: int, dropout: float):
    """Factory: MoE FFN for MoE layers, dense SwiGLU otherwise."""
    if config.is_moe_layer(layer_idx):
        expert_d_ff = config.expert_d_ff or config.d_ff
        return MoEFeedForward(
            d_model=config.d_model,
            expert_d_ff=expert_d_ff,
            n_experts=config.n_experts,
            n_experts_per_tok=config.n_experts_per_tok,
            n_shared_experts=config.n_shared_experts,
            dropout=dropout,
            aux_loss_coef=config.aux_loss_coef,
            router_z_loss_coef=config.router_z_loss_coef,
        )
    return FeedForward(config.d_model, config.d_ff, dropout)

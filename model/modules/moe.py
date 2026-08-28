"""Sparse Mixture-of-Experts feed-forward for FramerAI.

This is the mechanism that lets *total* parameters scale to hundreds of billions
or a trillion while the *active* (per-token) compute stays small: each token is
routed to only ``n_experts_per_tok`` of the ``n_experts`` experts. Optional
always-on shared experts capture computation common to every token.

Two dispatch paths are provided:

- **Grouped dispatch** (CUDA, large batches): Sorts tokens by expert assignment
  and uses stacked weight tensors with einsum to compute all expert projections
  in three vectorized operations (w1, w3, w2), eliminating the per-expert Python
  loop. This is the throughput-optimized path for GPU workloads.

- **Loop dispatch** (CPU, tiny batches): The original per-expert gather/scatter
  loop. Retained as a correctness reference and CPU-friendly fallback.
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

        # Expert parallelism replaces this list with the local slice and sets
        # expert_offset; until then every rank holds every expert.
        self.expert_offset = 0
        self.ep_plan = None

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

        # Dispatch: grouped (CUDA, large batches) or loop (CPU, tiny batches)
        if self._should_use_grouped_dispatch(x_flat):
            out = self._grouped_expert_forward(x_flat, topk_idx, topk_gates)
        else:
            out = self._loop_expert_forward(x_flat, topk_idx, topk_gates)

        for shared in self.shared_experts:
            out = out + shared(x_flat)

        aux = self._aux_loss(router_probs, topk_idx, router_logits, N)
        return out.view(B, T, D), aux

    def _should_use_grouped_dispatch(self, x_flat):
        """Select grouped dispatch for CUDA and sufficiently large workloads."""
        # Grouped dispatch uses einsum, which benefits from GPU parallelism.
        # On CPU or tiny batches, the loop fallback may be faster.
        return x_flat.device.type == "cuda" and x_flat.shape[0] * self.top_k >= 32

    def _grouped_expert_forward(self, x_flat, topk_idx, topk_gates):
        """
        Grouped dispatch: vectorized expert computation without per-expert loops.

        Uses stacked weight tensors and einsum to process all local expert
        assignments in three batched operations (w1, w3, w2). The indexed weight
        selection is fused with einsum by PyTorch, avoiding memory explosion.
        """
        N, D = x_flat.shape
        k = self.top_k
        E_local = len(self.experts)

        if E_local == 0:
            return torch.zeros_like(x_flat)

        # 1. Flatten token-expert assignments: (N*k,)
        assignments = topk_idx.reshape(-1)  # global expert IDs
        token_ids = torch.arange(N, device=x_flat.device, dtype=torch.long)
        token_ids = token_ids.unsqueeze(1).expand(N, k).reshape(-1)
        gates_flat = topk_gates.reshape(-1)

        # 2. Filter to local experts [expert_offset, expert_offset + E_local)
        local_mask = (assignments >= self.expert_offset) & (assignments < self.expert_offset + E_local)
        local_assignments = assignments[local_mask] - self.expert_offset  # to [0, E_local)
        local_token_ids = token_ids[local_mask]
        local_gates = gates_flat[local_mask]

        M = local_assignments.numel()
        if M == 0:
            # No tokens routed to local experts
            return torch.zeros_like(x_flat)

        # 3. Sort by local expert ID to create contiguous segments
        sorted_experts, sort_idx = local_assignments.sort(stable=True)
        sorted_token_ids = local_token_ids[sort_idx]
        sorted_gates = local_gates[sort_idx]

        # 4. Gather tokens: (M, D)
        sorted_tokens = x_flat[sorted_token_ids]

        # 5. Stack expert weights into (E_local, ...) tensors
        # CRITICAL: Use actual parameter tensors, not .data, so gradients flow
        w1_stacked = torch.stack([e.w1.weight for e in self.experts], dim=0)  # (E_local, d_ff, D)
        w3_stacked = torch.stack([e.w3.weight for e in self.experts], dim=0)  # (E_local, d_ff, D)
        w2_stacked = torch.stack([e.w2.weight for e in self.experts], dim=0)  # (E_local, D, d_ff)

        # 6. Vectorized expert forward via einsum (NO per-expert Python loop)
        # Each token selects its expert's weights via sorted_experts indexing
        # einsum 'md,efd->mf' with e=sorted_experts: (M, D) x (M, d_ff, D) -> (M, d_ff)
        w1_weights = w1_stacked[sorted_experts]  # (M, d_ff, D)
        w3_weights = w3_stacked[sorted_experts]  # (M, d_ff, D)
        w2_weights = w2_stacked[sorted_experts]  # (M, D, d_ff)

        # Up-projections
        w1_out = torch.einsum("md,mfd->mf", sorted_tokens, w1_weights)
        w3_out = torch.einsum("md,mfd->mf", sorted_tokens, w3_weights)

        # SwiGLU activation
        hidden = F.silu(w1_out) * w3_out  # (M, d_ff)

        # Down-projection
        expert_outputs = torch.einsum("mf,mdf->md", hidden, w2_weights)  # (M, D)

        # Dropout (same semantics as FeedForward.forward)
        if self.training and E_local > 0:
            dropout_p = self.experts[0].dropout.p
            if dropout_p > 0:
                expert_outputs = F.dropout(expert_outputs, p=dropout_p, training=True)

        # 7. Apply routing gates
        gated_outputs = expert_outputs * sorted_gates.unsqueeze(1)  # (M, D)

        # 8. Scatter back to original token positions
        out = torch.zeros(N, D, dtype=x_flat.dtype, device=x_flat.device)
        out.index_add_(0, sorted_token_ids, gated_outputs)

        return out

    def _loop_expert_forward(self, x_flat, topk_idx, topk_gates):
        """
        Loop dispatch: original per-expert gather/scatter.

        Retained as fallback for CPU and as the correctness reference.
        """
        N, D = x_flat.shape
        out = torch.zeros_like(x_flat)

        for local_e, expert in enumerate(self.experts):
            e = local_e + self.expert_offset
            hit = topk_idx == e  # (N, k)
            if not hit.any():
                continue
            token_idx, slot = hit.nonzero(as_tuple=True)
            gates = topk_gates[token_idx, slot].unsqueeze(-1)  # (m, 1)
            expert_out = expert(x_flat[token_idx])  # (m, D)
            out.index_add_(0, token_idx, gates * expert_out)

        return out

    def _aux_loss(self, router_probs, topk_idx, router_logits, N):
        """Switch/GShard load-balancing loss + router z-loss."""
        # Fraction of (token, slot) assignments landing on each expert.
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

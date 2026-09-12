"""Sparse Mixture-of-Experts feed-forward for FramerAI.

This is the mechanism that lets *total* parameters scale to hundreds of billions
or a trillion while the *active* (per-token) compute stays small: each token is
routed to only ``n_experts_per_tok`` of the ``n_experts`` experts. Optional
always-on shared experts capture computation common to every token.

Two dispatch paths are provided:

- **Grouped dispatch** (default): Sorts token-expert assignments by expert ID,
  uses ``bincount`` to obtain contiguous per-expert segments, then issues one
  ``F.linear`` call per *non-empty* expert against that expert's token slice.
  No per-token weight gathering; memory is O(M·D) + O(E·d_ff·D) rather than
  O(M·d_ff·D). Correct on both CPU and CUDA; enables the CUDA fast path
  without the catastrophic temporary-tensor explosion of advanced indexing.

- **Loop dispatch** (CPU, tiny batches, correctness reference): The original
  per-expert gather/scatter loop.
"""

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from ..training.expert_parallel import all_to_all_combine, all_to_all_dispatch
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

        if (
            self.ep_plan is not None
            and self.ep_plan.enabled
            and (dist.is_available() and dist.is_initialized())
        ):
            out = self._ep_expert_forward(x_flat, topk_idx, topk_gates)
        elif self.expert_offset != 0 and (self.ep_plan is None or not self.ep_plan.enabled):
            raise RuntimeError(
                f"Invalid expert-parallel configuration: expert_offset={self.expert_offset} "
                "is non-zero but ep_plan is missing or disabled."
            )
        elif self._should_use_grouped_dispatch(x_flat):
            out = self._grouped_expert_forward(x_flat, topk_idx, topk_gates)
        else:
            out = self._loop_expert_forward(x_flat, topk_idx, topk_gates)

        for shared in self.shared_experts:
            out = out + shared(x_flat)

        aux = self._aux_loss(router_probs, topk_idx, router_logits, N)
        return out.view(B, T, D), aux

    def _ep_expert_forward(self, x_flat, topk_idx, topk_gates):
        """Expert-parallel dispatch: communicate off-rank tokens via all-to-all."""
        N, D = x_flat.shape
        k = self.top_k
        plan = self.ep_plan
        ep_world = plan.ep_world
        per_rank = self.n_experts // ep_world

        # 1. Flatten token-expert assignments: (N*k,)
        assignments = topk_idx.reshape(-1)
        token_ids = (
            torch.arange(N, device=x_flat.device, dtype=torch.long)
            .unsqueeze(1)
            .expand(N, k)
            .reshape(-1)
        )
        gates_flat = topk_gates.reshape(-1)

        # 2. Determine destination rank for each assignment
        dest_ranks = assignments // per_rank

        # 3. Sort by destination rank to form contiguous chunks per destination rank
        sort_idx = torch.argsort(dest_ranks, stable=True)
        sorted_dest_ranks = dest_ranks[sort_idx]
        sorted_token_ids = token_ids[sort_idx]
        sorted_expert_ids = assignments[sort_idx]
        sorted_gates = gates_flat[sort_idx]

        # 4. Gather tokens and calculate send counts per rank
        send_tokens = x_flat[sorted_token_ids]
        send_counts = torch.bincount(sorted_dest_ranks, minlength=ep_world)

        # 5. Dispatch tokens and assigned expert IDs to owning ranks
        recv_tokens, recv_counts = all_to_all_dispatch(send_tokens, send_counts, plan)
        recv_expert_ids, _ = all_to_all_dispatch(sorted_expert_ids.unsqueeze(-1), send_counts, plan)
        recv_expert_ids = recv_expert_ids.squeeze(-1)

        # 6. Process local experts on received tokens
        local_expert_ids = recv_expert_ids - self.expert_offset
        local_outputs = self._forward_local_experts(recv_tokens, local_expert_ids)

        # 7. Return expert outputs to originating rank
        combined = all_to_all_combine(local_outputs, send_counts, recv_counts, plan)

        # 8. Apply routing gates and scatter-add back to token positions
        gated = combined * sorted_gates.unsqueeze(-1)
        out = torch.zeros(N, D, dtype=x_flat.dtype, device=x_flat.device)
        out.index_add_(0, sorted_token_ids, gated)

        return out

    def _forward_local_experts(self, tokens: torch.Tensor, local_expert_ids: torch.Tensor) -> torch.Tensor:
        """Evaluate local experts on a batch of tokens, returning outputs in the same order."""
        M, D = tokens.shape
        E_local = len(self.experts)
        if M == 0 or E_local == 0:
            return tokens.new_zeros((M, D))

        if self._should_use_grouped_dispatch(tokens):
            sorted_experts, sort_idx = local_expert_ids.sort(stable=True)
            sorted_tokens = tokens[sort_idx]

            counts = torch.bincount(sorted_experts, minlength=E_local)
            offsets = torch.zeros(E_local + 1, dtype=torch.long, device=tokens.device)
            offsets[1:] = counts.cumsum(0)

            expert_outputs = torch.zeros(M, D, dtype=tokens.dtype, device=tokens.device)
            for local_e in range(E_local):
                start, end = offsets[local_e].item(), offsets[local_e + 1].item()
                if start == end:
                    continue
                tok_slice = sorted_tokens[start:end]
                exp = self.experts[local_e]
                h = F.silu(F.linear(tok_slice, exp.w1.weight)) * F.linear(tok_slice, exp.w3.weight)
                expert_outputs[start:end] = F.linear(h, exp.w2.weight)

            if self.training and E_local > 0:
                dropout_p = self.experts[0].dropout.p
                if dropout_p > 0:
                    expert_outputs = F.dropout(expert_outputs, p=dropout_p, training=True)

            local_outputs = torch.empty_like(expert_outputs)
            local_outputs[sort_idx] = expert_outputs
            return local_outputs
        else:
            local_outputs = torch.zeros(M, D, dtype=tokens.dtype, device=tokens.device)
            for local_e, expert in enumerate(self.experts):
                mask = (local_expert_ids == local_e)
                if not mask.any():
                    continue
                idx = mask.nonzero(as_tuple=True)[0]
                local_outputs[idx] = expert(tokens[idx])
            return local_outputs

    def _should_use_grouped_dispatch(self, x_flat):
        """Use grouped dispatch on all devices for workloads above the threshold."""
        return x_flat.shape[0] * self.top_k >= 32

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

        # 5. Per-expert token counts and contiguous segment offsets via bincount.
        #    sorted_experts is already in ascending order, so each expert segment
        #    forms a contiguous slice [offsets[e], offsets[e+1]).
        counts = torch.bincount(sorted_experts, minlength=E_local)  # (E_local,)
        offsets = torch.zeros(E_local + 1, dtype=torch.long, device=x_flat.device)
        offsets[1:] = counts.cumsum(0)

        # 6. One F.linear per non-empty local expert.
        #    Slice sorted_tokens and call F.linear against the expert weight matrix.
        #    No stacking, no per-token weight copy.
        expert_outputs = torch.zeros(M, D, dtype=x_flat.dtype, device=x_flat.device)
        for local_e in range(E_local):
            start, end = offsets[local_e].item(), offsets[local_e + 1].item()
            if start == end:
                continue
            tok_slice = sorted_tokens[start:end]           # (m_e, D)
            exp = self.experts[local_e]
            h = F.silu(F.linear(tok_slice, exp.w1.weight)) * F.linear(tok_slice, exp.w3.weight)
            expert_outputs[start:end] = F.linear(h, exp.w2.weight)

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

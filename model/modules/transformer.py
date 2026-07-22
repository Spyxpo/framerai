"""Core transformer building blocks for FramerAI.

Modernized backbone: grouped-query attention (GQA), fused scaled-dot-product
attention (flash / mem-efficient kernels where available), an incremental
KV cache for O(n) decoding, RoPE with optional context-extension scaling, and
optional QK-normalization for stability at scale.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) with optional context-extension scaling.

    Supports ``linear`` position interpolation and ``ntk`` base scaling so the
    same weights can serve a longer context than they were trained on. cos/sin
    are computed for absolute positions, which makes incremental decoding with a
    KV cache correct (only the new positions are rotated per step).
    """

    def __init__(
        self,
        head_dim: int,
        max_seq_len: int = 2048,
        base: float = 10000.0,
        scaling_factor: float = 1.0,
        scaling_type: str = "linear",
    ):
        super().__init__()
        self.head_dim = head_dim
        self.max_seq_len = max_seq_len
        self.scaling_factor = max(1.0, float(scaling_factor))
        self.scaling_type = scaling_type

        if scaling_type == "ntk" and self.scaling_factor > 1.0:
            # NTK-aware: raise the base so high frequencies are stretched less.
            base = base * (self.scaling_factor ** (head_dim / (head_dim - 2)))

        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, seq_len: int, offset: int = 0, device=None) -> tuple:
        device = device or self.inv_freq.device
        positions = torch.arange(offset, offset + seq_len, device=device).float()
        if self.scaling_type == "linear" and self.scaling_factor > 1.0:
            positions = positions / self.scaling_factor
        freqs = torch.einsum("i,j->ij", positions, self.inv_freq.to(device))
        emb = torch.cat([freqs, freqs], dim=-1)
        return emb.cos(), emb.sin()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([-x2, x1], dim=-1)


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # x: (B, n_heads, T, head_dim); cos/sin: (T, head_dim)
    cos = cos.unsqueeze(0).unsqueeze(0)
    sin = sin.unsqueeze(0).unsqueeze(0)
    return (x * cos) + (rotate_half(x) * sin)


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand GQA key/value heads to the full query-head count."""
    if n_rep == 1:
        return x
    B, n_kv, T, hd = x.shape
    return (
        x[:, :, None, :, :]
        .expand(B, n_kv, n_rep, T, hd)
        .reshape(B, n_kv * n_rep, T, hd)
    )


class CausalSelfAttention(nn.Module):
    """Grouped-query causal self-attention with RoPE and a KV cache."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        n_kv_heads: int = None,
        use_qk_norm: bool = False,
        rope_theta: float = 10000.0,
        rope_scaling_factor: float = 1.0,
        rope_scaling_type: str = "linear",
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.n_kv_heads = n_kv_heads or n_heads
        assert n_heads % self.n_kv_heads == 0, "n_heads must be divisible by n_kv_heads"
        self.n_rep = n_heads // self.n_kv_heads
        self.head_dim = d_model // n_heads
        self.dropout_p = dropout

        self.q_proj = nn.Linear(d_model, n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(d_model, self.n_kv_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(n_heads * self.head_dim, d_model, bias=False)
        self.resid_dropout = nn.Dropout(dropout)

        self.use_qk_norm = use_qk_norm
        if use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim)
            self.k_norm = RMSNorm(self.head_dim)

        self.rope = RotaryPositionalEmbedding(
            self.head_dim, max_seq_len, base=rope_theta,
            scaling_factor=rope_scaling_factor, scaling_type=rope_scaling_type,
        )

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        past_kv: tuple = None,
        use_cache: bool = False,
    ):
        B, T, C = x.shape
        has_past = past_kv is not None
        offset = past_kv[0].shape[2] if has_past else 0

        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_kv_heads, self.head_dim).transpose(1, 2)

        if self.use_qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        # RoPE at absolute positions [offset, offset + T). Past keys are already
        # rotated, so only the new tokens are rotated here.
        cos, sin = self.rope(T, offset=offset, device=x.device)
        cos = cos.to(q.dtype)
        sin = sin.to(q.dtype)
        q = apply_rotary_emb(q, cos, sin)
        k = apply_rotary_emb(k, cos, sin)

        # Append to the cache.
        if past_kv is not None:
            k = torch.cat([past_kv[0], k], dim=2)
            v = torch.cat([past_kv[1], v], dim=2)
        present = (k, v) if use_cache else None

        # Expand GQA heads for the attention matmul.
        k = repeat_kv(k, self.n_rep)
        v = repeat_kv(v, self.n_rep)

        dropout_p = self.dropout_p if self.training else 0.0

        if attention_mask is None and not has_past:
            # Fast path: fused causal SDPA (flash / mem-efficient kernels).
            # Valid only when Lq == Lk (uncached prefill / training); PyTorch's
            # is_causal uses top-left alignment, which is wrong for a cached
            # query offset, so the cached path builds an explicit mask below.
            out = F.scaled_dot_product_attention(
                q, k, v, is_causal=True, dropout_p=dropout_p
            )
        else:
            attn_mask = self._build_attn_mask(attention_mask, T, k.shape[2], offset, q.dtype, x.device)
            out = F.scaled_dot_product_attention(
                q, k, v, attn_mask=attn_mask, dropout_p=dropout_p
            )

        out = out.transpose(1, 2).contiguous().view(B, T, self.n_heads * self.head_dim)
        out = self.resid_dropout(self.out_proj(out))
        return (out, present) if use_cache else out

    def _build_attn_mask(self, attention_mask, q_len, k_len, offset, dtype, device):
        """Offset-aware causal float mask, optionally combined with key-padding.

        A query at absolute position ``offset + i`` may attend to key position
        ``j`` iff ``j <= offset + i``. This is correct for uncached prefill,
        chunked prefill, and single-token decode against a KV cache.
        ``attention_mask`` (optional) is a (B, k_len) 1/0 keep/pad mask.
        """
        neg = torch.finfo(dtype).min
        q_pos = torch.arange(offset, offset + q_len, device=device).unsqueeze(1)  # (q,1)
        k_pos = torch.arange(k_len, device=device).unsqueeze(0)  # (1,k)
        causal = (k_pos > q_pos)  # True where disallowed
        mask = torch.zeros(q_len, k_len, dtype=dtype, device=device)
        mask = mask.masked_fill(causal, neg).unsqueeze(0).unsqueeze(0)  # (1,1,q,k)
        if attention_mask is not None:
            pad = (attention_mask == 0).view(attention_mask.shape[0], 1, 1, k_len)
            mask = mask.masked_fill(pad, neg)
        return mask


class FeedForward(nn.Module):
    """SwiGLU feed-forward network."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(d_model, d_ff, bias=False)
        self.w2 = nn.Linear(d_ff, d_model, bias=False)
        self.w3 = nn.Linear(d_model, d_ff, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with RMSNorm and optional MoE FFN."""

    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        max_seq_len: int = 2048,
        dropout: float = 0.1,
        n_kv_heads: int = None,
        use_qk_norm: bool = False,
        rope_theta: float = 10000.0,
        rope_scaling_factor: float = 1.0,
        rope_scaling_type: str = "linear",
        ffn: nn.Module = None,
    ):
        super().__init__()
        self.norm1 = RMSNorm(d_model)
        self.attn = CausalSelfAttention(
            d_model, n_heads, max_seq_len, dropout,
            n_kv_heads=n_kv_heads, use_qk_norm=use_qk_norm,
            rope_theta=rope_theta, rope_scaling_factor=rope_scaling_factor,
            rope_scaling_type=rope_scaling_type,
        )
        self.norm2 = RMSNorm(d_model)
        # ffn may be a dense FeedForward or a MoEFeedForward (returns aux loss).
        self.ffn = ffn if ffn is not None else FeedForward(d_model, d_ff, dropout)
        self.is_moe = hasattr(self.ffn, "is_moe") and self.ffn.is_moe

    def forward(
        self,
        x: torch.Tensor,
        attention_mask: torch.Tensor = None,
        past_kv: tuple = None,
        use_cache: bool = False,
    ):
        attn_out = self.attn(self.norm1(x), attention_mask, past_kv=past_kv, use_cache=use_cache)
        if use_cache:
            attn_out, present = attn_out
        else:
            present = None
        x = x + attn_out

        if self.is_moe:
            ffn_out, aux = self.ffn(self.norm2(x))
            x = x + ffn_out
        else:
            x = x + self.ffn(self.norm2(x))
            aux = None

        return {"x": x, "present": present, "aux": aux}

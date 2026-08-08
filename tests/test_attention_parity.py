"""Parity tests for the fused-SDPA attention rewrites.

The vision, temporal, and diffusion attention modules were converted from an
explicit `matmul -> softmax -> matmul` to `F.scaled_dot_product_attention`. The
diffusion modules keep a channel-first `(B, heads, head_dim, seq)` layout from
their Conv1d projections, so the conversion needs a transpose on the way in and
out. These tests pin that layout handling against the reference computation.
"""

import math

import pytest
import torch
import torch.nn.functional as F

from model.modules.diffusion import CrossAttention, SpatialAttention
from model.modules.video_generator import TemporalAttention
from model.modules.vision_encoder import VisionAttention


def reference_attention(q, k, v):
    """The explicit form every module used before the SDPA conversion."""
    scale = math.sqrt(q.shape[-1])
    weights = F.softmax(torch.matmul(q, k.transpose(-2, -1)) / scale, dim=-1)
    return torch.matmul(weights, v)


def test_token_layout_matches_reference():
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 4, 9, 16) for _ in range(3))
    assert torch.allclose(
        reference_attention(q, k, v),
        F.scaled_dot_product_attention(q, k, v),
        atol=1e-5,
    )


def test_channel_first_layout_matches_reference():
    """The diffusion modules carry (B, heads, head_dim, seq) from their Conv1d."""
    torch.manual_seed(0)
    q, k, v = (torch.randn(2, 4, 16, 9) for _ in range(3))

    reference = torch.einsum(
        "bhij,bhdj->bhdi",
        F.softmax(torch.einsum("bhdi,bhdj->bhij", q, k) / math.sqrt(16), dim=-1),
        v,
    )
    fused = F.scaled_dot_product_attention(
        q.transpose(-2, -1), k.transpose(-2, -1), v.transpose(-2, -1)
    ).transpose(-2, -1)

    assert torch.allclose(reference, fused, atol=1e-5)


@pytest.mark.parametrize("resolution", [8, 16, 32])
def test_spatial_attention_is_residual_and_shape_preserving(resolution):
    module = SpatialAttention(64, 8).eval()
    x = torch.randn(1, 64, resolution, resolution)
    with torch.no_grad():
        out = module(x)
    assert out.shape == x.shape
    assert torch.isfinite(out).all()


def test_cross_attention_conditions_on_context():
    module = CrossAttention(64, context_dim=32, n_heads=8).eval()
    x = torch.randn(1, 64, 8, 8)
    with torch.no_grad():
        a = module(x, torch.randn(1, 5, 32))
        b = module(x, torch.randn(1, 5, 32))
    assert a.shape == x.shape
    assert not torch.allclose(a, b), "output ignored the conditioning context"


def test_vision_attention_shape_and_determinism_in_eval():
    module = VisionAttention(64, 8, dropout=0.5).eval()
    x = torch.randn(2, 17, 64)
    with torch.no_grad():
        first, second = module(x), module(x)
    assert first.shape == x.shape
    # dropout_p must be gated on training, or eval-mode inference is stochastic.
    assert torch.allclose(first, second)


def test_temporal_attention_shape_and_determinism_in_eval():
    module = TemporalAttention(64, 8, dropout=0.5).eval()
    x = torch.randn(1, 4, 9, 64)
    with torch.no_grad():
        first, second = module(x), module(x)
    assert first.shape == x.shape
    assert torch.allclose(first, second)


def test_vision_encoder_handles_a_realistic_patch_count():
    """1024 patches per tile is the shape the large presets configure."""
    from model.modules.vision_encoder import VisionEncoder

    encoder = VisionEncoder(448, 14, 64, 4, 1, 0.0).eval()
    with torch.no_grad():
        out = encoder(torch.randn(1, 3, 448, 448))
    assert out.shape == (1, 1025, 64)
    assert torch.isfinite(out).all()

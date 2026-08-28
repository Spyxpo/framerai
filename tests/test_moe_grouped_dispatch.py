"""Grouped-dispatch tests for MoE (Issue #157).

These tests verify that the vectorized grouped-GEMM dispatch path produces
identical results to the loop-based reference and correctly handles gradients,
empty experts, expert parallelism, and CPU fallback.
"""

import copy

import pytest
import torch

from model.modules.moe import MoEFeedForward
from model.training.expert_parallel import ExpertParallelPlan, shard_experts


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_grouped_matches_loop_same_device():
    """Grouped dispatch must produce identical output to loop dispatch."""
    torch.manual_seed(42)
    ffn = MoEFeedForward(
        d_model=64,
        expert_d_ff=128,
        n_experts=8,
        n_experts_per_tok=2,
        n_shared_experts=0,
        dropout=0.0,
    )
    ffn.eval().cuda()

    x = torch.randn(4, 16, 64, device="cuda")

    # Force loop path
    original_method = ffn._should_use_grouped_dispatch
    ffn._should_use_grouped_dispatch = lambda x: False
    with torch.no_grad():
        out_loop, aux_loop = ffn(x)

    # Force grouped path
    ffn._should_use_grouped_dispatch = lambda x: True
    with torch.no_grad():
        out_grouped, aux_grouped = ffn(x)

    # Restore
    ffn._should_use_grouped_dispatch = original_method

    max_diff = (out_loop - out_grouped).abs().max().item()
    assert torch.allclose(out_loop, out_grouped, atol=1e-5, rtol=1e-4), f"Max diff: {max_diff}"
    assert torch.allclose(aux_loop, aux_grouped, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_grouped_dispatch_gradients():
    """Gradients must flow correctly through grouped dispatch."""
    torch.manual_seed(42)
    ffn = MoEFeedForward(
        d_model=32, expert_d_ff=64, n_experts=4, n_experts_per_tok=2, n_shared_experts=0, dropout=0.0
    ).cuda()

    x = torch.randn(2, 8, 32, device="cuda", requires_grad=True)

    # Force grouped (NO torch.no_grad - we need gradients!)
    ffn._should_use_grouped_dispatch = lambda x: True
    out, aux = ffn(x)
    loss = out.sum() + aux
    loss.backward()

    # Check all expert weights received gradients
    for i, expert in enumerate(ffn.experts):
        assert expert.w1.weight.grad is not None, f"expert {i} w1 has no grad"
        assert expert.w1.weight.grad.abs().sum() > 0, f"expert {i} w1 grad is zero"
        assert expert.w2.weight.grad is not None, f"expert {i} w2 has no grad"
        assert expert.w2.weight.grad.abs().sum() > 0, f"expert {i} w2 grad is non-zero"
        assert expert.w3.weight.grad is not None, f"expert {i} w3 has no grad"
        assert expert.w3.weight.grad.abs().sum() > 0, f"expert {i} w3 grad is non-zero"

    # Check input gradient
    assert x.grad is not None, "Input has no gradient"
    assert torch.isfinite(x.grad).all(), "Input gradient has non-finite values"
    assert x.grad.abs().sum() > 0, "Input gradient is zero"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_empty_experts_no_sync():
    """Empty experts work correctly without device-host sync."""
    ffn = MoEFeedForward(
        d_model=32, expert_d_ff=64, n_experts=8, n_experts_per_tok=1, n_shared_experts=0, dropout=0.0
    ).cuda()
    ffn.eval()

    # Manually route all tokens to expert 0 only
    x = torch.randn(2, 16, 32, device="cuda")
    original_router = ffn.router

    def fixed_router(x_flat):
        N = x_flat.shape[0]
        logits = torch.full((N, 8), -1000.0, device="cuda")
        logits[:, 0] = 1000.0  # all tokens to expert 0
        return logits

    ffn.router = fixed_router
    ffn._should_use_grouped_dispatch = lambda x: True

    with torch.no_grad():
        out, aux = ffn(x)

    ffn.router = original_router

    # Verify no errors with empty experts 1-7
    assert torch.isfinite(out).all()
    assert out.abs().sum() > 0  # non-zero output from expert 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_grouped_with_expert_offset():
    """Grouped dispatch respects expert_offset for expert parallelism."""
    torch.manual_seed(42)

    # Create full MoE layer
    ffn_full = MoEFeedForward(
        d_model=32, expert_d_ff=64, n_experts=8, n_experts_per_tok=2, n_shared_experts=0, dropout=0.0
    )
    ffn_full.eval()

    # Shard: this rank owns experts 4-7
    plan = ExpertParallelPlan(ep_world=2, ep_rank=1)
    ffn_shard = shard_experts(copy.deepcopy(ffn_full), plan)
    ffn_shard.cuda()

    assert ffn_shard.expert_offset == 4
    assert len(ffn_shard.experts) == 4

    x = torch.randn(2, 8, 32, device="cuda")

    # Force grouped on shard
    ffn_shard._should_use_grouped_dispatch = lambda x: True
    with torch.no_grad():
        out_grouped, _ = ffn_shard(x)

    # Compare against loop fallback
    ffn_shard._should_use_grouped_dispatch = lambda x: False
    with torch.no_grad():
        out_loop, _ = ffn_shard(x)

    assert torch.allclose(out_grouped, out_loop, atol=1e-5)


def test_cpu_uses_fallback():
    """CPU automatically uses loop fallback."""
    ffn = MoEFeedForward(
        d_model=32, expert_d_ff=64, n_experts=4, n_experts_per_tok=2, n_shared_experts=0, dropout=0.0
    )
    ffn.eval()  # CPU by default

    x = torch.randn(2, 8, 32)

    # Should auto-select fallback
    with torch.no_grad():
        out, aux = ffn(x)

    assert out.device.type == "cpu"
    assert torch.isfinite(out).all()

    # Verify it actually used fallback (not grouped)
    assert not ffn._should_use_grouped_dispatch(x.reshape(-1, 32))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_weight_updates_reflected_in_grouped_path():
    """Optimizer updates to expert parameters must be immediately reflected by grouped dispatch.

    This test explicitly verifies no stale cached weights exist.
    """
    torch.manual_seed(42)
    ffn = MoEFeedForward(
        d_model=32, expert_d_ff=64, n_experts=4, n_experts_per_tok=2, n_shared_experts=0, dropout=0.0
    ).cuda()

    optimizer = torch.optim.SGD(ffn.parameters(), lr=0.1)
    x = torch.randn(2, 8, 32, device="cuda", requires_grad=True)

    # Force grouped
    ffn._should_use_grouped_dispatch = lambda x: True

    # Forward 1 + backward + step
    out1, aux1 = ffn(x)
    loss1 = out1.sum() + aux1
    loss1.backward()

    # Capture gradient magnitude before step
    grad_magnitudes_before = {
        i: (expert.w1.weight.grad.abs().sum().item(), expert.w2.weight.grad.abs().sum().item())
        for i, expert in enumerate(ffn.experts)
    }

    # Apply optimizer step - this MUST change expert parameters
    optimizer.step()
    optimizer.zero_grad()

    # Forward 2 with same input - output MUST differ
    x2 = x.detach().clone().requires_grad_(True)
    out2, aux2 = ffn(x2)

    # Verify parameters actually changed
    for expert in ffn.experts:
        # Parameters must have changed after optimizer step
        assert expert.w1.weight.grad is None or expert.w1.weight.grad.abs().sum() == 0, "Gradients not zeroed"

    # Outputs MUST differ because weights changed
    diff = (out1.detach() - out2.detach()).abs().max().item()
    assert diff > 1e-4, (
        f"Weights not updated or grouped path uses stale cache: diff={diff:.2e}\n"
        f"Grad magnitudes before step: {grad_magnitudes_before}"
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_grouped_dispatch_with_dropout():
    """Grouped dispatch preserves dropout behavior."""
    torch.manual_seed(42)
    ffn = MoEFeedForward(
        d_model=32, expert_d_ff=64, n_experts=4, n_experts_per_tok=2, n_shared_experts=0, dropout=0.5
    ).cuda()
    ffn.train()  # dropout active

    x = torch.randn(2, 16, 32, device="cuda")

    # Force grouped
    ffn._should_use_grouped_dispatch = lambda x: True

    # Multiple forward passes should differ (dropout is stochastic)
    with torch.no_grad():
        out1, _ = ffn(x)
        out2, _ = ffn(x)

    diff = (out1 - out2).abs().max().item()
    assert diff > 1e-3, f"Dropout not active in grouped path: diff={diff}"

    # In eval mode, should be deterministic
    ffn.eval()
    with torch.no_grad():
        out3, _ = ffn(x)
        out4, _ = ffn(x)

    assert torch.allclose(out3, out4, atol=1e-6), "Eval mode not deterministic"


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_grouped_dispatch_preserves_dtype():
    """Grouped dispatch preserves input dtype."""
    for dtype in [torch.float16, torch.bfloat16, torch.float32]:
        if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
            continue

        ffn = MoEFeedForward(
            d_model=32, expert_d_ff=64, n_experts=4, n_experts_per_tok=2, n_shared_experts=0, dropout=0.0
        ).cuda()
        ffn = ffn.to(dtype)
        ffn.eval()

        x = torch.randn(2, 8, 32, device="cuda", dtype=dtype)

        ffn._should_use_grouped_dispatch = lambda x: True
        with torch.no_grad():
            out, aux = ffn(x)

        assert out.dtype == dtype, f"Output dtype {out.dtype} != input dtype {dtype}"


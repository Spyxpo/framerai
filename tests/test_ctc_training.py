"""Tests for CTC auxiliary head wiring into the training loop (Issue #158)."""

import torch

from conftest import tiny_config
from model.framer import FramerModel
from model.modules.audio_lm import CTCHead


def ctc_multimodal_config(**overrides):
    """Smallest config that builds audio encoder and CTC head."""
    base = dict(
        text_only=False,
        use_ctc_head=True,
        ctc_loss_weight=0.5,
        audio_sample_rate=16000,
        audio_n_fft=64,
        audio_hop_length=16,
        audio_n_mels=16,
        audio_max_frames=32,
        audio_d_model=32,
        audio_n_heads=4,
        audio_n_layers=1,
    )
    base.update(overrides)
    return tiny_config(**base)


def test_default_config_leaves_ctc_disabled():
    """Default FramerConfig must keep use_ctc_head=False and self.ctc_head=None."""
    config = tiny_config(text_only=False)
    assert not config.use_ctc_head
    model = FramerModel(config)
    assert model.ctc_head is None

    out = model(audio=torch.randn(1, 4000))
    assert "ctc_loss" not in out


def test_ctc_head_skipped_when_targets_missing():
    """When use_ctc_head=True but no ctc_targets exist, ctc_loss term is omitted."""
    config = ctc_multimodal_config(use_ctc_head=True)
    model = FramerModel(config)
    assert isinstance(model.ctc_head, CTCHead)

    out = model(audio=torch.randn(1, 4000))
    assert "ctc_loss" not in out


def test_ctc_head_computes_weighted_loss_and_flows_gradients():
    """When enabled with ctc_targets, ctc_loss is computed and gradients flow through CTC head and audio encoder."""
    torch.manual_seed(42)
    config = ctc_multimodal_config(use_ctc_head=True, ctc_loss_weight=0.5)
    model = FramerModel(config)

    audio = torch.randn(2, 4000)
    # CTC targets for batch size 2
    ctc_targets = torch.tensor([[1, 2, 3, 4], [5, 6, 7, -100]], dtype=torch.long)
    ctc_target_lengths = torch.tensor([4, 3], dtype=torch.long)

    out = model(
        audio=audio,
        ctc_targets=ctc_targets,
        ctc_target_lengths=ctc_target_lengths,
    )

    assert "ctc_loss" in out
    assert torch.isfinite(out["ctc_loss"])
    assert "loss" in out
    assert torch.isfinite(out["loss"])

    # Backward pass and check gradient flow
    out["loss"].backward()

    # CTC head projection weights must receive gradients
    assert model.ctc_head.proj.weight.grad is not None
    assert torch.abs(model.ctc_head.proj.weight.grad).sum() > 0

    # Audio encoder parameters must receive gradients
    has_audio_encoder_grad = any(
        p.grad is not None and torch.abs(p.grad).sum() > 0
        for p in model.audio_encoder.parameters()
    )
    assert has_audio_encoder_grad, "Gradients failed to flow back into AudioEncoder parameters"


def test_ctc_objective_optimization_decreases_loss():
    """Running multiple optimization steps with CTC loss decreases ctc_loss."""
    torch.manual_seed(42)
    config = ctc_multimodal_config(use_ctc_head=True, ctc_loss_weight=1.0)
    model = FramerModel(config)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-2)

    audio = torch.randn(2, 4000)
    ctc_targets = torch.tensor([[2, 4, 6], [3, 5, 7]], dtype=torch.long)

    initial_loss = None
    final_loss = None

    for step in range(15):
        optimizer.zero_grad()
        out = model(audio=audio, ctc_targets=ctc_targets)
        loss = out["ctc_loss"]
        if step == 0:
            initial_loss = loss.item()
        final_loss = loss.item()
        out["loss"].backward()
        optimizer.step()

    assert final_loss < initial_loss, f"Expected CTC loss to decrease, but initial={initial_loss}, final={final_loss}"

"""MoE tests: routing, aux loss, gradients reach experts, layer selection."""

import torch

from conftest import random_ids, tiny_moe_config
from model.framer import FramerModel
from model.modules.moe import MoEFeedForward


def test_moe_forward_returns_finite_aux():
    cfg = tiny_moe_config()
    model = FramerModel(cfg)
    ids = random_ids(cfg)
    out = model(input_ids=ids, labels=ids)
    assert "aux_loss" in out
    assert torch.isfinite(out["aux_loss"]) and out["aux_loss"] >= 0
    assert torch.isfinite(out["loss"])


def test_moe_shape_and_backward():
    ffn = MoEFeedForward(d_model=32, expert_d_ff=64, n_experts=4, n_experts_per_tok=2,
                         n_shared_experts=1, dropout=0.0)
    x = torch.randn(2, 8, 32, requires_grad=True)
    y, aux = ffn(x)
    assert y.shape == x.shape
    (y.sum() + aux).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_all_experts_receive_gradient_over_enough_tokens():
    torch.manual_seed(0)
    ffn = MoEFeedForward(d_model=32, expert_d_ff=64, n_experts=4, n_experts_per_tok=2,
                         n_shared_experts=0, dropout=0.0)
    x = torch.randn(4, 64, 32)  # many tokens -> every expert should be hit
    y, aux = ffn(x)
    (y.sum() + aux).backward()
    touched = [e.w1.weight.grad is not None and e.w1.weight.grad.abs().sum() > 0
               for e in ffn.experts]
    assert all(touched), "some experts never routed a token"


def test_layer_selection_respects_first_dense_and_freq():
    cfg = tiny_moe_config(n_layers=4, first_dense_layers=1, moe_layer_freq=2)
    model = FramerModel(cfg)
    flags = [model.layers[i].is_moe for i in range(cfg.n_layers)]
    # layer 0 dense (first_dense), then every 2nd layer is MoE from the offset.
    assert flags == [False, True, False, True]

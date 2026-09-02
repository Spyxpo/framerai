"""Tests for model parameter count reporting functionality."""

import torch.nn as nn

from conftest import tiny_config
from model.framer import FramerModel
from model.utils import count_parameters, get_parameter_counts


class SimpleDeterministicModel(nn.Module):
    """A simple deterministic PyTorch model with manually computable parameter counts.

    Architecture:
    - fc1: nn.Linear(in_features=10, out_features=5, bias=True)
      - Weight tensor: shape (5, 10) -> 50 params
      - Bias tensor: shape (5,) -> 5 params
      Total fc1 params: 55

    - fc2: nn.Linear(in_features=5, out_features=2, bias=False)
      - Weight tensor: shape (2, 5) -> 10 params
      - Bias: None
      Total fc2 params: 10

    - fc3: nn.Linear(in_features=2, out_features=1, bias=True)
      - Weight tensor: shape (1, 2) -> 2 params
      - Bias tensor: shape (1,) -> 1 param
      Total fc3 params: 3

    Total model parameters = 55 + 10 + 3 = 68.
    If fc1 is frozen (requires_grad = False):
      - Frozen params = 55
      - Trainable params = 10 + 3 = 13
    """

    def __init__(self, freeze_fc1: bool = False):
        super().__init__()
        self.fc1 = nn.Linear(10, 5, bias=True)
        self.fc2 = nn.Linear(5, 2, bias=False)
        self.fc3 = nn.Linear(2, 1, bias=True)

        if freeze_fc1:
            for p in self.fc1.parameters():
                p.requires_grad = False


def test_parameter_counts_all_trainable():
    model = SimpleDeterministicModel(freeze_fc1=False)

    # Expected values calculated manually:
    # fc1: 5 * 10 (weights) + 5 (bias) = 55
    # fc2: 2 * 5 (weights) = 10
    # fc3: 1 * 2 (weights) + 1 (bias) = 3
    # Total: 68
    # Trainable: 68
    expected_total = 68
    expected_trainable = 68

    counts = get_parameter_counts(model)
    assert counts["total"] == expected_total
    assert counts["trainable"] == expected_trainable
    assert count_parameters(model, trainable_only=True) == expected_trainable
    assert count_parameters(model, trainable_only=False) == expected_total


def test_parameter_counts_with_frozen_layers():
    model = SimpleDeterministicModel(freeze_fc1=True)

    # Expected values calculated manually:
    # fc1 (frozen, 55 params): requires_grad = False
    # fc2 (trainable, 10 params): requires_grad = True
    # fc3 (trainable, 3 params): requires_grad = True
    # Total: 68
    # Trainable: 13
    expected_total = 68
    expected_trainable = 13

    counts = get_parameter_counts(model)
    assert counts["total"] == expected_total
    assert counts["trainable"] == expected_trainable
    assert count_parameters(model, trainable_only=True) == expected_trainable
    assert count_parameters(model, trainable_only=False) == expected_total


def test_parameter_counts_on_framer_model():
    config = tiny_config()
    model = FramerModel(config)

    # By default, all FramerModel parameters require grad
    counts = get_parameter_counts(model)
    manual_total = sum(p.numel() for p in model.parameters())
    manual_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    assert counts["total"] == manual_total
    assert counts["trainable"] == manual_trainable
    assert counts["total"] == counts["trainable"]

    # Freeze a parameter and verify counts update correctly
    first_param = next(model.parameters())
    num_frozen = first_param.numel()
    first_param.requires_grad = False

    updated_counts = get_parameter_counts(model)
    assert updated_counts["total"] == manual_total
    assert updated_counts["trainable"] == manual_trainable - num_frozen

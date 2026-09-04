"""Tests for model parameter count reporting functionality."""

import json

import torch.nn as nn

from build import build_model, export_model, save_model_info
from conftest import tiny_config
from model.framer import FramerModel
from model.utils import (
    count_parameters,
    format_model_summary,
    get_component_parameter_counts,
    get_parameter_counts,
)


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


def test_model_info_json_output(tmp_path):
    config = tiny_config()
    model = SimpleDeterministicModel(freeze_fc1=False)

    # Test save_model_info with all trainable parameters
    save_model_info(model, config, str(tmp_path))
    info_file = tmp_path / "model_info.json"
    assert info_file.exists()

    with open(info_file) as f:
        data = json.load(f)

    assert data["model_name"] == "FramerAI"
    assert data["total_parameters"] == 68
    assert data["trainable_parameters"] == 68
    assert data["parameters"] == 68
    assert "config" in data
    assert "modalities" in data

    # Test save_model_info with frozen parameters
    model_frozen = SimpleDeterministicModel(freeze_fc1=True)
    save_model_info(model_frozen, config, str(tmp_path))

    with open(info_file) as f:
        data_frozen = json.load(f)

    assert data_frozen["parameters"] == 13
    assert data_frozen["total_parameters"] == 68
    assert data_frozen["trainable_parameters"] == 13
    assert data_frozen["total_parameters"] != data_frozen["trainable_parameters"]


def test_build_and_export_model_info(tmp_path):
    config = tiny_config()
    build_dir = str(tmp_path / "build_output")
    export_dir = str(tmp_path / "export_output")

    model, _ = build_model(config, output_dir=build_dir)
    build_info_path = tmp_path / "build_output" / "model_info.json"
    assert build_info_path.exists()

    with open(build_info_path) as f:
        build_data = json.load(f)

    total_params = sum(p.numel() for p in model.parameters())
    assert build_data["total_parameters"] == total_params
    assert build_data["trainable_parameters"] == total_params

    export_model(config, output_dir=build_dir, export_dir=export_dir)
    export_info_path = tmp_path / "export_output" / "model_info.json"
    assert export_info_path.exists()

    with open(export_info_path) as f:
        export_data = json.load(f)

    assert export_data["total_parameters"] == total_params
    assert export_data["trainable_parameters"] == total_params


def test_component_parameter_counts():
    model = SimpleDeterministicModel(freeze_fc1=False)
    components = get_component_parameter_counts(model)

    assert "fc1" in components
    assert "fc2" in components
    assert "fc3" in components
    assert components["fc1"] == {"total": 55, "trainable": 55}
    assert components["fc2"] == {"total": 10, "trainable": 10}
    assert components["fc3"] == {"total": 3, "trainable": 3}

    model_frozen = SimpleDeterministicModel(freeze_fc1=True)
    components_frozen = get_component_parameter_counts(model_frozen)

    assert components_frozen["fc1"] == {"total": 55, "trainable": 0}
    assert components_frozen["fc2"] == {"total": 10, "trainable": 10}
    assert components_frozen["fc3"] == {"total": 3, "trainable": 3}


def test_component_parameter_counts_framer_model():
    config = tiny_config()
    model = FramerModel(config)
    components = get_component_parameter_counts(model)

    assert "token_embed" in components
    assert "layers" in components
    assert "norm" in components
    assert "lm_head" in components

    total_sum_components = sum(c["total"] for c in components.values())
    assert total_sum_components >= count_parameters(model, trainable_only=False)


def test_format_model_summary():
    model = SimpleDeterministicModel(freeze_fc1=True)
    summary = format_model_summary(model, model_name="TestModel")

    assert "Model Summary: TestModel" in summary
    assert "Component Breakdown:" in summary
    assert "fc1" in summary
    assert "fc2" in summary
    assert "fc3" in summary
    assert "Total Parameters:     68" in summary
    assert "Trainable Parameters: 13" in summary

    summary_again = format_model_summary(model, model_name="TestModel")
    assert summary == summary_again


def test_build_model_logs_summary(caplog, tmp_path):
    import logging

    caplog.set_level(logging.INFO)
    config = tiny_config()
    build_dir = str(tmp_path / "build_output")

    build_model(config, output_dir=build_dir)

    log_text = caplog.text
    assert "Model Summary: FramerAI" in log_text
    assert "Component Breakdown:" in log_text
    assert "Total Parameters:" in log_text
    assert "Trainable Parameters:" in log_text

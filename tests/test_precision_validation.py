"""Precision validation tests to prevent silent fallback to fp32."""

import pytest

from conftest import tiny_config
from model.configs.model_config import FramerConfig


def test_valid_precisions_are_accepted():
    """Test that all documented precision values are accepted."""
    for precision in ("bf16", "fp16", "fp32"):
        config = tiny_config(precision=precision)
        # Should not raise - validation passes
        config.validate()
        assert config.precision == precision


def test_invalid_precision_bfloat16_is_rejected():
    """Test that common misspelling 'bfloat16' is rejected."""
    with pytest.raises(ValueError, match="precision.*must be one of"):
        tiny_config(precision="bfloat16").validate()


def test_invalid_precision_fp8_is_rejected():
    """Test that unsupported precision 'fp8' is rejected."""
    with pytest.raises(ValueError, match="precision.*must be one of"):
        tiny_config(precision="fp8").validate()


def test_invalid_precision_float16_is_rejected():
    """Test that PyTorch dtype name 'float16' is rejected."""
    with pytest.raises(ValueError, match="precision.*must be one of"):
        tiny_config(precision="float16").validate()


def test_invalid_precision_empty_string_is_rejected():
    """Test that empty string precision is rejected."""
    with pytest.raises(ValueError, match="precision.*must be one of"):
        tiny_config(precision="").validate()


def test_invalid_precision_none_is_rejected():
    """Test that None precision is rejected."""
    with pytest.raises(ValueError, match="precision.*must be one of"):
        tiny_config(precision=None).validate()


def test_precision_validation_error_message_lists_valid_options():
    """Test that validation error message includes all valid precision options."""
    try:
        tiny_config(precision="invalid").validate()
        raise AssertionError("Should have raised ValueError")
    except ValueError as e:
        error_msg = str(e)
        assert "bf16" in error_msg
        assert "fp16" in error_msg
        assert "fp32" in error_msg
        assert "precision" in error_msg


def test_precision_validation_with_custom_config():
    """Test precision validation works with custom FramerConfig instances."""
    # Valid precision should work
    config = FramerConfig(precision="bf16", text_only=True)
    config.validate()

    # Invalid precision should fail
    config = FramerConfig(precision="invalid", text_only=True)
    with pytest.raises(ValueError, match="precision.*must be one of"):
        config.validate()


def test_precision_validation_combined_with_other_errors():
    """Test that precision validation works alongside other validation errors."""
    # Multiple validation errors should be reported together
    with pytest.raises(ValueError) as exc_info:
        FramerConfig(
            precision="invalid",
            d_model=100,  # Not divisible by n_heads=16
            n_layers=0,   # Must be at least 1
        ).validate()

    error_msg = str(exc_info.value)
    assert "precision" in error_msg
    assert "d_model" in error_msg
    assert "n_layers" in error_msg

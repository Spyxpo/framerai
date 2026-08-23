"""Validation tests for safetensors and ONNX export round-trips."""

import numpy as np
import pytest
import torch

from conftest import random_ids, tiny_config
from model.framer import FramerModel


def test_safetensors_export_roundtrip(tmp_path):
    """Verify state dict saved to safetensors reloads into FramerModel and matches logits."""
    safetensors = pytest.importorskip("safetensors.torch")

    config = tiny_config()
    model = FramerModel(config).eval()
    input_ids = random_ids(config, batch=2, length=8)

    with torch.no_grad():
        orig_logits = model.forward_text(input_ids)

    st_path = str(tmp_path / "model.safetensors")
    tensors = {k: v.contiguous() for k, v in model.state_dict().items()}
    # Weight tying makes lm_head share storage with token_embed; drop the duplicate
    tensors.pop("lm_head.weight", None)
    safetensors.save_file(tensors, st_path)

    # Re-instantiate a fresh model
    restored_model = FramerModel(config).eval()
    loaded_state = safetensors.load_file(st_path)
    restored_model.load_state_dict(loaded_state, strict=False)
    # Ensure weight tying is restored
    restored_model.lm_head.weight = restored_model.token_embed.weight

    with torch.no_grad():
        restored_logits = restored_model.forward_text(input_ids)

    assert torch.allclose(orig_logits, restored_logits, rtol=1e-5, atol=1e-5)


def test_onnx_export_roundtrip(tmp_path):
    """Verify ONNX-exported model loads in onnxruntime and produces matching logits."""
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")

    config = tiny_config()
    model = FramerModel(config).eval()
    input_ids = random_ids(config, batch=1, length=12)

    with torch.no_grad():
        pt_logits = model.forward_text(input_ids)

    onnx_path = str(tmp_path / "model.onnx")

    class _ONNXWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, ids):
            return self.m.forward_text(ids)

    dummy_ids = torch.zeros((1, 8), dtype=torch.long)
    torch.onnx.export(
        _ONNXWrapper(model),
        (dummy_ids,),
        onnx_path,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "logits": {0: "batch", 1: "sequence"},
        },
        opset_version=14,
        dynamo=False,
    )

    session = ort.InferenceSession(onnx_path)
    onnx_inputs = {"input_ids": input_ids.numpy()}
    onnx_logits = session.run(None, onnx_inputs)[0]

    np.testing.assert_allclose(pt_logits.detach().numpy(), onnx_logits, rtol=1e-3, atol=1e-3)


def test_onnx_export_dynamic_shapes(tmp_path):
    """Verify ONNX model runs correctly with various batch sizes and sequence lengths."""
    pytest.importorskip("onnx")
    ort = pytest.importorskip("onnxruntime")

    config = tiny_config()
    model = FramerModel(config).eval()

    onnx_path = str(tmp_path / "model_dynamic.onnx")

    class _ONNXWrapper(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, ids):
            return self.m.forward_text(ids)

    dummy_ids = torch.zeros((1, 8), dtype=torch.long)
    torch.onnx.export(
        _ONNXWrapper(model),
        (dummy_ids,),
        onnx_path,
        input_names=["input_ids"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch", 1: "sequence"},
            "logits": {0: "batch", 1: "sequence"},
        },
        opset_version=14,
        dynamo=False,
    )

    session = ort.InferenceSession(onnx_path)

    for batch_size, seq_len in [(1, 4), (3, 16), (2, 32)]:
        ids = random_ids(config, batch=batch_size, length=seq_len)
        with torch.no_grad():
            pt_logits = model.forward_text(ids)

        onnx_logits = session.run(None, {"input_ids": ids.numpy()})[0]
        np.testing.assert_allclose(pt_logits.detach().numpy(), onnx_logits, rtol=1e-3, atol=1e-3)

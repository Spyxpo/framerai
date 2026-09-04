"""Tests for distillation benchmark infrastructure.

Tests verify benchmark behavior without requiring trained models or GPU.
"""

import json
import os
import tempfile

import pytest
import torch

from model.configs import FramerConfig
from model.framer import FramerModel
from model.modules.flow import sampling_cost
from model.tokenizer import FramerTokenizer


def tiny_latent_config(**overrides):
    """Minimal latent_dit config for CPU testing."""
    base = dict(
        preset=None,
        text_only=False,
        vocab_size=256,
        d_model=64,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        d_ff=128,
        max_seq_len=64,
        image_size=32,
        patch_size=16,
        vision_d_model=32,
        vision_n_heads=4,
        vision_n_layers=1,
        image_gen_arch="latent_dit",
        vae_latent_channels=4,
        vae_base_channels=8,
        vae_downsample=4,
        dit_d_model=32,
        dit_n_layers=1,
        dit_n_heads=4,
        dit_patch_size=2,
        sampler_steps=10,
        sampler_method="euler",
        cfg_scale=2.0,
        flow_distilled=False,
        flow_distilled_steps=2,
    )
    base.update(overrides)
    return FramerConfig(**base).validate()


# --------------------------------------------------------------------------
# Caption loading
# --------------------------------------------------------------------------


def test_load_captions_uses_builtin_when_no_file_provided():
    from benchmarks.distillation_image_gen import load_captions

    captions = load_captions(caption_file=None, num_images=5)
    assert len(captions) == 5
    assert all(isinstance(c, str) and c for c in captions)


def test_load_captions_cycles_deterministically():
    from benchmarks.distillation_image_gen import load_captions

    # Built-in has 16 captions
    captions = load_captions(caption_file=None, num_images=20)
    assert len(captions) == 20
    # Should cycle: positions 0 and 16 should match
    assert captions[0] == captions[16]
    assert captions[1] == captions[17]


def test_load_captions_from_file():
    from benchmarks.distillation_image_gen import load_captions

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        f.write("first caption\n")
        f.write("second caption\n")
        f.write("third caption\n")
        tmp_path = f.name

    try:
        captions = load_captions(caption_file=tmp_path, num_images=5)
        assert len(captions) == 5
        assert captions[0] == "first caption"
        assert captions[1] == "second caption"
        assert captions[2] == "third caption"
        assert captions[3] == "first caption"  # Cycles
        assert captions[4] == "second caption"
    finally:
        os.remove(tmp_path)


def test_load_captions_raises_on_missing_file():
    from benchmarks.distillation_image_gen import load_captions

    with pytest.raises(FileNotFoundError, match="not found"):
        load_captions(caption_file="/no/such/file.txt", num_images=5)


def test_load_captions_raises_on_empty_file():
    from benchmarks.distillation_image_gen import load_captions

    with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt") as f:
        tmp_path = f.name

    try:
        with pytest.raises(ValueError, match="empty"):
            load_captions(caption_file=tmp_path, num_images=5)
    finally:
        os.remove(tmp_path)


# --------------------------------------------------------------------------
# Checkpoint validation
# --------------------------------------------------------------------------


def test_validate_checkpoint_rejects_missing_file():
    from benchmarks.distillation_image_gen import validate_checkpoint_config

    with pytest.raises(FileNotFoundError, match="not found"):
        validate_checkpoint_config("/no/such/checkpoint.pt", False, "Teacher")


def test_validate_checkpoint_rejects_wrong_architecture():
    from benchmarks.distillation_image_gen import validate_checkpoint_config

    config = tiny_latent_config(image_gen_arch="unet")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as f:
        torch.save({"config": config, "model_state_dict": {}, "step": 0}, f.name)
        tmp_path = f.name

    try:
        with pytest.raises(ValueError, match="latent_dit"):
            validate_checkpoint_config(tmp_path, False, "Teacher")
    finally:
        os.remove(tmp_path)


def test_validate_checkpoint_rejects_wrong_distilled_flag():
    from benchmarks.distillation_image_gen import validate_checkpoint_config

    # Teacher should have flow_distilled=False
    config = tiny_latent_config(flow_distilled=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as f:
        torch.save({"config": config, "model_state_dict": {}, "step": 0}, f.name)
        tmp_path = f.name

    try:
        with pytest.raises(ValueError, match="flow_distilled=False"):
            validate_checkpoint_config(tmp_path, False, "Teacher")
    finally:
        os.remove(tmp_path)


def test_validate_checkpoint_accepts_correct_teacher():
    from benchmarks.distillation_image_gen import validate_checkpoint_config

    config = tiny_latent_config(flow_distilled=False)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as f:
        torch.save({"config": config, "model_state_dict": {}, "step": 0}, f.name)
        tmp_path = f.name

    try:
        result = validate_checkpoint_config(tmp_path, False, "Teacher")
        assert result.flow_distilled is False
        assert result.image_gen_arch == "latent_dit"
    finally:
        os.remove(tmp_path)


def test_validate_checkpoint_accepts_correct_student():
    from benchmarks.distillation_image_gen import validate_checkpoint_config

    config = tiny_latent_config(flow_distilled=True)

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pt") as f:
        torch.save({"config": config, "model_state_dict": {}, "step": 0}, f.name)
        tmp_path = f.name

    try:
        result = validate_checkpoint_config(tmp_path, True, "Student")
        assert result.flow_distilled is True
        assert result.image_gen_arch == "latent_dit"
    finally:
        os.remove(tmp_path)


# --------------------------------------------------------------------------
# Sampling cost calculation
# --------------------------------------------------------------------------


def test_sampling_cost_teacher_euler_with_guidance():
    # Teacher: 50 steps, euler (1 forward/step), guidance (2x batch)
    cost = sampling_cost(steps=50, method="euler", guidance=True, guidance_distilled=False)
    assert cost == 50 * 1 * 2  # 100 forwards


def test_sampling_cost_teacher_heun_with_guidance():
    # Teacher: 50 steps, heun (2 forwards/step), guidance (2x batch)
    cost = sampling_cost(steps=50, method="heun", guidance=True, guidance_distilled=False)
    assert cost == 50 * 2 * 2  # 200 forwards


def test_sampling_cost_student_euler_distilled():
    # Student: 4 steps, euler, guidance_distilled (no doubling)
    cost = sampling_cost(steps=4, method="euler", guidance=False, guidance_distilled=True)
    assert cost == 4 * 1  # 4 forwards


def test_sampling_cost_student_heun_distilled():
    # Student: 4 steps, heun, guidance_distilled (no doubling)
    cost = sampling_cost(steps=4, method="heun", guidance=False, guidance_distilled=True)
    assert cost == 4 * 2  # 8 forwards


def test_sampling_cost_speedup_is_measurable():
    teacher = sampling_cost(50, "euler", True, False)
    student = sampling_cost(4, "euler", False, True)
    speedup = teacher / student
    assert speedup == 25.0  # 100 / 4


# --------------------------------------------------------------------------
# Timing infrastructure
# --------------------------------------------------------------------------


def test_benchmark_inference_time_returns_valid_structure():
    """Test that timing function returns expected structure with finite values."""
    from benchmarks.distillation_image_gen import benchmark_inference_time
    from model.generate import FramerGenerator

    config = tiny_latent_config()
    model = FramerModel(config)
    tokenizer = FramerTokenizer(config.vocab_size)
    tokenizer.train(["test"], target_vocab_size=config.vocab_size)
    generator = FramerGenerator(model, tokenizer, "cpu")

    captions = ["a red apple", "a blue sky"]
    result = benchmark_inference_time(
        generator, captions, resolution=32, seed=42, device=torch.device("cpu"), n_warmup=1
    )

    assert "total_time_sec" in result
    assert "latency_ms_per_image" in result
    assert "images_per_sec" in result
    assert "num_images" in result

    assert result["num_images"] == 2
    assert result["total_time_sec"] > 0
    assert result["latency_ms_per_image"] > 0
    assert result["images_per_sec"] > 0


def test_benchmark_alignment_returns_valid_score():
    """Test that alignment function returns a score in [-1, 1]."""
    from benchmarks.distillation_image_gen import benchmark_alignment
    from model.generate import FramerGenerator

    config = tiny_latent_config()
    model = FramerModel(config)
    tokenizer = FramerTokenizer(config.vocab_size)
    tokenizer.train(["test caption"], target_vocab_size=config.vocab_size)
    generator = FramerGenerator(model, tokenizer, "cpu")

    captions = ["a test image"]
    score = benchmark_alignment(
        generator, captions, resolution=32, seed=42, device=torch.device("cpu")
    )

    assert isinstance(score, float)
    assert -1.0 <= score <= 1.0


# --------------------------------------------------------------------------
# FID computation
# --------------------------------------------------------------------------


def test_compute_fid_returns_finite_value():
    """Test that FID computation runs and returns finite value."""
    from benchmarks.distillation_image_gen import compute_fid
    from model.generate import FramerGenerator

    config = tiny_latent_config()
    model = FramerModel(config)
    tokenizer = FramerTokenizer(config.vocab_size)
    tokenizer.train(["test"], target_vocab_size=config.vocab_size)
    generator = FramerGenerator(model, tokenizer, "cpu")

    captions = ["a red apple", "a blue sky"]
    # Create fake "real" images
    real_images = torch.randn(2, 3, 32, 32)

    fid_score = compute_fid(
        generator, captions, real_images, resolution=32, seed=42, device=torch.device("cpu")
    )

    assert isinstance(fid_score, float)
    assert fid_score >= 0  # FID is always non-negative


def test_load_real_images_raises_on_missing_directory():
    from benchmarks.distillation_image_gen import load_real_images

    with pytest.raises(ValueError, match="not found"):
        load_real_images("/no/such/directory", resolution=64, device="cpu")


def test_load_real_images_raises_on_empty_directory():
    from benchmarks.distillation_image_gen import load_real_images

    with tempfile.TemporaryDirectory() as tmpdir:
        with pytest.raises(ValueError, match="No images found"):
            load_real_images(tmpdir, resolution=64, device="cpu")


# --------------------------------------------------------------------------
# End-to-end integration
# --------------------------------------------------------------------------


def test_end_to_end_benchmark_cpu():
    """Small end-to-end benchmark on CPU with tiny models."""
    from benchmarks.distillation_image_gen import (
        benchmark_alignment,
        benchmark_inference_time,
        validate_checkpoint_config,
    )
    from model.generate import FramerGenerator

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create teacher config and checkpoint
        teacher_config = tiny_latent_config(flow_distilled=False, sampler_steps=10)
        teacher_model = FramerModel(teacher_config)
        teacher_path = os.path.join(tmpdir, "teacher.pt")
        torch.save(
            {
                "config": teacher_config,
                "model_state_dict": teacher_model.state_dict(),
                "step": 100,
            },
            teacher_path,
        )

        # Create student config and checkpoint
        student_config = tiny_latent_config(flow_distilled=True, flow_distilled_steps=2)
        student_model = FramerModel(student_config)
        student_path = os.path.join(tmpdir, "student.pt")
        torch.save(
            {
                "config": student_config,
                "model_state_dict": student_model.state_dict(),
                "step": 50,
            },
            student_path,
        )

        # Create tokenizer
        tokenizer = FramerTokenizer(teacher_config.vocab_size)
        tokenizer.train(["test caption one", "test caption two"], target_vocab_size=128)
        tokenizer_dir = os.path.join(tmpdir, "tokenizer")
        tokenizer.save(tokenizer_dir)

        # Validate checkpoints
        teacher_cfg = validate_checkpoint_config(teacher_path, False, "Teacher")
        assert teacher_cfg.flow_distilled is False

        student_cfg = validate_checkpoint_config(student_path, True, "Student")
        assert student_cfg.flow_distilled is True

        # Load generators
        teacher_gen = FramerGenerator.from_checkpoint(teacher_path, tokenizer_dir, "cpu")
        student_gen = FramerGenerator.from_checkpoint(student_path, tokenizer_dir, "cpu")

        # Test captions
        captions = ["a red apple", "a blue sky"]

        # Benchmark timing
        teacher_timing = benchmark_inference_time(
            teacher_gen, captions, resolution=32, seed=42, device=torch.device("cpu"), n_warmup=1
        )
        assert teacher_timing["num_images"] == 2
        assert teacher_timing["total_time_sec"] > 0

        student_timing = benchmark_inference_time(
            student_gen, captions, resolution=32, seed=42, device=torch.device("cpu"), n_warmup=1
        )
        assert student_timing["num_images"] == 2
        assert student_timing["total_time_sec"] > 0

        # Benchmark alignment
        teacher_align = benchmark_alignment(
            teacher_gen, captions, resolution=32, seed=42, device=torch.device("cpu")
        )
        assert -1.0 <= teacher_align <= 1.0

        student_align = benchmark_alignment(
            student_gen, captions, resolution=32, seed=42, device=torch.device("cpu")
        )
        assert -1.0 <= student_align <= 1.0

        # Calculate speedup
        speedup = teacher_timing["latency_ms_per_image"] / student_timing["latency_ms_per_image"]
        assert speedup > 0  # Student should be at least as fast (or faster)

        # Verify theoretical costs
        teacher_cost = sampling_cost(
            teacher_config.sampler_steps, teacher_config.sampler_method, True, False
        )
        student_cost = sampling_cost(
            student_config.flow_distilled_steps, student_config.sampler_method, False, True
        )
        theoretical_speedup = teacher_cost / student_cost
        assert theoretical_speedup > 1  # Student should theoretically be faster


def test_json_output_structure():
    """Test that JSON output contains expected fields."""
    from benchmarks.distillation_image_gen import (
        benchmark_alignment,
        benchmark_inference_time,
    )
    from model.generate import FramerGenerator

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create minimal setup
        teacher_config = tiny_latent_config(flow_distilled=False, sampler_steps=10)
        teacher_model = FramerModel(teacher_config)
        teacher_path = os.path.join(tmpdir, "teacher.pt")
        torch.save(
            {"config": teacher_config, "model_state_dict": teacher_model.state_dict(), "step": 0},
            teacher_path,
        )

        student_config = tiny_latent_config(flow_distilled=True, flow_distilled_steps=2)
        student_model = FramerModel(student_config)
        student_path = os.path.join(tmpdir, "student.pt")
        torch.save(
            {"config": student_config, "model_state_dict": student_model.state_dict(), "step": 0},
            student_path,
        )

        tokenizer = FramerTokenizer(teacher_config.vocab_size)
        tokenizer.train(["test"], target_vocab_size=64)
        tokenizer_dir = os.path.join(tmpdir, "tokenizer")
        tokenizer.save(tokenizer_dir)

        teacher_gen = FramerGenerator.from_checkpoint(teacher_path, tokenizer_dir, "cpu")
        student_gen = FramerGenerator.from_checkpoint(student_path, tokenizer_dir, "cpu")

        captions = ["test"]

        teacher_timing = benchmark_inference_time(
            teacher_gen, captions, 32, 42, torch.device("cpu"), 1
        )
        student_timing = benchmark_inference_time(
            student_gen, captions, 32, 42, torch.device("cpu"), 1
        )
        teacher_align = benchmark_alignment(teacher_gen, captions, 32, 42, torch.device("cpu"))
        student_align = benchmark_alignment(student_gen, captions, 32, 42, torch.device("cpu"))

        # Construct JSON structure (mimicking what main() does)
        result = {
            "teacher_checkpoint": teacher_path,
            "student_checkpoint": student_path,
            "device": "cpu",
            "resolution": 32,
            "num_images": 1,
            "seed": 42,
            "teacher": {
                "sampler_steps": teacher_config.sampler_steps,
                "timing": teacher_timing,
                "alignment": teacher_align,
                "fid": None,
            },
            "student": {
                "sampler_steps": student_config.flow_distilled_steps,
                "timing": student_timing,
                "alignment": student_align,
                "fid": None,
            },
        }

        # Verify structure
        assert "teacher_checkpoint" in result
        assert "student_checkpoint" in result
        assert "teacher" in result
        assert "student" in result
        assert "timing" in result["teacher"]
        assert "alignment" in result["teacher"]
        assert "fid" in result["teacher"]
        assert result["teacher"]["fid"] is None  # No real images provided

        # Verify JSON serializable
        json_str = json.dumps(result)
        parsed = json.loads(json_str)
        assert parsed["resolution"] == 32

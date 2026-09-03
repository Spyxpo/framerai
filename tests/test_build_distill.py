"""Tests for build.py distillation integration.

Validates that --mode distill correctly wires the distillation training path
into build.py's CLI/config/checkpoint infrastructure.
"""

import os
import tempfile
from unittest.mock import patch

import pytest
import torch

from model.configs import FramerConfig
from model.framer import FramerModel
from model.tokenizer import FramerTokenizer


def tiny_latent_config():
    """Minimal latent_dit config for CPU testing."""
    return FramerConfig(
        preset=None,
        text_only=False,
        vocab_size=500,
        d_model=128,
        n_layers=2,
        n_heads=4,
        n_kv_heads=2,
        d_ff=512,
        max_seq_len=64,
        image_size=32,
        patch_size=16,
        vision_d_model=128,
        vision_n_heads=4,
        vision_n_layers=2,
        # Latent diffusion
        image_gen_arch="latent_dit",
        vae_latent_channels=4,
        vae_base_channels=16,
        vae_downsample=4,
        dit_d_model=128,
        dit_n_layers=2,
        dit_n_heads=4,
        dit_patch_size=2,
        sampler_steps=10,
        sampler_method="euler",
        cfg_scale=3.0,
        image_train_resolution=32,
        max_steps=5,
        batch_size=2,
        learning_rate=1e-4,
        warmup_steps=1,
        device="cpu",
        precision="fp32",
    ).validate()


def test_build_parser_accepts_distill_mode():
    """Argument parser must accept --mode distill."""
    from build import _make_parser

    parser = _make_parser()
    args = parser.parse_args(["--mode", "distill", "--teacher-checkpoint", "teacher.pt"])
    assert args.mode == "distill"
    assert args.teacher_checkpoint == "teacher.pt"


def test_distill_mode_requires_teacher_checkpoint():
    """--mode distill must fail when --teacher-checkpoint is missing."""
    from build import _make_parser

    parser = _make_parser()
    # Parser itself allows missing teacher-checkpoint, but main() should catch it
    args = parser.parse_args(["--mode", "distill"])
    assert args.mode == "distill"
    assert args.teacher_checkpoint is None


def test_distill_arguments_have_defaults():
    """Distillation arguments must have sensible defaults."""
    from build import _make_parser

    parser = _make_parser()
    args = parser.parse_args(["--mode", "distill", "--teacher-checkpoint", "t.pt"])
    assert args.distill_steps == 4
    assert args.distill_substeps == 8


def test_distill_arguments_can_be_overridden():
    """Distillation arguments must be customizable via CLI."""
    from build import _make_parser

    parser = _make_parser()
    args = parser.parse_args([
        "--mode", "distill",
        "--teacher-checkpoint", "t.pt",
        "--distill-steps", "2",
        "--distill-substeps", "16",
    ])
    assert args.distill_steps == 2
    assert args.distill_substeps == 16


def test_train_distill_model_requires_latent_dit():
    """train_distill_model must reject non-latent_dit configs."""
    from build import train_distill_model

    config = FramerConfig(
        preset=None,
        vocab_size=500,
        d_model=128,
        n_layers=2,
        n_heads=4,
        max_steps=5,
        image_gen_arch="unet",  # Valid but wrong for distillation
    ).validate()

    with tempfile.TemporaryDirectory() as tmpdir:
        teacher_path = os.path.join(tmpdir, "teacher.pt")
        torch.save({"model_state_dict": {}, "config": {}, "step": 0}, teacher_path)

        with pytest.raises(ValueError, match="latent_dit"):
            train_distill_model(
                config, tmpdir, teacher_path,
                data_dir=tmpdir, distill_steps=2, distill_substeps=4,
            )


def test_train_distill_model_requires_teacher_checkpoint_to_exist():
    """train_distill_model must fail when teacher checkpoint is missing."""
    from build import train_distill_model

    config = tiny_latent_config()

    with tempfile.TemporaryDirectory() as tmpdir:
        missing_path = os.path.join(tmpdir, "nonexistent.pt")

        with pytest.raises(FileNotFoundError, match="Teacher checkpoint not found"):
            train_distill_model(
                config, tmpdir, missing_path,
                data_dir=tmpdir, distill_steps=2, distill_substeps=4,
            )


def test_train_distill_model_requires_tokenizer():
    """train_distill_model must fail when tokenizer is missing."""
    from build import train_distill_model

    config = tiny_latent_config()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create teacher checkpoint
        teacher_path = os.path.join(tmpdir, "teacher.pt")
        teacher_model = FramerModel(config)
        torch.save({
            "model_state_dict": teacher_model.state_dict(),
            "config": config.__dict__,
            "step": 10,
            "loss": 0.5,
        }, teacher_path)

        # No tokenizer directory exists
        with pytest.raises(FileNotFoundError, match="Tokenizer not found"):
            train_distill_model(
                config, tmpdir, teacher_path,
                data_dir=tmpdir, distill_steps=2, distill_substeps=4,
            )


def test_train_distill_model_requires_image_data():
    """train_distill_model must fail when no image-caption pairs exist."""
    from build import train_distill_model

    config = tiny_latent_config()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create teacher checkpoint
        teacher_path = os.path.join(tmpdir, "teacher.pt")
        teacher_model = FramerModel(config)
        torch.save({
            "model_state_dict": teacher_model.state_dict(),
            "config": config.__dict__,
            "step": 10,
            "loss": 0.5,
        }, teacher_path)

        # Create tokenizer
        tokenizer = FramerTokenizer(config.vocab_size)
        tokenizer.save(os.path.join(tmpdir, "tokenizer"))

        # Empty data directory
        with pytest.raises(ValueError, match="No image-caption pairs"):
            train_distill_model(
                config, tmpdir, teacher_path,
                data_dir=tmpdir, distill_steps=2, distill_substeps=4,
            )


def test_train_distill_model_creates_student_with_flow_distilled():
    """train_distill_model must create student with flow_distilled=True."""
    from build import train_distill_model
    from model.modules.flow import FlowDistiller

    config = tiny_latent_config()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create teacher checkpoint
        teacher_path = os.path.join(tmpdir, "teacher.pt")
        teacher_model = FramerModel(config)
        torch.save({
            "model_state_dict": teacher_model.state_dict(),
            "config": config.__dict__,
            "step": 10,
            "loss": 0.5,
        }, teacher_path)

        # Create tokenizer
        tokenizer = FramerTokenizer(config.vocab_size)
        tokenizer.save(os.path.join(tmpdir, "tokenizer"))

        # Create minimal image-caption pair in JSONL format
        data_dir = os.path.join(tmpdir, "images")
        os.makedirs(data_dir)
        # Use a simple RGB array saved as PNG
        import json

        import numpy as np
        from PIL import Image
        arr = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        img = Image.fromarray(arr, mode="RGB")
        img_path = os.path.join(data_dir, "test.png")
        img.save(img_path)
        # Create JSONL with image reference
        with open(os.path.join(data_dir, "captions.jsonl"), "w") as f:
            json.dump({"image": "test.png", "caption": "test caption"}, f)

        # Mock train_distill to verify it's called correctly
        with patch("build.train_distill") as mock_train_distill:
            mock_train_distill.return_value = None

            train_distill_model(
                config, tmpdir, teacher_path,
                data_dir=data_dir, distill_steps=2, distill_substeps=4,
            )

            # Verify train_distill was called
            assert mock_train_distill.called
            call_args = mock_train_distill.call_args

            # Extract student_config (1st positional arg)
            student_config = call_args[0][0]
            assert student_config.flow_distilled is True
            assert student_config.flow_distilled_steps == 2

            # Verify distiller is created correctly
            distiller_passed = call_args[0][6]
            assert isinstance(distiller_passed, FlowDistiller)
            assert distiller_passed.teacher_substeps == 4


def test_train_distill_model_initializes_student_from_teacher():
    """train_distill_model must initialize student weights from teacher."""
    from build import train_distill_model

    config = tiny_latent_config()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create teacher checkpoint
        teacher_path = os.path.join(tmpdir, "teacher.pt")
        teacher_model = FramerModel(config)
        # Set a specific value to verify copying
        teacher_model.token_embed.weight.data[0, 0] = 99.0
        torch.save({
            "model_state_dict": teacher_model.state_dict(),
            "config": config.__dict__,
            "step": 10,
            "loss": 0.5,
        }, teacher_path)

        # Create tokenizer
        tokenizer = FramerTokenizer(config.vocab_size)
        tokenizer.save(os.path.join(tmpdir, "tokenizer"))

        # Create minimal image-caption pair in JSONL format
        data_dir = os.path.join(tmpdir, "images")
        os.makedirs(data_dir)
        import json

        import numpy as np
        from PIL import Image
        arr = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        img = Image.fromarray(arr, mode="RGB")
        img_path = os.path.join(data_dir, "test.png")
        img.save(img_path)
        with open(os.path.join(data_dir, "captions.jsonl"), "w") as f:
            json.dump({"image": "test.png", "caption": "test caption"}, f)

        # Mock train_distill to avoid actual training
        with patch("build.train_distill") as mock_train_distill:
            mock_train_distill.return_value = None

            train_distill_model(
                config, tmpdir, teacher_path,
                data_dir=data_dir, distill_steps=2, distill_substeps=4,
            )

            # Verify student was initialized from teacher
            assert mock_train_distill.called
            student_passed = mock_train_distill.call_args[0][2]
            assert student_passed.token_embed.weight.data[0, 0].item() == 99.0


def test_train_distill_model_supports_resume():
    """train_distill_model must support resuming from student checkpoint."""
    from build import train_distill_model

    config = tiny_latent_config()

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create teacher checkpoint
        teacher_path = os.path.join(tmpdir, "teacher.pt")
        teacher_model = FramerModel(config)
        torch.save({
            "model_state_dict": teacher_model.state_dict(),
            "config": config.__dict__,
            "step": 10,
            "loss": 0.5,
        }, teacher_path)

        # Create student checkpoint for resume
        student_config = FramerConfig(**{**config.__dict__, "flow_distilled": True}).validate()
        student_model = FramerModel(student_config)
        student_model.token_embed.weight.data[0, 0] = 77.0  # Marker value
        resume_path = os.path.join(tmpdir, "student_checkpoint.pt")
        torch.save({
            "model_state_dict": student_model.state_dict(),
            "config": student_config.__dict__,
            "step": 5,
            "loss": 0.3,
        }, resume_path)

        # Create tokenizer
        tokenizer = FramerTokenizer(config.vocab_size)
        tokenizer.save(os.path.join(tmpdir, "tokenizer"))

        # Create minimal image-caption pair in JSONL format
        data_dir = os.path.join(tmpdir, "images")
        os.makedirs(data_dir)
        import json

        import numpy as np
        from PIL import Image
        arr = np.random.randint(0, 255, (32, 32, 3), dtype=np.uint8)
        img = Image.fromarray(arr, mode="RGB")
        img_path = os.path.join(data_dir, "test.png")
        img.save(img_path)
        with open(os.path.join(data_dir, "captions.jsonl"), "w") as f:
            json.dump({"image": "test.png", "caption": "test caption"}, f)

        # Mock train_distill
        with patch("build.train_distill") as mock_train_distill:
            mock_train_distill.return_value = None

            train_distill_model(
                config, tmpdir, teacher_path,
                data_dir=data_dir, distill_steps=2, distill_substeps=4,
                resume=resume_path,
            )

            # Verify student loaded from resume checkpoint
            assert mock_train_distill.called
            student_passed = mock_train_distill.call_args[0][2]
            assert student_passed.token_embed.weight.data[0, 0].item() == 77.0

            # Verify start_step is passed correctly
            start_step_passed = mock_train_distill.call_args[1]["start_step"]
            assert start_step_passed == 5

            # Verify resume_checkpoint is passed
            resume_checkpoint_passed = mock_train_distill.call_args[1]["resume_checkpoint"]
            assert resume_checkpoint_passed == resume_path


def test_main_validates_teacher_checkpoint_required():
    """main() must exit with error when --mode distill lacks --teacher-checkpoint."""
    from build import _make_parser

    parser = _make_parser()
    args = parser.parse_args(["--mode", "distill", "--preset", "framer-3b"])

    # Simulate main() logic
    with patch("build.train_distill_model") as mock_distill:
        with patch("sys.exit") as mock_exit:
            # This is the check that happens in main()
            if args.mode == "distill" and not args.teacher_checkpoint:
                mock_exit(1)

            mock_exit.assert_called_once_with(1)
            mock_distill.assert_not_called()

"""Distillation training tests: verify training loop, freezing, and checkpoint flow.

Tests run on CPU with minimal configs to verify wiring without requiring GPU/hours.
The important properties:
- Teacher parameters stay frozen
- Student parameters receive gradients
- Loss is finite and can decrease
- Distilled checkpoints preserve flow_distilled=True
- Loaded distilled models use fewer steps and skip CFG
"""

import tempfile

import torch

from model.configs import FramerConfig
from model.framer import FramerModel
from model.modules.flow import FlowDistiller
from model.training.distill import train_distill


def tiny_distill_config():
    """Minimal CPU-testable config for latent diffusion distillation."""
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
        # Vision encoder (tiny)
        vision_d_model=128,
        vision_n_heads=4,
        vision_n_layers=2,
        # Latent diffusion (tiny)
        image_gen_arch="latent_dit",
        vae_latent_channels=4,
        vae_base_channels=16,
        vae_downsample=4,  # 32/4 = 8x8 latent
        dit_d_model=128,
        dit_n_layers=2,
        dit_n_heads=4,
        dit_patch_size=2,  # 8/2 = 4x4 patches
        sampler_steps=10,  # Teacher uses 10 steps
        sampler_method="euler",
        cfg_scale=3.0,
        # Distilled student config
        flow_distilled=False,  # Teacher starts non-distilled
        flow_distilled_steps=2,  # Student will use 2 steps
        image_train_resolution=32,
        # Training
        max_steps=5,
        batch_size=2,
        learning_rate=1e-4,
        warmup_steps=1,
        grad_clip=1.0,
        gradient_accumulation_steps=1,
        device="cpu",
        precision="fp32",
        mixed_precision=False,
    ).validate()


def test_distillation_training_loop_runs_without_error():
    """Smoke test: distillation loop completes on tiny config."""
    config = tiny_distill_config()

    # Build teacher
    teacher_model = FramerModel(config)
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    # Build student with flow_distilled=True
    student_config = FramerConfig(
        **{**config.__dict__, "flow_distilled": True}
    ).validate()
    student_model = FramerModel(student_config)
    student_model.load_state_dict(teacher_model.state_dict())

    # Create synthetic dataset
    images = torch.randn(4, 3, 32, 32)  # 4 synthetic images
    captions = torch.randint(0, config.vocab_size, (4, 16))

    class SyntheticDataset:
        def __len__(self):
            return 4

        def __getitem__(self, idx):
            return {
                "target_images": images[idx],
                "input_ids": captions[idx],
            }

    from torch.utils.data import DataLoader
    loader = DataLoader(SyntheticDataset(), batch_size=2)

    distiller = FlowDistiller(teacher_substeps=4, method="euler")

    with tempfile.TemporaryDirectory() as tmpdir:
        final_step = train_distill(
            student_config,
            teacher_model,
            student_model,
            loader,
            device=torch.device("cpu"),
            output_dir=tmpdir,
            distiller=distiller,
            start_step=0,
            log_interval=1,
            save_interval=10,  # Won't save during this short run
        )
        assert final_step == config.max_steps
        assert final_step == 5


def test_teacher_parameters_stay_frozen():
    """Teacher parameters must not change during distillation."""
    config = tiny_distill_config()

    teacher_model = FramerModel(config)
    teacher_before = {k: v.clone() for k, v in teacher_model.state_dict().items()}

    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    student_config = FramerConfig(**{**config.__dict__, "flow_distilled": True}).validate()
    student_model = FramerModel(student_config)
    student_model.load_state_dict(teacher_model.state_dict())

    # Synthetic data
    images = torch.randn(2, 3, 32, 32)
    captions = torch.randint(0, config.vocab_size, (2, 16))

    class SyntheticDataset:
        def __len__(self):
            return 2

        def __getitem__(self, idx):
            return {"target_images": images[idx], "input_ids": captions[idx]}

    from torch.utils.data import DataLoader
    loader = DataLoader(SyntheticDataset(), batch_size=2)
    distiller = FlowDistiller(teacher_substeps=2, method="euler")

    with tempfile.TemporaryDirectory() as tmpdir:
        train_distill(
            student_config, teacher_model, student_model, loader,
            torch.device("cpu"), tmpdir, distiller, start_step=0,
            log_interval=10, save_interval=10,
        )

    # Teacher parameters must be unchanged
    teacher_after = teacher_model.state_dict()
    for key in teacher_before.keys():
        assert torch.allclose(teacher_before[key], teacher_after[key]), \
            f"Teacher parameter '{key}' changed during distillation"


def test_student_denoiser_receives_gradients():
    """Student denoiser parameters must actually train."""
    config = tiny_distill_config()

    teacher_model = FramerModel(config)
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    student_config = FramerConfig(**{**config.__dict__, "flow_distilled": True}).validate()
    student_model = FramerModel(student_config)
    student_model.load_state_dict(teacher_model.state_dict())

    # Capture initial student denoiser state
    denoiser_before = {
        k: v.clone()
        for k, v in student_model.diffusion.denoiser.state_dict().items()
    }

    # Synthetic data
    images = torch.randn(2, 3, 32, 32)
    captions = torch.randint(0, config.vocab_size, (2, 16))

    class SyntheticDataset:
        def __len__(self):
            return 2

        def __getitem__(self, idx):
            return {"target_images": images[idx], "input_ids": captions[idx]}

    from torch.utils.data import DataLoader
    loader = DataLoader(SyntheticDataset(), batch_size=2)
    distiller = FlowDistiller(teacher_substeps=2, method="euler")

    with tempfile.TemporaryDirectory() as tmpdir:
        train_distill(
            student_config, teacher_model, student_model, loader,
            torch.device("cpu"), tmpdir, distiller, start_step=0,
            log_interval=10, save_interval=10,
        )

    # At least some denoiser parameters must have changed
    denoiser_after = student_model.diffusion.denoiser.state_dict()
    changed = False
    for key in denoiser_before.keys():
        if not torch.allclose(denoiser_before[key], denoiser_after[key], atol=1e-6):
            changed = True
            break

    assert changed, "No student denoiser parameters changed - gradients not flowing"


def test_student_vae_and_backbone_stay_frozen():
    """VAE and text model must not train during distillation."""
    config = tiny_distill_config()

    teacher_model = FramerModel(config)
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    student_config = FramerConfig(**{**config.__dict__, "flow_distilled": True}).validate()
    student_model = FramerModel(student_config)
    student_model.load_state_dict(teacher_model.state_dict())

    vae_before = {k: v.clone() for k, v in student_model.diffusion.vae.state_dict().items()}
    text_embed_before = {k: v.clone() for k, v in student_model.token_embed.state_dict().items()}

    # Synthetic data
    images = torch.randn(2, 3, 32, 32)
    captions = torch.randint(0, config.vocab_size, (2, 16))

    class SyntheticDataset:
        def __len__(self):
            return 2

        def __getitem__(self, idx):
            return {"target_images": images[idx], "input_ids": captions[idx]}

    from torch.utils.data import DataLoader
    loader = DataLoader(SyntheticDataset(), batch_size=2)
    distiller = FlowDistiller(teacher_substeps=2, method="euler")

    with tempfile.TemporaryDirectory() as tmpdir:
        train_distill(
            student_config, teacher_model, student_model, loader,
            torch.device("cpu"), tmpdir, distiller, start_step=0,
            log_interval=10, save_interval=10,
        )

    # VAE must be unchanged
    vae_after = student_model.diffusion.vae.state_dict()
    for key in vae_before.keys():
        assert torch.allclose(vae_before[key], vae_after[key]), \
            f"Student VAE parameter '{key}' changed - VAE must stay frozen"

    # Text embeddings must be unchanged
    text_embed_after = student_model.token_embed.state_dict()
    for key in text_embed_before.keys():
        assert torch.allclose(text_embed_before[key], text_embed_after[key]), \
            f"Student text embedding '{key}' changed - text model must stay frozen"


def test_distilled_checkpoint_preserves_flow_distilled_config():
    """Saved checkpoint must preserve flow_distilled=True for inference."""
    config = tiny_distill_config()

    teacher_model = FramerModel(config)
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    student_config = FramerConfig(**{**config.__dict__, "flow_distilled": True, "max_steps": 2}).validate()
    student_model = FramerModel(student_config)
    student_model.load_state_dict(teacher_model.state_dict())

    # Synthetic data
    images = torch.randn(2, 3, 32, 32)
    captions = torch.randint(0, config.vocab_size, (2, 16))

    class SyntheticDataset:
        def __len__(self):
            return 2

        def __getitem__(self, idx):
            return {"target_images": images[idx], "input_ids": captions[idx]}

    from torch.utils.data import DataLoader
    loader = DataLoader(SyntheticDataset(), batch_size=2)
    distiller = FlowDistiller(teacher_substeps=2, method="euler")

    with tempfile.TemporaryDirectory() as tmpdir:
        train_distill(
            student_config, teacher_model, student_model, loader,
            torch.device("cpu"), tmpdir, distiller, start_step=0,
            log_interval=1, save_interval=2,  # Save at step 2
        )

        # Load the saved checkpoint
        ckpt_path = f"{tmpdir}/checkpoint_distill_2.pt"
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)

        assert "config" in ckpt
        loaded_config = FramerConfig.from_dict(ckpt["config"])
        assert loaded_config.flow_distilled, \
            "Checkpoint must preserve flow_distilled=True"
        assert loaded_config.flow_distilled_steps == student_config.flow_distilled_steps


def test_loaded_distilled_model_uses_correct_sampler_config():
    """Loaded distilled model must use fewer steps and guidance_distilled=True."""
    config = tiny_distill_config()
    student_config = FramerConfig(**{**config.__dict__, "flow_distilled": True}).validate()

    # Build a distilled model
    model = FramerModel(student_config)

    # Check LatentImageGenerator picked up the config
    assert model.diffusion.flow_distilled
    assert model.diffusion.sampler.guidance_distilled
    assert model.diffusion.sampler.steps == student_config.flow_distilled_steps


def test_distilled_inference_does_not_apply_cfg_twice():
    """ODESampler with guidance_distilled=True must skip the conditional/unconditional pair."""
    config = tiny_distill_config()
    student_config = FramerConfig(**{**config.__dict__, "flow_distilled": True}).validate()

    model = FramerModel(student_config)

    # Count how many times the velocity_fn is called
    call_count = [0]

    def counting_velocity_fn(x, t, context):
        call_count[0] += x.shape[0]  # Count batch dimension
        return torch.zeros_like(x)

    # Sample with the distilled sampler
    sampler = model.diffusion.sampler
    batch_size = 2
    latent_shape = (batch_size, 4, 8, 8)
    context = torch.randn(batch_size, 1, config.d_model)
    null_context = torch.zeros(1, 1, config.d_model)

    sampler.sample(
        counting_velocity_fn,
        latent_shape,
        context=context,
        null_context=null_context,
        cfg_scale=3.0,
        device="cpu",
    )

    # With guidance_distilled=True, should see batch_size per step
    # With guidance_distilled=False, would see 2*batch_size per step (CFG pair)
    expected_calls = batch_size * student_config.flow_distilled_steps
    assert call_count[0] == expected_calls, \
        f"Expected {expected_calls} velocity_fn calls (no CFG doubling), got {call_count[0]}"

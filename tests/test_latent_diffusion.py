"""Latent image generation: VAE, diffusion transformer, rectified flow, CFG.

The pixel-space U-Net cannot run at the resolutions it is configured for, and
the classifier-free guidance the README advertised did not exist. These tests
cover the replacement: the compression that makes high resolution tractable, the
adaLN-zero initialization that makes a deep denoiser trainable, the flow
objective that replaces a 1000-step chain with tens of steps, and guidance
against a learned null embedding.
"""

import pytest
import torch

from conftest import tiny_config
from model.framer import FramerModel
from model.modules.dit import DiT, DiTBlock, sincos_pos_embed_2d
from model.modules.flow import ODESampler, RectifiedFlow
from model.modules.latent_diffusion import LatentImageGenerator, build_image_generator
from model.modules.vae import KLVAE, DiagonalGaussian


def latent_config(**overrides):
    """Smallest config that exercises the latent path end to end."""
    base = dict(
        text_only=False,
        image_gen_arch="latent_dit",
        image_size=32,
        patch_size=16,
        vision_d_model=32,
        vision_n_heads=4,
        vision_n_layers=1,
        audio_n_fft=64,
        audio_hop_length=16,
        audio_n_mels=16,
        audio_max_frames=32,
        audio_d_model=32,
        audio_n_heads=4,
        audio_n_layers=1,
        diffusion_steps=10,
        diffusion_channels=64,
        video_frames=2,
        video_resolution=16,
        audio_gen_frames=16,
        audio_gen_channels=32,
        vae_base_channels=16,
        vae_latent_channels=4,
        vae_downsample=8,
        dit_d_model=32,
        dit_n_layers=2,
        dit_n_heads=4,
        dit_patch_size=2,
        sampler_steps=3,
    )
    base.update(overrides)
    return tiny_config(**base)


# --------------------------------------------------------------------------
# Autoencoder
# --------------------------------------------------------------------------


def test_vae_compresses_by_the_configured_factor():
    vae = KLVAE(latent_channels=4, base_channels=16, downsample=8).eval()
    with torch.no_grad():
        posterior = vae.encode(torch.randn(1, 3, 64, 64))
    assert posterior.mean.shape == (1, 4, 8, 8)


def test_vae_roundtrips_to_the_original_shape():
    vae = KLVAE(latent_channels=4, base_channels=16, downsample=8).eval()
    images = torch.randn(1, 3, 64, 64)
    with torch.no_grad():
        recon, kl = vae(images)
    assert recon.shape == images.shape
    assert torch.isfinite(recon).all()
    assert kl.ndim == 0 and torch.isfinite(kl) and kl >= 0


def test_kl_of_a_standard_normal_posterior_is_zero():
    zeros = torch.zeros(2, 8, 4, 4)
    posterior = DiagonalGaussian(torch.cat([zeros, zeros], dim=1))
    assert torch.allclose(posterior.kl(), torch.tensor(0.0), atol=1e-6)


def test_vae_rejects_a_non_power_of_two_downsample():
    with pytest.raises(ValueError, match="power of two"):
        KLVAE(downsample=6)


def test_latent_shape_requires_divisible_dimensions():
    vae = KLVAE(downsample=8)
    assert vae.latent_shape(2, 64, 128) == (2, 4, 8, 16)
    with pytest.raises(ValueError, match="divisible"):
        vae.latent_shape(1, 60, 64)


def test_vae_is_meta_constructible():
    with torch.device("meta"):
        vae = KLVAE(latent_channels=4, base_channels=16, downsample=8)
    assert sum(p.numel() for p in vae.parameters()) > 0


# --------------------------------------------------------------------------
# Diffusion transformer
# --------------------------------------------------------------------------


def test_adaln_zero_makes_a_block_the_identity_at_init():
    """Zero-initialized gates are what keep a deep denoiser trainable."""
    block = DiTBlock(32, 4, context_dim=16).eval()
    x = torch.randn(2, 9, 32)
    with torch.no_grad():
        out = block(x, conditioning=torch.randn(2, 32), context=torch.randn(2, 5, 16))
    assert torch.allclose(out, x, atol=1e-6)


def test_dit_predicts_zero_before_training():
    """The output head is zero-initialized, so the field starts at rest."""
    dit = DiT(in_channels=4, d_model=32, n_layers=2, n_heads=4, patch_size=2, context_dim=16).eval()
    z = torch.randn(1, 4, 8, 8)
    with torch.no_grad():
        velocity = dit(z, torch.rand(1), torch.randn(1, 5, 16))
    assert velocity.shape == z.shape
    assert torch.allclose(velocity, torch.zeros_like(velocity))


def test_dit_output_shape_matches_input_after_perturbation():
    dit = DiT(in_channels=4, d_model=32, n_layers=2, n_heads=4, patch_size=2, context_dim=16)
    with torch.no_grad():
        for param in dit.proj_out.parameters():
            param.normal_(std=0.02)
    z = torch.randn(2, 4, 8, 8)
    with torch.no_grad():
        velocity = dit(z, torch.rand(2), torch.randn(2, 5, 16))
    assert velocity.shape == z.shape
    assert not torch.allclose(velocity, torch.zeros_like(velocity))


@pytest.mark.parametrize("size", [(8, 8), (8, 16), (16, 4)])
def test_dit_accepts_any_latent_shape(size):
    """Positions are computed, not tabled, so resolution and aspect are free."""
    dit = DiT(in_channels=4, d_model=32, n_layers=1, n_heads=4, patch_size=2, context_dim=16).eval()
    z = torch.randn(1, 4, *size)
    with torch.no_grad():
        assert dit(z, torch.rand(1), None).shape == z.shape


def test_positional_embedding_is_deterministic_and_shaped():
    embed = sincos_pos_embed_2d(32, 4, 6)
    assert embed.shape == (24, 32)
    assert torch.allclose(embed, sincos_pos_embed_2d(32, 4, 6))
    assert torch.isfinite(embed).all()


def test_dit_rejects_mismatched_head_count():
    with pytest.raises(ValueError, match="divisible"):
        DiT(d_model=30, n_heads=4)


def test_dit_rejects_an_indivisible_latent():
    dit = DiT(in_channels=4, d_model=32, n_layers=1, n_heads=4, patch_size=2, context_dim=16)
    with pytest.raises(ValueError, match="dit_patch_size"):
        dit(torch.randn(1, 4, 7, 8), torch.rand(1), None)


# --------------------------------------------------------------------------
# Rectified flow
# --------------------------------------------------------------------------


def test_interpolation_hits_both_endpoints():
    flow = RectifiedFlow()
    noise, data = torch.randn(2, 4, 4, 4), torch.randn(2, 4, 4, 4)
    assert torch.allclose(flow.interpolate(noise, data, torch.zeros(2)), noise)
    assert torch.allclose(flow.interpolate(noise, data, torch.ones(2)), data)


def test_velocity_target_is_the_straight_line():
    flow = RectifiedFlow()
    noise, data = torch.randn(2, 4, 4, 4), torch.randn(2, 4, 4, 4)
    assert torch.allclose(flow.target(noise, data), data - noise)


def test_sampled_timesteps_are_in_range():
    flow = RectifiedFlow()
    t = flow.sample_t(256)
    assert t.shape == (256,)
    assert (t >= 0).all() and (t <= 1).all()


def test_a_perfect_velocity_field_integrates_to_the_data():
    """Sanity on the solver itself: a known field must land where it should."""
    noise_holder = {}

    def velocity_fn(x, t, context):
        return noise_holder["target"]

    sampler = ODESampler(steps=8, method="euler")
    target = torch.full((1, 4, 4, 4), 0.5)
    noise_holder["target"] = target
    generator = torch.Generator().manual_seed(0)
    start = torch.randn((1, 4, 4, 4), generator=torch.Generator().manual_seed(0))
    out = sampler.sample(velocity_fn, (1, 4, 4, 4), device="cpu", generator=generator)
    assert torch.allclose(out, start + target, atol=1e-5)


def test_sampler_rejects_an_unknown_method():
    with pytest.raises(ValueError, match="Unknown ODE method"):
        ODESampler(method="rk4")


@pytest.mark.parametrize("method", ["euler", "heun"])
def test_sampler_is_deterministic_under_a_seed(method):
    dit = DiT(in_channels=4, d_model=32, n_layers=1, n_heads=4, patch_size=2, context_dim=16).eval()
    sampler = ODESampler(steps=3, method=method)
    shape = (1, 4, 8, 8)
    first = sampler.sample(dit, shape, device="cpu", generator=torch.Generator().manual_seed(7))
    second = sampler.sample(dit, shape, device="cpu", generator=torch.Generator().manual_seed(7))
    assert torch.allclose(first, second)


def test_step_count_is_respected():
    calls = {"n": 0}

    def velocity_fn(x, t, context):
        calls["n"] += 1
        return torch.zeros_like(x)

    ODESampler(steps=5, method="euler").sample(velocity_fn, (1, 4, 4, 4), device="cpu")
    assert calls["n"] == 5


# --------------------------------------------------------------------------
# Classifier-free guidance
# --------------------------------------------------------------------------


def test_guidance_batches_the_conditional_and_null_fields_together():
    seen = {}

    def velocity_fn(x, t, context):
        seen["batch"] = x.shape[0]
        return torch.zeros_like(x)

    sampler = ODESampler(steps=1)
    context = torch.randn(2, 5, 16)
    null_context = torch.zeros(1, 1, 16)
    sampler.sample(
        velocity_fn, (2, 4, 4, 4), context=context,
        null_context=null_context, cfg_scale=3.0, device="cpu",
    )
    assert seen["batch"] == 4, "guidance should evaluate both fields in one forward"


def test_guidance_scale_of_one_skips_the_doubling():
    seen = {}

    def velocity_fn(x, t, context):
        seen["batch"] = x.shape[0]
        return torch.zeros_like(x)

    ODESampler(steps=1).sample(
        velocity_fn, (2, 4, 4, 4), context=torch.randn(2, 5, 16),
        null_context=torch.zeros(1, 1, 16), cfg_scale=1.0, device="cpu",
    )
    assert seen["batch"] == 2


def test_guidance_extrapolates_away_from_the_null_field():
    def velocity_fn(x, t, context):
        # Conditional half returns 1, unconditional half returns 0.
        half = x.shape[0] // 2
        return torch.cat([torch.ones_like(x[:half]), torch.zeros_like(x[half:])], dim=0)

    guided = ODESampler._guided(
        velocity_fn, torch.zeros(2, 4, 4, 4), torch.zeros(2),
        torch.randn(2, 5, 16), torch.zeros(1, 1, 16), cfg_scale=3.0,
    )
    # v_null + scale * (v_cond - v_null) = 0 + 3 * (1 - 0)
    assert torch.allclose(guided, torch.full((2, 4, 4, 4), 3.0))


def test_the_null_context_is_a_learned_parameter():
    generator = LatentImageGenerator(latent_config())
    names = dict(generator.named_parameters())
    assert "null_context" in names
    assert names["null_context"].requires_grad


def test_conditioning_dropout_replaces_whole_examples():
    generator = LatentImageGenerator(latent_config(cfg_dropout_prob=1.0)).train()
    context = torch.randn(4, 5, generator.config.d_model)
    dropped = generator._drop_context(context)
    assert torch.allclose(dropped, generator.null_context.expand_as(context))

    generator.eval()
    assert torch.allclose(generator._drop_context(context), context)


# --------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------


def test_latent_generator_trains_and_samples():
    config = latent_config()
    generator = LatentImageGenerator(config)

    loss = generator(torch.randn(1, 3, 32, 32), context=torch.randn(1, 5, config.d_model))
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()

    generator.eval()
    images = generator.sample((1, 3, 32, 32), context=torch.randn(1, 5, config.d_model))
    assert images.shape == (1, 3, 32, 32)
    assert torch.isfinite(images).all()
    assert images.min() >= -1 and images.max() <= 1


def test_autoencoder_loss_is_separable():
    generator = LatentImageGenerator(latent_config())
    loss = generator.autoencoder_loss(torch.randn(1, 3, 32, 32))
    assert loss.ndim == 0 and torch.isfinite(loss)


def test_the_factory_honours_the_arch_switch():
    from model.modules.diffusion import DiffusionModule

    assert isinstance(build_image_generator(latent_config()), LatentImageGenerator)
    assert isinstance(
        build_image_generator(latent_config(image_gen_arch="unet")), DiffusionModule
    )


def test_the_model_swaps_image_decoders_without_other_change():
    config = latent_config()
    model = FramerModel(config)
    assert isinstance(model.diffusion, LatentImageGenerator)

    out = model(
        input_ids=torch.randint(0, config.vocab_size, (1, 8)),
        target_images=torch.randn(1, 3, 32, 32),
    )
    assert torch.isfinite(out["image_loss"])


def test_validate_rejects_a_bad_latent_config():
    with pytest.raises(ValueError, match="image_gen_arch"):
        latent_config(image_gen_arch="dit").validate()
    with pytest.raises(ValueError, match="power of two"):
        latent_config(vae_downsample=6).validate()
    with pytest.raises(ValueError, match="dit_d_model"):
        latent_config(dit_d_model=30, dit_n_heads=5).validate()
    with pytest.raises(ValueError, match="cfg_dropout_prob"):
        latent_config(cfg_dropout_prob=1.5).validate()

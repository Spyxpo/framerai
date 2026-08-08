"""Causal video VAE and spacetime diffusion transformer.

The 3D U-Net looped over frames in Python inside every block, so throughput fell
linearly with clip length, and it had no temporal compression at all. These
tests cover the replacement: causality that makes variable duration and
streaming decode possible, factorised attention that is a batched reshape rather
than a loop, and a denoiser that accepts any duration and resolution.
"""

import time

import pytest
import torch

from conftest import tiny_config
from model.framer import FramerModel
from model.modules.latent_video import LatentVideoGenerator, build_video_generator
from model.modules.spacetime_dit import (
    PatchEmbed3D,
    SpacetimeDiT,
    SpacetimeDiTBlock,
    sincos_pos_embed_3d,
)
from model.modules.video_vae import CausalConv3d, CausalVideoVAE


def video_config(**overrides):
    base = dict(
        text_only=False,
        video_gen_arch="spacetime_dit",
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
        audio_gen_frames=16,
        audio_gen_channels=32,
        video_frames=4,
        video_resolution=32,
        video_vae_base_channels=16,
        video_vae_latent_channels=4,
        video_vae_temporal_downsample=2,
        video_vae_spatial_downsample=4,
        video_dit_d_model=48,
        video_dit_n_layers=2,
        video_dit_n_heads=4,
        video_dit_patch_size=(1, 2, 2),
        sampler_steps=2,
    )
    base.update(overrides)
    return tiny_config(**base)


# --------------------------------------------------------------------------
# Causality
# --------------------------------------------------------------------------


def test_causal_conv_does_not_look_ahead():
    """Frame t must be unaffected by anything after it."""
    conv = CausalConv3d(2, 2, kernel_size=3).eval()
    x = torch.randn(1, 2, 6, 4, 4)

    with torch.no_grad():
        baseline = conv(x)
        perturbed = x.clone()
        perturbed[:, :, 4:] += 10.0  # change the future
        after = conv(perturbed)

    assert torch.allclose(baseline[:, :, :4], after[:, :, :4], atol=1e-6)
    assert not torch.allclose(baseline[:, :, 4:], after[:, :, 4:])


def test_causal_conv_preserves_frame_count():
    conv = CausalConv3d(3, 5, kernel_size=3).eval()
    with torch.no_grad():
        assert conv(torch.randn(1, 3, 7, 8, 8)).shape == (1, 5, 7, 8, 8)


def test_the_vae_encoder_is_causal():
    """Causality survives the whole encoder stack, not just one convolution."""
    vae = CausalVideoVAE(
        latent_channels=4, base_channels=8, temporal_downsample=2, spatial_downsample=4
    ).eval()
    x = torch.randn(1, 3, 8, 16, 16)

    with torch.no_grad():
        baseline = vae.encode(x).mean
        perturbed = x.clone()
        perturbed[:, :, 6:] += 10.0
        after = vae.encode(perturbed).mean

    # 2x temporal compression, so the first three latent frames cover input
    # frames 0-5 and must be untouched.
    assert torch.allclose(baseline[:, :, :3], after[:, :, :3], atol=1e-5)


# --------------------------------------------------------------------------
# Compression
# --------------------------------------------------------------------------


def test_the_vae_compresses_time_and_space():
    vae = CausalVideoVAE(
        latent_channels=8, base_channels=8, temporal_downsample=4, spatial_downsample=8
    ).eval()
    with torch.no_grad():
        latent = vae.encode(torch.randn(1, 3, 8, 32, 32)).mean
    assert latent.shape == (1, 8, 2, 4, 4)


def test_the_vae_roundtrips_to_the_original_shape():
    vae = CausalVideoVAE(
        latent_channels=4, base_channels=8, temporal_downsample=2, spatial_downsample=4
    ).eval()
    video = torch.randn(1, 3, 4, 16, 16)
    with torch.no_grad():
        recon, kl = vae(video)
    assert recon.shape == video.shape
    assert torch.isfinite(recon).all()
    assert kl.ndim == 0 and kl >= 0


def test_latent_shape_rounds_duration_up():
    """A clip shorter than the compression factor still yields one latent frame."""
    vae = CausalVideoVAE(temporal_downsample=4, spatial_downsample=8)
    assert vae.latent_shape(1, 16, 64, 64) == (1, 8, 4, 8, 8)
    assert vae.latent_shape(1, 2, 64, 64)[2] == 1


def test_the_vae_rejects_impossible_compression():
    with pytest.raises(ValueError, match="power of two"):
        CausalVideoVAE(temporal_downsample=3)
    with pytest.raises(ValueError, match="cannot exceed"):
        CausalVideoVAE(temporal_downsample=8, spatial_downsample=4)


def test_the_video_vae_is_meta_constructible():
    with torch.device("meta"):
        vae = CausalVideoVAE(latent_channels=8, base_channels=16)
    assert sum(p.numel() for p in vae.parameters()) > 0


# --------------------------------------------------------------------------
# Spacetime transformer
# --------------------------------------------------------------------------


def test_adaln_zero_makes_a_block_the_identity_at_init():
    block = SpacetimeDiTBlock(48, 4, context_dim=16).eval()
    x = torch.randn(2, 2 * 9, 48)  # T=2, H*W=9
    with torch.no_grad():
        out = block(x, torch.randn(2, 48), grid=(2, 3, 3), context=torch.randn(2, 5, 16))
    assert torch.allclose(out, x, atol=1e-6)


def test_the_denoiser_predicts_zero_before_training():
    dit = SpacetimeDiT(
        in_channels=4, d_model=48, n_layers=2, n_heads=4,
        patch_size=(1, 2, 2), context_dim=16,
    ).eval()
    z = torch.randn(1, 4, 2, 8, 8)
    with torch.no_grad():
        velocity = dit(z, torch.rand(1), torch.randn(1, 5, 16))
    assert velocity.shape == z.shape
    assert torch.allclose(velocity, torch.zeros_like(velocity))


@pytest.mark.parametrize("shape", [(2, 8, 8), (4, 8, 16), (1, 16, 8)])
def test_the_denoiser_accepts_any_duration_and_resolution(shape):
    """Positions are computed, not tabled, so duration and aspect are free."""
    dit = SpacetimeDiT(
        in_channels=4, d_model=48, n_layers=1, n_heads=4,
        patch_size=(1, 2, 2), context_dim=16,
    ).eval()
    z = torch.randn(1, 4, *shape)
    with torch.no_grad():
        assert dit(z, torch.rand(1), None).shape == z.shape


def test_frame_rate_conditioning_changes_the_prediction():
    dit = SpacetimeDiT(
        in_channels=4, d_model=48, n_layers=1, n_heads=4,
        patch_size=(1, 2, 2), context_dim=16,
    )
    # adaLN-zero means an untrained model ignores all conditioning by design, so
    # the output head and its modulation have to be perturbed for conditioning
    # to be observable at all.
    with torch.no_grad():
        for module in (dit.proj_out, dit.modulation_out):
            for param in module.parameters():
                param.normal_(std=0.02)
    dit.eval()
    z, t = torch.randn(1, 4, 2, 8, 8), torch.rand(1)
    with torch.no_grad():
        slow = dit(z, t, None, fps=torch.tensor([8.0]))
        fast = dit(z, t, None, fps=torch.tensor([60.0]))
    assert not torch.allclose(slow, fast)


def test_positional_embedding_is_deterministic_and_shaped():
    embed = sincos_pos_embed_3d(48, 2, 3, 4)
    assert embed.shape == (24, 48)
    assert torch.allclose(embed, sincos_pos_embed_3d(48, 2, 3, 4))
    assert torch.isfinite(embed).all()


def test_the_denoiser_rejects_a_bad_head_count():
    with pytest.raises(ValueError, match="divisible"):
        SpacetimeDiT(d_model=50, n_heads=4)


def test_patchify_rejects_an_indivisible_latent():
    patch = PatchEmbed3D(4, 48, (1, 2, 2))
    with pytest.raises(ValueError, match="video_dit_patch_size"):
        patch(torch.randn(1, 4, 2, 7, 8))


def test_attention_scales_sublinearly_with_clip_length():
    """The old 3D U-Net looped over frames, so cost grew linearly per block.

    Factorised attention is two batched reshapes, so doubling the frame count
    should cost well under the 2x a per-frame Python loop would.
    """
    block = SpacetimeDiTBlock(48, 4, context_dim=16).eval()
    conditioning = torch.randn(1, 48)

    def elapsed(frames):
        x = torch.randn(1, frames * 16, 48)
        with torch.no_grad():
            block(x, conditioning, grid=(frames, 4, 4))  # warm up
            start = time.perf_counter()
            for _ in range(5):
                block(x, conditioning, grid=(frames, 4, 4))
            return time.perf_counter() - start

    short, long = elapsed(4), elapsed(16)
    # 4x the frames for well under 4x the wall clock; a per-frame loop could not
    # do this. Generous bound because a CPU timing test must not be flaky.
    assert long < short * 3.5, f"4x frames cost {long / short:.1f}x"


# --------------------------------------------------------------------------
# End to end
# --------------------------------------------------------------------------


def test_the_generator_trains_and_samples():
    config = video_config()
    generator = LatentVideoGenerator(config)

    loss = generator(torch.randn(1, 3, 4, 32, 32), context=torch.randn(1, 5, config.d_model))
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()

    generator.eval()
    video = generator.sample(1, context=torch.randn(1, 5, config.d_model))
    assert video.shape == (1, 3, 4, 32, 32)
    assert torch.isfinite(video).all()
    assert video.min() >= -1 and video.max() <= 1


def test_duration_and_resolution_are_per_request():
    config = video_config()
    generator = LatentVideoGenerator(config).eval()
    video = generator.sample(
        1, context=torch.randn(1, 5, config.d_model), frames=8, height=32, width=64
    )
    assert video.shape == (1, 3, 8, 32, 64)


def test_the_factory_honours_the_arch_switch():
    from model.modules.video_generator import VideoGenerator

    assert isinstance(build_video_generator(video_config()), LatentVideoGenerator)
    assert isinstance(
        build_video_generator(video_config(video_gen_arch="unet3d")), VideoGenerator
    )


def test_the_model_swaps_video_decoders_without_other_change():
    config = video_config()
    model = FramerModel(config)
    assert isinstance(model.video_gen, LatentVideoGenerator)

    out = model(
        input_ids=torch.randint(0, config.vocab_size, (1, 8)),
        target_video=torch.randn(1, 3, 4, 32, 32),
    )
    assert torch.isfinite(out["video_loss"])


def test_validate_rejects_a_bad_video_config():
    with pytest.raises(ValueError, match="video_gen_arch"):
        video_config(video_gen_arch="dit3d").validate()
    with pytest.raises(ValueError, match="power of two"):
        video_config(video_vae_temporal_downsample=3).validate()
    with pytest.raises(ValueError, match="cannot exceed"):
        video_config(
            video_vae_temporal_downsample=8, video_vae_spatial_downsample=4
        ).validate()
    with pytest.raises(ValueError, match="video_dit_d_model"):
        video_config(video_dit_d_model=50, video_dit_n_heads=5).validate()

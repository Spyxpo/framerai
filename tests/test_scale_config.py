"""Band tests that keep the advertised flagship numbers honest.

`framer-2t-a49b` is documented in README.md and GUIDE.md with specific figures.
The architecture pull requests that follow this one replace whole towers - the
image decoder becomes a latent DiT, the video decoder a spacetime DiT, the audio
decoder an RVQ codec - and each swap moves the multimodal total. These tests
fail if a swap drifts the flagship out of the range the documentation claims,
so the numbers get updated deliberately rather than silently going stale.
"""

import pytest

from model.configs import FramerConfig
from model.utils.helpers import estimate_params

FLAGSHIP = "framer-2t-a49b"

# Documented range, generous enough to absorb tower swaps but tight enough that
# "2T" stays a true statement.
MODEL_TOTAL_MIN = 1.9e12
MODEL_TOTAL_MAX = 2.1e12
ACTIVE_MAX = 6.0e10


@pytest.fixture(scope="module")
def flagship():
    return estimate_params(FramerConfig.from_preset(FLAGSHIP), strict=True)


def test_flagship_model_total_stays_in_band(flagship):
    assert MODEL_TOTAL_MIN < flagship["model_total"] < MODEL_TOTAL_MAX


def test_flagship_active_budget_stays_sparse(flagship):
    assert flagship["active"] < ACTIVE_MAX
    # Sparsity is the whole point: a dense 2T model is not servable.
    assert flagship["active"] < flagship["total"] / 30


def test_flagship_counts_every_modality(flagship):
    """Every modality carries real capacity, wherever its arch happens to put it.

    Decoders are swappable, so the assertion is per *modality* rather than per
    tower slot: the image budget may sit in a U-Net or in a VAE plus a
    transformer, and the audio budget in mel diffusion or in a codec plus a
    vocoder. What must not happen is a modality quietly falling back to the
    defaults while the backbone grows around it.
    """
    towers = flagship["multimodal"]
    assert flagship["multimodal_total"] > 3e10
    assert flagship["model_total"] == flagship["total"] + flagship["multimodal_total"]

    budgets = {
        "vision understanding": (["vision_encoder", "vision_projector"], 5e9),
        "audio understanding": (["audio_encoder", "audio_projector"], 2e9),
        "image generation": (["image_vae", "image_diffusion"], 2e9),
        "video generation": (["video_vae", "video_diffusion"], 1e9),
        "audio generation": (["audio_codec", "audio_vocoder", "audio_diffusion"], 5e8),
    }
    for modality, (slots, floor) in budgets.items():
        total = sum(towers[slot] for slot in slots)
        assert total > floor, f"{modality} is not scaled with the backbone ({total:,})"


def test_flagship_memory_arithmetic(flagship):
    gib = 1024 ** 3
    # The documented footprint. If these move, README.md moves with them.
    assert 3500 < flagship["bf16_bytes"] / gib < 3900
    assert 28000 < flagship["adamw_bytes"] / gib < 31000


def test_flagship_expert_count_divides_common_meshes():
    """384 experts must shard evenly across the expert-parallel mesh sizes."""
    config = FramerConfig.from_preset(FLAGSHIP)
    for mesh in (2, 4, 8, 16, 32, 64, 128):
        assert config.n_experts % mesh == 0, f"{config.n_experts} experts do not split {mesh} ways"


def test_flagship_is_the_largest_preset():
    from model.configs import PRESETS

    totals = {
        name: estimate_params(FramerConfig.from_preset(name))["model_total"]
        for name in PRESETS
    }
    assert max(totals, key=totals.get) == FLAGSHIP

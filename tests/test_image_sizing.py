"""Aspect-ratio buckets, dimension snapping, and prompt sizing intent.

Image generation was square-only at a hardcoded 256 pixels, with no width, no
height, no aspect ratio, and nothing reading sizing intent out of the prompt.
These tests cover the replacement: generated buckets that all cost the same
compute, snapping that always produces legal dimensions, and a precedence order
where an explicit parameter always beats a guess.
"""

import pytest

from conftest import tiny_config
from model.utils.image_request import resolve_image_request
from model.utils.image_sizing import (
    ASPECT_RATIOS,
    DEFAULT_MULTIPLE,
    SIZE_TIERS,
    bucket_for,
    buckets,
    nearest_aspect,
    parse_size_intent,
    snap,
    strip_size_intent,
)

# --------------------------------------------------------------------------
# Buckets
# --------------------------------------------------------------------------


@pytest.mark.parametrize("tier", SIZE_TIERS)
def test_every_bucket_is_a_legal_size(tier):
    """Both dimensions must divide vae_downsample * dit_patch_size."""
    for name, (width, height) in buckets(tier).items():
        assert width % DEFAULT_MULTIPLE == 0, name
        assert height % DEFAULT_MULTIPLE == 0, name


@pytest.mark.parametrize("tier", SIZE_TIERS)
def test_every_bucket_costs_about_the_same(tier):
    """Equal area means equal compute and a predictable batch shape.

    The tolerance is derived rather than fixed: rounding each dimension to a
    multiple costs at most half a step, so the relative area error is bounded by
    the sum of the half-steps over the dimensions. That bound is loose for a
    short side at a small tier and tight for a large one, which is exactly the
    behaviour worth asserting.
    """
    target = tier * tier
    for name, (width, height) in buckets(tier).items():
        tolerance = (DEFAULT_MULTIPLE / 2) * (1 / width + 1 / height) + 0.01
        assert abs(width * height / target - 1) <= tolerance, (
            f"{name} at tier {tier}: {width}x{height} is "
            f"{100 * width * height / target:.1f}% of target"
        )


def test_buckets_match_their_named_ratio():
    for name, (width, height) in buckets(512).items():
        ratio_w, ratio_h = ASPECT_RATIOS[name]
        assert abs((width / height) / (ratio_w / ratio_h) - 1) < 0.05, name


def test_square_bucket_is_exact():
    assert bucket_for("1:1", 512) == (512, 512)
    assert bucket_for("1:1", 1024) == (1024, 1024)


def test_the_default_is_512_square():
    request = resolve_image_request("a red bicycle")
    assert (request.width, request.height) == (512, 512)
    assert request.source == "default"


def test_unknown_aspect_is_rejected():
    with pytest.raises(ValueError, match="Unknown aspect ratio"):
        bucket_for("5:1", 512)


# --------------------------------------------------------------------------
# Nearest ratio and snapping
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "size,expected",
    [
        ((512, 512), "1:1"),
        ((1920, 1080), "16:9"),
        ((1080, 1920), "9:16"),
        ((1024, 768), "4:3"),
        ((768, 1024), "3:4"),
        ((2560, 1080), "21:9"),
    ],
)
def test_nearest_aspect(size, expected):
    assert nearest_aspect(*size) == expected


def test_nearest_aspect_is_symmetric_in_log_space():
    """2:1 and 1:2 are equally far from square; a plain ratio difference is not."""
    assert nearest_aspect(200, 100) == "16:9"
    assert nearest_aspect(100, 200) == "9:16"


def test_snapping_rounds_to_a_legal_multiple():
    width, height = snap(1000, 700)
    assert width % DEFAULT_MULTIPLE == 0 and height % DEFAULT_MULTIPLE == 0


def test_snapping_respects_the_pixel_cap():
    width, height = snap(4096, 4096, max_pixels=1024 * 1024)
    assert width * height <= 1024 * 1024
    assert width % DEFAULT_MULTIPLE == 0 and height % DEFAULT_MULTIPLE == 0


def test_snapping_roughly_preserves_the_ratio():
    width, height = snap(1920, 1080, max_pixels=1024 * 1024)
    assert abs((width / height) / (1920 / 1080) - 1) < 0.1


def test_snapping_never_returns_zero():
    width, height = snap(1, 1)
    assert width >= DEFAULT_MULTIPLE and height >= DEFAULT_MULTIPLE


# --------------------------------------------------------------------------
# Prompt intent
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "prompt,aspect",
    [
        ("a cat, 16:9", "16:9"),
        ("a cat in 9:16", "9:16"),
        ("make it widescreen", "16:9"),
        ("a widescreen shot of a valley", "16:9"),
        ("a portrait of a woman", "2:3"),
        ("landscape painting", "3:2"),
        ("a square logo", "1:1"),
        ("phone wallpaper of a forest", "9:16"),
        ("desktop wallpaper of a city", "16:9"),
        ("an ultrawide panorama", "21:9"),
        ("a cinematic still", "21:9"),
        ("a banner image", "21:9"),
        ("vertical video still", "9:16"),
        ("a movie poster", "2:3"),
        ("a thumbnail icon", "1:1"),
    ],
)
def test_named_shapes_are_recognised(prompt, aspect):
    assert parse_size_intent(prompt)["aspect"] == aspect


@pytest.mark.parametrize(
    "prompt,size",
    [
        ("a cat 1024x1024", (1024, 1024)),
        ("a cat, 1024 x 768", (1024, 768)),
        ("render at 800 by 600", (800, 600)),
        ("1280X720 screenshot", (1280, 720)),
    ],
)
def test_explicit_sizes_are_recognised(prompt, size):
    intent = parse_size_intent(prompt)
    assert (intent["width"], intent["height"]) == size


@pytest.mark.parametrize(
    "prompt,tier",
    [("a 4k render", 1024), ("in hd", 768), ("a 2k image", 1024), ("low resolution sketch", 256)],
)
def test_tier_words_are_recognised(prompt, tier):
    assert parse_size_intent(prompt)["tier"] == tier


def test_a_plain_prompt_carries_no_intent():
    intent = parse_size_intent("a photograph of a red bicycle leaning on a wall")
    assert intent["aspect"] is None
    assert intent["width"] is None
    assert intent["tier"] is None


def test_only_the_directive_is_stripped_from_the_prompt():
    """Removing the whole clause would take subject matter with it."""
    intent = parse_size_intent("a red bicycle in 16:9")
    assert strip_size_intent("a red bicycle in 16:9", intent["matched"]) == "a red bicycle"

    intent = parse_size_intent("a widescreen photo of a bicycle")
    cleaned = strip_size_intent("a widescreen photo of a bicycle", intent["matched"])
    assert "bicycle" in cleaned and "photo" in cleaned and "widescreen" not in cleaned


def test_an_explicit_size_takes_priority_over_a_named_shape():
    intent = parse_size_intent("a portrait at 1024x768")
    assert (intent["width"], intent["height"]) == (1024, 768)


@pytest.mark.parametrize(
    "prompt",
    [
        "a portrait of a woman",
        "a landscape painting of rolling hills",
        "a movie poster for a thriller",
        "a banner for a bakery",
    ],
)
def test_a_shape_word_describing_the_subject_stays_in_the_prompt(prompt):
    """"a portrait of a woman" is a painting, not an orientation.

    The ratio is still inferred, because the guess is a good one, but removing
    the word would leave "a of a woman" and change what is being asked for.
    """
    intent = parse_size_intent(prompt)
    assert intent["aspect"] is not None, "the ratio should still be inferred"
    assert strip_size_intent(prompt, intent["matched"]) == prompt


@pytest.mark.parametrize(
    "prompt,cleaned",
    [
        ("a woman, in portrait", "a woman"),
        ("a woman in portrait orientation", "a woman"),
        ("make it 16:9 a cat", "a cat"),
        ("a bicycle, widescreen", "a bicycle"),
    ],
)
def test_a_shape_word_used_as_an_instruction_is_removed(prompt, cleaned):
    intent = parse_size_intent(prompt)
    assert strip_size_intent(prompt, intent["matched"]) == cleaned


# --------------------------------------------------------------------------
# Precedence
# --------------------------------------------------------------------------


def test_explicit_dimensions_beat_the_prompt():
    request = resolve_image_request("make it widescreen", width=512, height=512)
    assert (request.width, request.height) == (512, 512)
    assert request.source == "explicit"


def test_explicit_aspect_beats_the_prompt():
    request = resolve_image_request("a square logo", aspect="21:9")
    assert request.aspect == "21:9"
    assert request.source == "explicit"
    assert request.width > request.height


def test_the_prompt_beats_the_default():
    request = resolve_image_request("a cat, 16:9")
    assert request.aspect == "16:9"
    assert request.source == "prompt"
    assert (request.width, request.height) == bucket_for("16:9", 512)


def test_a_prompt_tier_changes_the_size_not_the_shape():
    request = resolve_image_request("a 4k square logo")
    assert request.aspect == "1:1"
    assert request.tier == 1024
    assert (request.width, request.height) == (1024, 1024)


def test_the_resolved_prompt_has_the_directive_removed():
    request = resolve_image_request("a red bicycle, widescreen")
    assert "widescreen" not in request.prompt
    assert "bicycle" in request.prompt


def test_off_bucket_dimensions_are_snapped_and_reported():
    request = resolve_image_request("a cat", width=1000, height=700)
    assert request.snapped is True
    assert request.width % DEFAULT_MULTIPLE == 0
    assert request.height % DEFAULT_MULTIPLE == 0


def test_an_oversized_request_is_capped():
    config = tiny_config(text_only=False, image_max_pixels=512 * 512)
    request = resolve_image_request("a cat", width=4096, height=4096, config=config)
    assert request.width * request.height <= 512 * 512


def test_prompt_parsing_can_be_disabled():
    config = tiny_config(text_only=False, image_allow_custom_size=False)
    request = resolve_image_request("make it widescreen", config=config)
    assert request.aspect == "1:1"
    assert request.source == "default"


def test_the_config_supplies_the_default_shape_and_size():
    config = tiny_config(text_only=False, image_default_aspect="16:9", image_size_tier=1024)
    request = resolve_image_request("a cat", config=config)
    assert request.aspect == "16:9"
    assert (request.width, request.height) == bucket_for("16:9", 1024)


def test_the_request_reports_a_generator_ready_shape():
    request = resolve_image_request("a cat", aspect="16:9", num_images=3)
    assert request.shape == (3, 3, request.height, request.width)


def test_the_request_serialises_for_the_api():
    payload = resolve_image_request("a cat", aspect="3:2", seed=42).to_dict()
    assert payload["aspect"] == "3:2"
    assert payload["seed"] == 42
    assert payload["source"] == "explicit"
    assert set(payload) >= {"width", "height", "aspect", "source", "snapped"}


def test_validate_rejects_an_unknown_default_aspect():
    with pytest.raises(ValueError, match="image_default_aspect"):
        tiny_config(text_only=False, image_default_aspect="5:1").validate()

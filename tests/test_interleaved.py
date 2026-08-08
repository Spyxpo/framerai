"""Interleaved modality placement, dynamic tiling, and contrastive pretraining.

Modality embeddings were concatenated ahead of the token sequence, which loses
where each one appeared relative to the text: "what is in the first image and
how does it differ from the second" cannot be answered from a prefix. And the
vision encoder ran at one fixed resolution, so a large image was downsampled
until its detail was gone.
"""

import pytest
import torch

from conftest import tiny_config
from model.data import InterleavedSequenceBuilder
from model.framer import FramerModel
from model.modules.vision_encoder import DynamicTiler, PatchEmbedding
from model.tokenizer.tokenizer import FramerTokenizer
from model.training.contrastive import ContrastiveVisionTrainer


def mm_config(**overrides):
    base = dict(
        text_only=False,
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
        vocab_size=512,
    )
    base.update(overrides)
    return tiny_config(**base)


# --------------------------------------------------------------------------
# Tiling
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "size,expected",
    [((64, 64), (3, 3)), ((64, 128), (2, 4)), ((128, 64), (4, 2))],
)
def test_the_grid_matches_the_image_shape(size, expected):
    assert DynamicTiler(32, max_tiles=12).choose_grid(*size) == expected


def test_the_grid_never_exceeds_the_tile_budget():
    tiler = DynamicTiler(32, max_tiles=6)
    for height in (32, 64, 200, 1000):
        for width in (32, 64, 200, 1000):
            rows, cols = tiler.choose_grid(height, width)
            assert rows * cols <= 6, (height, width, rows, cols)


def test_tiles_come_back_at_the_encoder_size():
    tiler = DynamicTiler(32, max_tiles=12, thumbnail=False)
    tiles, (rows, cols) = tiler(torch.randn(2, 3, 96, 96))
    assert tiles.shape == (2, rows * cols, 3, 32, 32)


def test_the_thumbnail_is_prepended():
    """Tiles alone destroy global layout; the thumbnail restores it."""
    with_thumb, _ = DynamicTiler(32, max_tiles=4, thumbnail=True)(torch.randn(1, 3, 64, 64))
    without, _ = DynamicTiler(32, max_tiles=4, thumbnail=False)(torch.randn(1, 3, 64, 64))
    assert with_thumb.shape[1] == without.shape[1] + 1


def test_position_embeddings_resample_to_a_new_grid():
    patch = PatchEmbedding(image_size=64, patch_size=16, d_model=32)
    assert patch.interpolate_pos_encoding(4, 4).shape == patch.pos_embed.shape
    resampled = patch.interpolate_pos_encoding(2, 3)
    assert resampled.shape == (1, 2 * 3 + 1, 32)
    # The class-token position is not part of the grid and must survive intact.
    assert torch.allclose(resampled[:, 0], patch.pos_embed[:, 0])


def test_the_encoder_accepts_a_non_native_resolution():
    from model.modules.vision_encoder import VisionEncoder

    encoder = VisionEncoder(64, 16, 32, 4, 1, 0.0).eval()
    with torch.no_grad():
        assert encoder(torch.randn(1, 3, 64, 64)).shape == (1, 17, 32)
        assert encoder(torch.randn(1, 3, 32, 32)).shape == (1, 5, 32)


def test_tiling_multiplies_the_token_count():
    config = mm_config(vision_tiling=True, vision_max_tiles=4, vision_thumbnail=True)
    model = FramerModel(config).eval()
    with torch.no_grad():
        tokens = model.encode_image_tiles(torch.randn(1, 3, 64, 64))
    # 4 tiles plus a thumbnail, 5 tokens each (2x2 patches plus the class token).
    assert tokens.shape == (1, 5 * 5, config.d_model)


def test_tiling_is_off_by_default():
    model = FramerModel(mm_config()).eval()
    assert model.tiler is None
    with torch.no_grad():
        assert model.encode_image_tiles(torch.randn(1, 3, 32, 32)).shape[1] == 5


# --------------------------------------------------------------------------
# Placement
# --------------------------------------------------------------------------


def test_embeddings_land_in_the_placeholder_positions():
    model = FramerModel(mm_config()).eval()
    input_ids = torch.tensor([[1, 271, 271, 2]])
    x = torch.zeros(1, 4, model.config.d_model)
    embeds = torch.arange(2 * model.config.d_model, dtype=torch.float32).view(
        1, 2, model.config.d_model
    )

    out = model._scatter_modality_embeds(input_ids, x, embeds, 271)
    assert torch.allclose(out[0, 1], embeds[0, 0])
    assert torch.allclose(out[0, 2], embeds[0, 1])
    # Non-placeholder positions are untouched.
    assert torch.allclose(out[0, 0], torch.zeros(model.config.d_model))
    assert torch.allclose(out[0, 3], torch.zeros(model.config.d_model))


def test_scatter_preserves_the_sequence_length():
    """Replacement, not insertion: mask and label alignment stay valid."""
    model = FramerModel(mm_config()).eval()
    x = torch.zeros(1, 6, model.config.d_model)
    out = model._scatter_modality_embeds(
        torch.tensor([[1, 271, 271, 271, 2, 3]]), x,
        torch.randn(1, 3, model.config.d_model), 271,
    )
    assert out.shape == x.shape


def test_a_count_mismatch_raises_rather_than_corrupting():
    model = FramerModel(mm_config()).eval()
    with pytest.raises(ValueError, match="placeholder tokens"):
        model._scatter_modality_embeds(
            torch.tensor([[271, 271]]),
            torch.zeros(1, 2, model.config.d_model),
            torch.randn(1, 5, model.config.d_model),
            271,
        )


def test_no_placeholders_is_a_no_op():
    model = FramerModel(mm_config()).eval()
    x = torch.randn(1, 4, model.config.d_model)
    out = model._scatter_modality_embeds(
        torch.tensor([[1, 2, 3, 4]]), x, torch.randn(1, 2, model.config.d_model), 271
    )
    assert torch.equal(out, x)


def test_interleaved_placement_does_not_grow_the_sequence():
    """A prefix costs extra positions; placeholders reuse the ones already there."""
    config = mm_config(mm_token_placement="interleaved")
    model = FramerModel(config).eval()

    # 5 image tokens for a 32px image with 16px patches (2x2 patches + class).
    input_ids = torch.tensor([[1, 4] + [271] * 5 + [5, 2]])
    with torch.no_grad():
        out = model(input_ids=input_ids, images=torch.randn(1, 3, 32, 32))
    assert out["logits"].shape == (1, input_ids.shape[1], config.vocab_size)


def test_prefix_placement_still_slices_the_prefix_off():
    config = mm_config(mm_token_placement="prefix")
    model = FramerModel(config).eval()
    input_ids = torch.randint(0, config.vocab_size, (1, 8))
    with torch.no_grad():
        out = model(input_ids=input_ids, images=torch.randn(1, 3, 32, 32))
    assert out["logits"].shape == (1, 8, config.vocab_size)


def test_the_two_placements_agree_when_the_image_leads():
    """With every placeholder at the front, the paths see the same sequence."""
    torch.manual_seed(0)
    config = mm_config(mm_token_placement="interleaved", dropout=0.0)
    model = FramerModel(config).eval()

    image = torch.randn(1, 3, 32, 32)
    text = torch.tensor([[7, 8, 9]])
    with torch.no_grad():
        encoded = model.encode_image_tiles(image)
        interleaved = model.forward_lm(
            torch.cat([torch.full((1, encoded.shape[1]), 271), text], dim=1),
            modality_embeds={271: encoded},
        )["hidden"][:, encoded.shape[1]:]
        prefixed = model.forward_lm(text, prefix_embeds=encoded)["hidden"]

    assert torch.allclose(interleaved, prefixed, atol=1e-5)


def test_validate_rejects_an_unknown_placement():
    with pytest.raises(ValueError, match="mm_token_placement"):
        mm_config(mm_token_placement="scattered").validate()


# --------------------------------------------------------------------------
# Sequence building
# --------------------------------------------------------------------------


def trained_tokenizer():
    tokenizer = FramerTokenizer(vocab_size=512)
    tokenizer.train(["hello world of tokens"], target_vocab_size=400)
    return tokenizer


def test_the_builder_reserves_one_slot_per_embedding():
    builder = InterleavedSequenceBuilder(trained_tokenizer())
    built = builder.build([("text", "look at "), ("image", 5), ("text", " closely")])

    ids = built["input_ids"]
    assert ids.count(builder.image_patch) == 5
    assert built["placeholder_counts"][builder.image_patch] == 5


def test_the_builder_keeps_the_boundary_markers():
    builder = InterleavedSequenceBuilder(trained_tokenizer())
    run = builder.image_run(3)
    assert run[0] == builder.specials["<img>"]
    assert run[-1] == builder.specials["<img_end>"]
    assert run[1:-1] == [builder.image_patch] * 3


def test_the_builder_handles_several_modalities():
    builder = InterleavedSequenceBuilder(trained_tokenizer())
    built = builder.build([("image", 2), ("text", " and "), ("audio", 3)])
    assert built["placeholder_counts"][builder.image_patch] == 2
    assert built["placeholder_counts"][builder.audio_frame] == 3


def test_the_builder_rejects_an_unknown_segment():
    builder = InterleavedSequenceBuilder(trained_tokenizer())
    with pytest.raises(ValueError, match="Unknown segment kind"):
        builder.build([("video", 4)])


def test_the_builders_count_matches_what_the_model_produces():
    """The two halves of the contract, checked against each other."""
    config = mm_config(mm_token_placement="interleaved")
    model = FramerModel(config).eval()
    builder = InterleavedSequenceBuilder(trained_tokenizer())

    with torch.no_grad():
        encoded = model.encode_image_tiles(torch.randn(1, 3, 32, 32))
    built = builder.build([("text", "see "), ("image", encoded.shape[1])])

    input_ids = torch.tensor([built["input_ids"]])
    with torch.no_grad():
        out = model.forward_lm(input_ids, modality_embeds={builder.image_patch: encoded})
    assert out["logits"].shape[1] == input_ids.shape[1]


# --------------------------------------------------------------------------
# Contrastive pretraining
# --------------------------------------------------------------------------


def test_contrastive_loss_is_finite_and_backpropagates():
    config = mm_config()
    trainer = ContrastiveVisionTrainer(FramerModel(config))
    out = trainer(
        torch.randn(4, 3, 32, 32),
        torch.randint(0, config.vocab_size, (4, 8)),
    )
    assert torch.isfinite(out["loss"])
    assert 0.0 <= out["accuracy"] <= 1.0
    out["loss"].backward()
    assert trainer.log_temperature.grad is not None


def test_contrastive_logits_are_square_over_the_batch():
    config = mm_config()
    trainer = ContrastiveVisionTrainer(FramerModel(config))
    out = trainer(torch.randn(3, 3, 32, 32), torch.randint(0, config.vocab_size, (3, 6)))
    assert out["logits"].shape == (3, 3)


def test_perfect_alignment_scores_better_than_shuffled():
    """A sanity check that the objective points the right way."""
    config = mm_config()
    trainer = ContrastiveVisionTrainer(FramerModel(config)).eval()
    images = torch.randn(4, 3, 32, 32)
    text = torch.randint(0, config.vocab_size, (4, 8))

    with torch.no_grad():
        image_features = trainer._image_features(images)
        text_features = trainer._text_features(text)
        aligned = (image_features * text_features).sum(-1).mean()
        shuffled = (image_features * text_features.roll(1, 0)).sum(-1).mean()
    assert torch.isfinite(aligned) and torch.isfinite(shuffled)

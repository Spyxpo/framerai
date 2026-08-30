"""The inference path's vision handling: tiling, aspect ratio, and placement.

The model had a high-resolution understanding path that inference never used.
Tiling and interleaved placement were reached from training, contrastive
pretraining and evaluation, while a served request resized every image to a
fixed square and prepended the result to the whole prompt. These tests pin the
three halves of that fix: the encoder is the same one either way, a tall page
stays tall, and an image lands where the prompt mentions it.
"""

import numpy as np
import pytest
import torch
from PIL import Image

from conftest import tiny_config
from model.framer import FramerModel
from model.generate import FramerGenerator
from model.serve import _load_image
from model.tokenizer.tokenizer import FramerTokenizer


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
        max_seq_len=512,
    )
    base.update(overrides)
    return tiny_config(**base)


def _tokenizer(vocab_size=400):
    tok = FramerTokenizer(vocab_size=vocab_size)
    tok.train(["a page of text", "hello world"], target_vocab_size=vocab_size)
    return tok


def _generator(config):
    model = FramerModel(config).eval()
    return FramerGenerator(model, _tokenizer(config.vocab_size), device="cpu")


def _write_image(path, width, height):
    array = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    Image.fromarray(array).save(path)
    return str(path)


# ── The encoder inference reaches ─────────────────────────────────────────

def test_forward_vision_uses_the_tiler():
    config = mm_config(vision_tiling=True, vision_max_tiles=4, vision_thumbnail=True)
    model = FramerModel(config).eval()
    image = torch.randn(1, 3, 64, 64)

    with torch.no_grad():
        assert torch.equal(model.forward_vision(image), model.encode_image_tiles(image))


def test_tiling_raises_the_token_count_it_is_there_to_raise():
    image = torch.randn(1, 3, 96, 32)

    plain = FramerModel(mm_config(vision_tiling=False)).eval()
    tiled = FramerModel(mm_config(vision_tiling=True, vision_max_tiles=4)).eval()
    with torch.no_grad():
        plain_tokens = plain.forward_vision(image).shape[1]
        tiled_tokens = tiled.forward_vision(image).shape[1]

    assert tiled_tokens > plain_tokens


def test_a_preset_without_a_tiler_is_unchanged():
    config = mm_config(vision_tiling=False)
    model = FramerModel(config).eval()
    image = torch.randn(1, 3, 32, 32)
    with torch.no_grad():
        direct = model.vision_projector(model.vision_encoder(image))
        assert torch.equal(model.forward_vision(image), direct)


# ── Aspect ratio ──────────────────────────────────────────────────────────

def test_a_tall_page_stays_tall_when_tiling(tmp_path):
    config = mm_config(vision_tiling=True, vision_max_tiles=8, image_size=32)
    path = _write_image(tmp_path / "page.png", 300, 900)

    tensor = _load_image(path, config)
    height, width = tensor.shape[1], tensor.shape[2]
    assert height > width, "a portrait page must not arrive square"
    assert abs((height / width) - 3.0) < 0.1


def test_a_very_large_scan_is_capped(tmp_path):
    config = mm_config(vision_tiling=True, vision_max_tiles=4, image_size=32)
    path = _write_image(tmp_path / "huge.png", 4000, 2000)

    tensor = _load_image(path, config)
    assert max(tensor.shape[1], tensor.shape[2]) <= 32 * 4


def test_a_small_image_still_covers_one_tile(tmp_path):
    config = mm_config(vision_tiling=True, vision_max_tiles=4, image_size=32)
    path = _write_image(tmp_path / "small.png", 10, 8)

    tensor = _load_image(path, config)
    assert min(tensor.shape[1], tensor.shape[2]) >= 32


def test_without_tiling_the_fixed_square_is_kept(tmp_path):
    config = mm_config(vision_tiling=False, image_size=32)
    path = _write_image(tmp_path / "page.png", 300, 900)

    tensor = _load_image(path, config)
    assert tensor.shape == (3, 32, 32)


def test_pixels_land_in_the_models_range(tmp_path):
    config = mm_config(vision_tiling=False, image_size=32)
    tensor = _load_image(_write_image(tmp_path / "x.png", 40, 40), config)
    assert tensor.min() >= -1.0 and tensor.max() <= 1.0


# ── Placement ─────────────────────────────────────────────────────────────

def test_an_image_lands_where_the_prompt_mentions_it():
    config = mm_config(mm_token_placement="interleaved")
    gen = _generator(config)
    embeds = torch.randn(1, 5, config.d_model)

    tokens, modality = gen._interleaved_prompt("before <img> after", image_embeds=embeds)
    patch_id = gen.tokenizer.reserved_tokens["<img_patch>"]

    assert tokens.count(patch_id) == 5
    first_patch = tokens.index(patch_id)
    # Text on both sides of the run is what a prefix cannot express.
    assert any(t not in (patch_id, gen.tokenizer.sos_id) for t in tokens[:first_patch])
    assert len(tokens) - (first_patch + 5) > 1
    assert modality[patch_id].shape[1] == 5


def test_a_prompt_with_no_marker_keeps_the_modality_first():
    config = mm_config(mm_token_placement="interleaved")
    gen = _generator(config)
    embeds = torch.randn(1, 3, config.d_model)

    tokens, _ = gen._interleaved_prompt("describe this", image_embeds=embeds)
    patch_id = gen.tokenizer.reserved_tokens["<img_patch>"]
    # sos, then the image run's opening marker.
    assert tokens[1] == gen.tokenizer.special_tokens["<img>"]
    assert tokens.index(patch_id) == 2


def test_prefix_placement_is_still_the_default():
    gen = _generator(mm_config())
    assert not gen._interleaving()


# ── Chunked prefill with placed modalities ────────────────────────────────

def test_placed_embeddings_are_split_across_prefill_chunks():
    config = mm_config(mm_token_placement="interleaved")
    gen = _generator(config)
    patch_id = gen.tokenizer.reserved_tokens["<img_patch>"]

    piece = torch.tensor([[patch_id, 5, patch_id, 6]])
    embeds = torch.arange(4 * config.d_model, dtype=torch.float32).reshape(1, 4, config.d_model)
    consumed = {patch_id: 0}

    first = gen._chunk_modality_embeds(piece, {patch_id: embeds}, consumed)
    assert first[patch_id].shape[1] == 2
    assert consumed[patch_id] == 2
    assert torch.equal(first[patch_id][0], embeds[0, :2])

    second = gen._chunk_modality_embeds(piece, {patch_id: embeds}, consumed)
    assert torch.equal(second[patch_id][0], embeds[0, 2:4])


def test_a_chunk_with_no_placeholders_asks_for_no_embeddings():
    gen = _generator(mm_config(mm_token_placement="interleaved"))
    patch_id = gen.tokenizer.reserved_tokens["<img_patch>"]
    embeds = torch.randn(1, 2, gen.model.config.d_model)

    chunk = gen._chunk_modality_embeds(torch.tensor([[7, 8]]), {patch_id: embeds}, {patch_id: 0})
    assert chunk == {}


def test_chunk_size_does_not_change_the_answer():
    config = mm_config(mm_token_placement="interleaved", vision_tiling=True, vision_max_tiles=2)
    gen = _generator(config)
    image = torch.randn(3, 32, 32)

    torch.manual_seed(0)
    whole = gen.generate_text("read <img> please", max_new_tokens=4, image=image)
    torch.manual_seed(0)
    chunked = gen.generate_text(
        "read <img> please", max_new_tokens=4, image=image, prefill_chunk_size=3
    )
    assert whole == chunked


def test_a_placed_image_reaches_generation_end_to_end():
    config = mm_config(mm_token_placement="interleaved", vision_tiling=True, vision_max_tiles=2)
    gen = _generator(config)
    out = gen.generate_text("what is in <img>", max_new_tokens=3, image=torch.randn(3, 32, 32))
    assert isinstance(out, str) and out


def test_a_text_only_model_still_refuses_images():
    gen = _generator(tiny_config(text_only=True))
    with pytest.raises(RuntimeError, match="text_only"):
        gen.model.forward_vision(torch.randn(1, 3, 32, 32))

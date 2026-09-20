"""Focused regression tests for multi-image generation in model/serve.py."""

import os
from unittest.mock import MagicMock

from PIL import Image

from model.serve import _save_image, _save_images, handle
from model.utils.image_request import ImageRequest


def test_save_images_saves_all_and_returns_filenames(tmp_path):
    img1 = Image.new("RGB", (32, 32), color="red")
    img2 = Image.new("RGB", (32, 32), color="blue")
    img3 = Image.new("RGB", (32, 32), color="green")

    files = _save_images([img1, img2, img3], str(tmp_path))
    assert len(files) == 3
    assert len(set(files)) == 3
    for f in files:
        assert os.path.isfile(tmp_path / f)
        assert os.path.getsize(tmp_path / f) > 0


def test_save_image_backward_compatibility(tmp_path):
    img1 = Image.new("RGB", (32, 32), color="red")
    img2 = Image.new("RGB", (32, 32), color="blue")

    filename = _save_image([img1, img2], str(tmp_path))
    assert isinstance(filename, str)
    assert os.path.isfile(tmp_path / filename)


def test_save_images_empty(tmp_path):
    assert _save_images([], str(tmp_path)) == []
    assert _save_image([], str(tmp_path)) == ""


def test_serve_handle_image_multiple_images(tmp_path):
    img1 = Image.new("RGB", (64, 64), color="red")
    img2 = Image.new("RGB", (64, 64), color="blue")
    req = ImageRequest(prompt="two cats", width=64, height=64, num_images=2)

    gen = MagicMock()
    gen.generate_image.return_value = ([img1, img2], req)

    res = handle(gen, "image", {"prompt": "two cats", "num_images": 2, "out_dir": str(tmp_path)})

    assert "files" in res
    assert "file" in res
    assert len(res["files"]) == 2
    assert res["file"] == res["files"][0]
    for f in res["files"]:
        assert os.path.isfile(tmp_path / f)
        assert os.path.getsize(tmp_path / f) > 0


def test_serve_handle_image_single_image(tmp_path):
    img1 = Image.new("RGB", (64, 64), color="red")
    req = ImageRequest(prompt="one cat", width=64, height=64, num_images=1)

    gen = MagicMock()
    gen.generate_image.return_value = ([img1], req)

    res = handle(gen, "image", {"prompt": "one cat", "out_dir": str(tmp_path)})

    assert "files" in res
    assert "file" in res
    assert len(res["files"]) == 1
    assert res["file"] == res["files"][0]
    assert os.path.isfile(tmp_path / res["file"])

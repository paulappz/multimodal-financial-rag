"""Figures are shrunk to fit Claude's image limits before the vision call."""

import base64
import os

import pymupdf

from finrag.llm import MAX_IMAGE_SIDE, fit_image, image_block


def _png(width: int, height: int, noise: bool) -> bytes:
    samples = os.urandom(width * height * 3) if noise else bytes(width * height * 3)
    return pymupdf.Pixmap(pymupdf.csRGB, width, height, samples, False).tobytes("png")


def test_small_image_passes_through_unchanged():
    png = _png(800, 600, noise=False)
    assert fit_image(png) == (png, "image/png")


def test_large_photo_page_is_downscaled_under_5mb():
    png = _png(3000, 2200, noise=True)  # incompressible, like a full-page photo spread
    assert len(png) > 5 * 1024 * 1024

    data, media_type = fit_image(png)
    pix = pymupdf.Pixmap(data)
    assert max(pix.width, pix.height) <= MAX_IMAGE_SIDE
    block = image_block(png)
    assert len(block["source"]["data"]) < 5 * 1024 * 1024  # the API measures the base64 payload
    assert block["source"]["media_type"] == media_type
    assert base64.standard_b64decode(block["source"]["data"]) == data

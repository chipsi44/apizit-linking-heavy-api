from __future__ import annotations

from io import BytesIO

import pytest
from PIL import Image

from config import Settings
from ml import ImageService, TextService, decode_image


@pytest.mark.model
def test_real_models_load_and_execute():
    settings = Settings()
    text_vector = TextService(settings).embed(["APIZIT exposes ordinary Python functions."])[0]

    buffer = BytesIO()
    Image.new("RGB", (64, 48), color=(210, 120, 40)).save(buffer, format="PNG")
    decoded = decode_image(buffer.getvalue(), "image/png", settings)
    image_service = ImageService(settings)
    image_vector = image_service.embed(decoded)
    predictions = image_service.classify(decoded)

    assert text_vector.shape == (384,)
    assert image_vector.shape == (2048,)
    assert len(predictions) == 5
    assert 0.0 <= predictions[0]["score"] <= 1.0

"""Transport-independent functions declared by apizit_linking.yaml."""

from __future__ import annotations

import platform
from importlib.metadata import version
from time import sleep
from typing import Any

import cv2
import numpy
import PIL
import scipy
import sklearn
import torch
import torchvision
import transformers

from config import Settings
from errors import APIError, ModelUnavailableError
from ml import ImageService, ModelRegistry, TextService, decode_image

SLOW_RESPONSE_SECONDS = 80
SETTINGS = Settings()
REGISTRY = ModelRegistry()
REGISTRY.register("text", lambda: TextService(SETTINGS))
REGISTRY.register("image", lambda: ImageService(SETTINGS))


def _text_field(value: str, field: str) -> str:
    value = value.strip()
    if not value:
        raise APIError("INVALID_REQUEST", f"The field '{field}' must not be empty.")
    if len(value) > SETTINGS.max_text_length:
        raise APIError(
            "TEXT_TOO_LONG",
            f"The field '{field}' exceeds the {SETTINGS.max_text_length} character limit.",
        )
    return value


def _uploaded_image(upload: Any):
    data = upload.file.read(SETTINGS.max_upload_bytes + 1)
    return decode_image(data, upload.content_type or "", SETTINGS)


def health() -> dict[str, str]:
    return {"status": "ok"}


def info() -> dict[str, object]:
    return {
        "version": SETTINGS.version,
        "framework": "linking",
        "profile": "heavy",
        "python": platform.python_version(),
        "libraries": {
            "numpy": numpy.__version__,
            "opencv": cv2.__version__,
            "pillow": PIL.__version__,
            "scikit_learn": sklearn.__version__,
            "scipy": scipy.__version__,
            "sentence_transformers": version("sentence-transformers"),
            "torch": torch.__version__,
            "torchvision": torchvision.__version__,
            "transformers": transformers.__version__,
        },
        "models": {"text": SETTINGS.text_model_id, "image": SETTINGS.image_model_id},
        "device": "cpu",
    }


def echo(message: str, count: int) -> dict[str, object]:
    return {"received": {"message": _text_field(message, "message"), "count": count}}


def item(item_id: int, include_details: bool = False) -> dict[str, object]:
    if item_id < 1:
        raise APIError("INVALID_REQUEST", "The item ID must be a positive integer.")
    response: dict[str, object] = {
        "item_id": item_id,
        "include_details": include_details,
    }
    if include_details:
        response["details"] = f"Reference item {item_id}"
    return response


def slow() -> dict[str, object]:
    sleep(SLOW_RESPONSE_SECONDS)
    return {"delay_seconds": SLOW_RESPONSE_SECONDS, "status": "completed"}


def ready() -> dict[str, object]:
    model_status: dict[str, bool] = {}
    failures: list[str] = []
    for name in ("text", "image"):
        try:
            REGISTRY.get(name)
            model_status[name] = True
        except ModelUnavailableError as error:
            model_status[name] = False
            failures.append(error.message)
    payload: dict[str, object] = {
        "status": "ready" if not failures else "unavailable",
        "models": model_status,
        "device": "cpu",
        "version": SETTINGS.version,
    }
    if failures:
        payload["error"] = {"code": "MODEL_UNAVAILABLE", "message": " ".join(failures)}
    return payload


def text_embedding(text: str) -> dict[str, object]:
    value = _text_field(text, "text")
    service: TextService = REGISTRY.get("text")
    vector = service.embed([value])[0]
    return {
        "model": service.model_id,
        "dimension": int(vector.shape[0]),
        "embedding": vector.tolist(),
    }


def text_similarity(left: str, right: str) -> dict[str, object]:
    left_value = _text_field(left, "left")
    right_value = _text_field(right, "right")
    service: TextService = REGISTRY.get("text")
    return {
        "model": service.model_id,
        "similarity": service.similarity(left_value, right_value),
    }


def image_analyze(file: Any) -> dict[str, object]:
    decoded = _uploaded_image(file)
    service: ImageService = REGISTRY.get("image")
    return {
        "model": service.model_id,
        "image": {"width": decoded.width, "height": decoded.height, "format": decoded.format},
        "predictions": service.classify(decoded),
    }


def image_embedding(file: Any) -> dict[str, object]:
    decoded = _uploaded_image(file)
    service: ImageService = REGISTRY.get("image")
    vector = service.embed(decoded)
    return {
        "model": service.model_id,
        "dimension": int(vector.shape[0]),
        "embedding": vector.tolist(),
    }

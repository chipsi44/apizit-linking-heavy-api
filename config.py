"""Environment-driven business configuration with no HTTP adapter dependency."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Settings:
    service_name: str = "apizit-linking-heavy-api"
    version: str = "1.0.0"
    max_upload_bytes: int = int(os.getenv("MAX_UPLOAD_BYTES", str(4 * 1024 * 1024)))
    max_text_length: int = int(os.getenv("MAX_TEXT_LENGTH", "5000"))
    max_image_pixels: int = int(os.getenv("MAX_IMAGE_PIXELS", "20000000"))
    model_cache_dir: str = os.getenv("MODEL_CACHE_DIR", "models")
    text_model_id: str = "sentence-transformers/all-MiniLM-L6-v2"
    image_model_id: str = "resnet50-imagenet1k-v2"

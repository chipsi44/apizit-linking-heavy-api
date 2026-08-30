"""Transport-independent business errors."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(slots=True)
class APIError(Exception):
    code: str
    message: str


class ModelUnavailableError(APIError):
    def __init__(self, message: str) -> None:
        super().__init__("MODEL_UNAVAILABLE", message)

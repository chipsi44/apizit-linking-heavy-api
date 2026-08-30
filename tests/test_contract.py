from __future__ import annotations

import ast
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import httpx
from apizit_linking import compile_linking_file
from apizit_linking.fastapi import create_app

from ml import ModelRegistry

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_manifest_declares_exact_route_contract():
    result = compile_linking_file(PROJECT_ROOT / "apizit_linking.yaml", PROJECT_ROOT)

    assert result.is_valid, [diagnostic.to_dict() for diagnostic in result.diagnostics]
    assert {(route.definition.method, route.definition.path) for route in result.routes} == {
        ("GET", "/health"),
        ("GET", "/info"),
        ("POST", "/echo"),
        ("GET", "/items/{item_id}"),
        ("GET", "/slow"),
        ("GET", "/ready"),
        ("POST", "/text/embedding"),
        ("POST", "/text/similarity"),
        ("POST", "/image/analyze"),
        ("POST", "/image/embedding"),
    }


def test_business_code_has_no_web_or_apizit_imports():
    imported_roots = set()
    for filename in ("config.py", "errors.py", "ml.py", "service.py"):
        tree = ast.parse((PROJECT_ROOT / filename).read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported_roots.update(alias.name.partition(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                imported_roots.add(node.module.partition(".")[0])

    assert not imported_roots & {"apizit_linking", "fastapi", "flask", "mangum"}


def client(application=None, *, raise_app_exceptions=True):
    application = application or create_app(PROJECT_ROOT)
    transport = httpx.ASGITransport(
        app=application,
        raise_app_exceptions=raise_app_exceptions,
    )
    return httpx.AsyncClient(transport=transport, base_url="http://example.test")


def linked_globals(application) -> dict[str, object]:
    route = next(route for route in application.routes if route.path == "/health")
    linked_function = route.endpoint.__closure__[0].cell_contents
    return linked_function.__globals__


class TestHttpContract(unittest.IsolatedAsyncioTestCase):
    async def test_health_is_immediate(self):
        application = create_app(PROJECT_ROOT)
        registry = Mock()
        with patch.dict(linked_globals(application), {"REGISTRY": registry}):
            async with client(application) as api_client:
                response = await api_client.get("/health")

        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
        registry.get.assert_not_called()

    async def test_info_contract(self):
        async with client() as api_client:
            response = await api_client.get("/info")

        assert response.status_code == 200
        assert response.json()["framework"] == "linking"
        assert response.json()["profile"] == "heavy"

    async def test_common_routes_and_binding_error(self):
        async with client() as api_client:
            echo = await api_client.post("/echo", json={"message": "hello", "count": 2})
            item = await api_client.get("/items/7", params={"include_details": "true"})
            invalid = await api_client.post("/echo", json={"message": "hello"})

        assert echo.status_code == 200
        assert echo.json() == {"received": {"message": "hello", "count": 2}}
        assert item.status_code == 200
        assert item.json() == {
            "details": "Reference item 7",
            "include_details": True,
            "item_id": 7,
        }
        assert invalid.status_code == 400
        assert "error" in invalid.json()

    async def test_slow_uses_exact_duration_without_waiting(self):
        application = create_app(PROJECT_ROOT)
        mocked_sleep = Mock()
        with patch.dict(linked_globals(application), {"sleep": mocked_sleep}):
            async with client(application) as api_client:
                response = await api_client.get("/slow")

        assert response.status_code == 200
        mocked_sleep.assert_called_once_with(80)
        assert response.json() == {"delay_seconds": 80, "status": "completed"}


class TestHeavyHttpContract(unittest.IsolatedAsyncioTestCase):
    async def test_ready_and_text_routes_use_controlled_services(self):
        from tests.conftest import StubImageService, StubTextService

        registry = ModelRegistry()
        registry.register("text", StubTextService)
        registry.register("image", StubImageService)
        application = create_app(PROJECT_ROOT)
        with patch.dict(linked_globals(application), {"REGISTRY": registry}):
            async with client(application) as api_client:
                ready = await api_client.get("/ready")
                embedding = await api_client.post(
                    "/text/embedding", json={"text": "Deploy with APIZIT"}
                )
                similarity = await api_client.post(
                    "/text/similarity", json={"left": "Python API", "right": "Public service"}
                )

        assert ready.status_code == 200
        assert ready.json()["models"] == {"text": True, "image": True}
        assert embedding.json() == {
            "model": "sentence-transformers/all-MiniLM-L6-v2",
            "dimension": 3,
            "embedding": [0.25, 0.5, 0.75],
        }
        assert similarity.json()["similarity"] == 0.8125

    async def test_image_routes_accept_real_multipart_binding(self):
        from tests.conftest import StubImageService, StubTextService

        buffer = __import__("io").BytesIO()
        __import__("PIL.Image", fromlist=["Image"]).new("RGB", (24, 16), color=(210, 120, 40)).save(
            buffer, format="PNG"
        )
        registry = ModelRegistry()
        registry.register("text", StubTextService)
        registry.register("image", StubImageService)
        files = {"file": ("sample.png", buffer.getvalue(), "image/png")}
        application = create_app(PROJECT_ROOT)
        with patch.dict(linked_globals(application), {"REGISTRY": registry}):
            async with client(application) as api_client:
                analyze = await api_client.post("/image/analyze", files=files)
                embedding = await api_client.post("/image/embedding", files=files)

        assert analyze.status_code == 200
        assert analyze.json()["image"] == {"width": 24, "height": 16, "format": "PNG"}
        assert analyze.json()["predictions"][0] == {"label": "orange", "score": 0.8}
        assert embedding.status_code == 200
        assert embedding.json()["dimension"] == 2048
        assert embedding.json()["embedding"][:3] == [0.0, 1.0, 2.0]

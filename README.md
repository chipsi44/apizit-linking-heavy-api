# APIZIT Linking Heavy API

A standalone declarative Python-to-HTTP reference project for larger APIZIT builds and CPU
machine-learning workloads. Its ten routes are bound in the root `apizit_linking.yaml`; the
business modules contain no web-framework or APIZIT imports. Model services load lazily, keeping
`/health` immediate.

## Run locally

Python 3.12 is required.

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
python -m pip install -r requirements-torch.txt
python -m pip install -r requirements-dev.txt
apizit-linking validate .
apizit-linking preview . --host 0.0.0.0 --port 8000
```

The first call to `/ready` or an ML route downloads the configured public models. `/health` does
not. `/slow` intentionally waits exactly 80 seconds and is only a timeout probe.

## Routes

- `GET /health`
- `GET /info`
- `POST /echo`
- `GET /items/{item_id}?include_details=true`
- `GET /slow`
- `GET /ready`
- `POST /text/embedding`
- `POST /text/similarity`
- `POST /image/analyze`
- `POST /image/embedding`

Image routes expect a multipart upload named `file`. Normal tests use controlled model services
and download no weights. Run `pytest -m model` to opt into the real public models.

The production manifests intentionally omit `apizit-linking`, FastAPI, Mangum, Uvicorn, and
`python-multipart`; APIZIT supplies those adapter dependencies. They are local-only dependencies
in `requirements-dev.txt`. This project intentionally has no Dockerfile.

# Repository Instructions

This repository is the public Heavy APIZIT Linking reference API. Keep it standalone, deployable
with Python 3.12, and aligned with `docs/reference-api-repository-conventions.md` in the canonical
APIZIT platform repository.

Keep `apizit_linking.yaml` at the root and retain exactly ten declared routes. Business modules
must not import Flask, FastAPI, Mangum, or APIZIT. APIZIT-managed runtime packages
(`apizit-linking`, FastAPI, Mangum, and `python-multipart`) belong only in development requirements,
never production manifests.

Keep all direct dependencies pinned, retain the real CPU ML stack, and load models lazily.
`/health` must stay immediate. `/slow` must wait exactly 80 seconds but must never be a health
check. Normal tests must use model doubles; only `pytest -m model` may download model weights.
Never add credentials, caches, model files, a Dockerfile, generated handlers, or cloud resources.

Before committing, run Linking validation, Ruff, pytest, and the import smoke documented in the
README.

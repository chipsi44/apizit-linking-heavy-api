# Contributing

Use Python 3.12 and an isolated virtual environment. Install CPU PyTorch first, followed by the
development dependencies:

```bash
python -m pip install -r requirements-torch.txt
python -m pip install -r requirements-dev.txt
```

Run `apizit-linking validate .`, `ruff check .`, `ruff format --check .`, and `pytest -q` before
opening a pull request. Normal CI uses controlled model services and downloads no model weights.
Use `pytest -m model` only for an intentional real-model execution.

The manifest, ten-route contract, Python version, Heavy dependency profile, and transport-free
business-code rule are suite-wide conventions. Changing one requires a coordinated update to the
canonical APIZIT guide and all nine reference repositories.

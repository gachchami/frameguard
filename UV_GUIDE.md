# `uv` in this project

`uv` manages the lightweight FrameGuard application environment. The GPU model
server is intentionally a separate process because ROCm/vLLM packages depend on
the exact AMD image and GPU architecture.

## Install application dependencies

```bash
uv sync
```

This creates `.venv/` and installs the dependencies declared in `pyproject.toml`.

## Run commands without activating the environment

```bash
uv run python app.py
uv run pytest -q
uv run ruff check .
uv run python scripts/create_demo_video.py
```

`uv run` finds the project environment and executes the command inside it.

## Add a dependency

```bash
uv add package-name
```

For a development-only tool:

```bash
uv add --dev package-name
```

## Lock reproducibly

The first online `uv sync` creates `uv.lock`. Commit it with `pyproject.toml`.
Later machines can use:

```bash
uv sync --locked
```

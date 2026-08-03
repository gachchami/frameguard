# Apply FrameGuard Git Overlay

Run these commands from the FrameGuard repository root after restoring the repository with Git:

```bash
cd /persistent/projects/frameguard

# The archive contains repository-relative paths and does not include a wrapper directory.
unzip -o /path/to/frameguard_git_overlay.zip -d .

# Review every changed file before committing.
git status --short
git diff --stat
git diff

# Validate.
uv run --no-sync python -m compileall app.py frameguard
uv run --no-sync pytest -q

# Commit only after review and successful tests.
git add app.py frameguard tests LOGGING_POLICY.md APPLY_GIT_OVERLAY.md
git commit -m "Add deterministic detection and privacy-safe observability"
```

The archive deliberately does **not** contain or overwrite:

- `frameguard/__init__.py`
- `frameguard/video.py`
- `frameguard/visual_locator.py`

Extracting it overlays only the intended new and modified files. It does not delete any existing repository files.

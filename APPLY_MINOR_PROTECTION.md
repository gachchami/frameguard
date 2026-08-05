# Apply the FrameGuard likely-minors overlay

This is a repository-root overlay. It has no installer directory and does not
delete files that are not present in the archive.

From the FrameGuard repository root:

```bash
unzip -o frameguard_minor_protection_flat_overlay.zip -d .
```

The overlay replaces `app.py` with the combined interface containing all three
face modes:

- Blur every detected face
- Blur only the uploaded reference face
- Blur likely minors and uncertain ages

It also adds:

- `frameguard/minor_protection.py`
- `frameguard/minor_pipeline.py`
- focused unit tests

Run:

```bash
uv run --no-sync python -m compileall -q app.py frameguard/minor_protection.py frameguard/minor_pipeline.py
uv run --no-sync pytest -q tests/test_minor_protection.py tests/test_minor_pipeline.py
./scripts/start.sh
```

The implementation assumes the current YuNet + SFace FrameGuard baseline is
already present.

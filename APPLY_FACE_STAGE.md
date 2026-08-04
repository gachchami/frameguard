# Apply the FrameGuard neural face-redaction stage

This is a repository-root overlay. Extract it over an existing, clean
FrameGuard checkout. It adds or replaces only the listed files; it does not
delete files absent from the archive.

## 1. Apply on the development machine

```bash
cd /path/to/frameguard
git status --short
unzip -o frameguard_face_stage_overlay.zip -d .
```

Review:

```bash
git status --short
git diff --stat
git diff
```

## 2. Download the YuNet model on the internet-connected machine

```bash
uv run python scripts/download_yunet_model.py
uv run python scripts/check_yunet_model.py
```

The downloader verifies SHA-256 before writing the model. Commit the ONNX file
because the AMD target container cannot download it from the internet:

```bash
git add \
  app.py \
  frameguard \
  scripts \
  tests \
  models \
  APPLY_FACE_STAGE.md

git commit -m "Add neural face detection and temporal redaction tracks"
git push
```

## 3. Pull and test in the AMD target container

```bash
cd /persistent/projects/frameguard
git pull --ff-only
unset VIRTUAL_ENV
export UV_LINK_MODE=copy
uv sync

uv run python -m compileall app.py frameguard
uv run pytest -q
uv run python scripts/check_yunet_model.py
```

Expected unit-test result for this overlay on the current baseline:

```text
11 passed
```

## 4. Start FrameGuard

Keep the existing vLLM server running. In the FrameGuard terminal:

```bash
cd /persistent/projects/frameguard
unset VIRTUAL_ENV

export FRAMEGUARD_DETECTOR=qwen
export FRAMEGUARD_API_BASE=http://127.0.0.1:8091/v1
export FRAMEGUARD_MODEL=/workspace/persistent/Qwen2.5-Omni-3B
export FRAMEGUARD_FACE_MODEL=models/face_detection_yunet_2023mar.onnx
export FRAMEGUARD_HOST=127.0.0.1
export FRAMEGUARD_LOG_LEVEL=INFO

uv run python app.py
```

The UI now has **Redact faces with YuNet** plus advanced controls for sampling,
confidence, tracking gap, and minimum observations.

## 5. Acceptance test

Use a short video containing one or more moving faces. After processing:

```bash
REPORT=$(find outputs -maxdepth 1 -name '*_report_*.json' \
  -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)
LOG=$(find outputs -type f -name '*.jsonl' \
  -printf '%T@ %p\n' | sort -nr | head -1 | cut -d' ' -f2-)

jq '.metrics | {
  face_detection_enabled,
  face_frames_sampled,
  face_detections,
  face_tracks,
  face_rejected_tracks,
  face_scan_seconds
}' "$REPORT"

jq 'select(.event | startswith("face_"))' "$LOG"
```

Pass conditions:

- `face_detections > 0` for a video with visible faces.
- `face_tracks >= 1`.
- The findings table includes `face_001`, `face_002`, and so on.
- Blur remains stable between sampled frames.
- Logs contain counts and track geometry summaries, but no image crops, face
  embeddings, names, or biometric identities.

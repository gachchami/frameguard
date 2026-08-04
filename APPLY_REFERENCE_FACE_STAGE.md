# Apply the reference-face redaction stage

This overlay adds a second face-redaction mode:

- blur every face detected by YuNet;
- upload one reference image and blur only matching faces using SFace.

It does not replace Qwen2.5-Omni. Omni continues to analyze video and audio for
semantic secrets, while YuNet/SFace handle spatial face detection and matching.

## 1. Apply as a Git-safe overlay

From the repository root:

```bash
unzip -o frameguard_reference_face_overlay.zip -d .
git status --short
git diff --stat
```

The archive overlays individual files. It does not contain or delete
`frameguard/__init__.py`, `frameguard/video.py`, or
`frameguard/visual_locator.py`.

## 2. Download the SFace model on the Mac

```bash
uv run python scripts/download_sface_model.py
uv run python -m scripts.check_sface_model
```

Expected model:

```text
models/face_recognition_sface_2021dec.onnx
```

Expected SHA-256:

```text
0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79
```

Commit the model so the offline AMD container receives it through Git.

## 3. Validate on the Mac

```bash
uv run python -m compileall app.py frameguard scripts
uv run pytest -q
```

## 4. Push and pull in the AMD container

```bash
git add app.py frameguard scripts tests models APPLY_REFERENCE_FACE_STAGE.md
git commit -m "Add reference-face redaction with SFace"
git push
```

In the target container:

```bash
cd /persistent/projects/frameguard
git pull --ff-only
unset VIRTUAL_ENV
export UV_LINK_MODE=copy
uv sync
uv run python -m compileall app.py frameguard scripts
uv run pytest -q
uv run python -m scripts.check_yunet_model
uv run python -m scripts.check_sface_model
```

## 5. Run

```bash
export FRAMEGUARD_FACE_MODEL=models/face_detection_yunet_2023mar.onnx
export FRAMEGUARD_FACE_RECOGNITION_MODEL=models/face_recognition_sface_2021dec.onnx
export FRAMEGUARD_LOG_LEVEL=DEBUG
uv run python app.py
```

## 6. Test reference-only redaction

Use `dual_subjects_*.mp4` from the provided dataset.

1. Select **Blur only the uploaded reference face**.
2. Upload `reference/subject_a.png`.
3. Confirm only subject A is blurred.
4. Repeat with `reference/subject_b.png`.

The default SFace cosine threshold is `0.363`, matching OpenCV's published
example threshold. Treat it as a starting point and tune it using your own
consented validation set.

## Privacy behavior

- The uploaded reference image is not written to the JSON audit report.
- The reference image is not written to the JSONL run log.
- The derived SFace embedding is retained only in process memory for the run.
- FrameGuard does not assign a name or search a face database.

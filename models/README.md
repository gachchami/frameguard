# FrameGuard face models

FrameGuard uses two OpenCV Zoo models:

```text
models/face_detection_yunet_2023mar.onnx
models/face_recognition_sface_2021dec.onnx
```

## Download on an internet-connected development machine

```bash
uv run python scripts/download_yunet_model.py
uv run python scripts/download_sface_model.py
```

Expected SHA-256 values:

```text
YuNet:  8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4
SFace:  0ba9fbfa01b5270c96627c4ef784da859931e02f04419c829e83484087c34e79
```

Commit both ONNX files to the FrameGuard repository so an offline target
container receives them through `git pull`.

## Roles

- YuNet detects face boxes and five facial landmarks.
- SFace compares a detected face with one user-uploaded reference face.

When reference-only mode is used, the uploaded image and derived embedding are
kept in memory for the current run. FrameGuard does not write the image, crop,
or embedding to the run log or JSON audit report.

## Validate

```bash
uv run python -m scripts.check_yunet_model
uv run python -m scripts.check_sface_model
```

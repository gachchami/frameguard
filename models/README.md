# FrameGuard model files

FrameGuard expects the YuNet model at:

```text
models/face_detection_yunet_2023mar.onnx
```

Download and verify it on an internet-connected development machine:

```bash
uv run python scripts/download_yunet_model.py
```

The expected SHA-256 is:

```text
8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4
```

Commit the small ONNX file to the FrameGuard repository so the offline target
container receives it through `git pull`.

YuNet and the files in its OpenCV Zoo directory are MIT-licensed. FrameGuard
uses face **detection**, not face recognition: it stores boxes and temporary
track labels, not biometric embeddings or identities.

Validate the model and OpenCV integration with:

```bash
uv run python scripts/check_yunet_model.py
```

# Running FrameGuard on a Mac laptop

The laptop run has two separate goals:

1. Verify the normal Python application and video-redaction plumbing.
2. Later connect the same application to the real Qwen2.5-Omni server on an AMD Linux machine.

The included laptop smoke-test mode is **not an LLM**. It returns controlled
findings for `samples/frameguard_demo.mp4` so you can test chunking, OCR boxes,
visual blur, audio mute, the JSON report, and React interface.

## Install system tools

```bash
brew install uv ffmpeg tesseract espeak-ng node
```

`espeak-ng` is only needed to regenerate the synthetic sample.

## Install Python dependencies

```bash
uv sync
```

## Run tests

```bash
uv run pytest -q
uv run ruff check .
```

## Regenerate the sample, if desired

```bash
uv run python scripts/create_demo_video.py
```

## Start FrameGuard with the real Qwen model

```bash
./scripts/start-mac.sh
```

The launcher downloads and starts `Qwen/Qwen2.5-Omni-3B` locally on first use,
then opens `http://localhost:7860`. Model loading can take several minutes.
Qwen's official Transformers implementation handles the video's visual and
audio streams together and uses Metal acceleration when available.

## What should work locally

- React UI and FastAPI backend
- MP4 upload
- FFmpeg chunking
- OCR localization with Tesseract
- Visual blurring
- Audio muting
- Output MP4
- JSON audit report

## What this does not validate

- Qwen2.5-Omni model loading
- Video-and-audio understanding by the model
- AMD GPU acceleration
- ROCm or vLLM-Omni

When the AMD server is ready, choose **Qwen2.5-Omni model server** and point the
API-base field to that server. The rest of the application remains unchanged.

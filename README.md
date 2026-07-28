# FrameGuard

FrameGuard is a local multimodal video-redaction prototype. In its real model
mode, Qwen2.5-Omni examines both the visible frames and embedded audio of a
screen recording. FrameGuard then uses OCR for precise visual coordinates and
FFmpeg/OpenCV for deterministic redaction.

## Architecture

```text
MP4
  -> split into short clips
  -> Qwen2.5-Omni sees video and hears embedded audio
  -> structured findings
  -> OCR locates visible values in pixels
  -> OpenCV blurs visual regions
  -> FFmpeg mutes spoken-secret intervals
  -> redacted MP4 + JSON audit report
```

## Start on a Mac laptop

See [LAPTOP_GUIDE.md](LAPTOP_GUIDE.md). The short path is:

```bash
brew install ffmpeg tesseract espeak-ng
uv sync
uv run pytest -q
uv run python app.py
```

Open `http://localhost:7860`, keep **Laptop smoke test — no LLM** selected, and
upload `samples/frameguard_demo.mp4`.

The smoke-test mode is deliberately labeled as non-LLM. It validates the full
application and redaction plumbing against the controlled sample.

## Start the real model on the AMD Linux machine

In the matching ROCm/vLLM-Omni environment:

```bash
export FRAMEGUARD_MODEL=Qwen/Qwen2.5-Omni-3B
bash scripts/start_model_server.sh
```

Then run FrameGuard and select **Qwen2.5-Omni model server** in the UI:

```bash
export FRAMEGUARD_DETECTOR=qwen
uv run python scripts/check_server.py
uv run python app.py
```

## Tests and lint

```bash
uv run pytest -q
uv run ruff check .
```

## Important MVP limitations

The model provides approximate timing. Audio redaction therefore uses padded
model intervals rather than forced word alignment. Visual findings are blurred
only when Tesseract can locate the model-identified exact value. The JSON report
keeps unlocalized visual findings visible instead of claiming they were removed.

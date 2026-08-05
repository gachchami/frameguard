# FrameGuard

FrameGuard is a local multimodal video-redaction system. It analyzes video and
audio, detects sensitive information and faces, applies visual or audio
redaction, and produces a machine-readable audit trail.

The current implementation combines:

- **Qwen2.5-Omni-3B** through a local vLLM endpoint for semantic video/audio analysis.
- **Tesseract OCR and deterministic pattern matching** for precise text localization.
- **YuNet** for neural face detection and temporal tracking.
- **SFace** for uploaded reference-face matching.
- **Qwen holistic child/adult classification** for experimental child-focused redaction.
- **OpenCV and FFmpeg** for visual redaction, audio muting, and MP4 export.

## Capabilities

| Capability | Implementation |
|---|---|
| Sensitive visual text | Qwen semantic detection plus OCR/regex verification |
| Sensitive spoken content | Qwen audio analysis with padded mute intervals |
| QR-code redaction | OpenCV QR detection |
| Blur all faces | YuNet detection with temporal tracks and interpolated boxes |
| Blur one reference face | YuNet tracks plus SFace similarity matching |
| Blur visually apparent children | YuNet tracks plus multi-view, multi-timestamp Qwen classification |
| Auditability | JSON report and privacy-safe JSONL run log |

## Processing pipeline

```text
Uploaded MP4
    |
    +--> split into short video chunks
    |       |
    |       +--> extract mono WAV audio
    |       +--> Qwen2.5-Omni semantic analysis
    |
    +--> OCR / regex / QR scan
    |
    +--> YuNet face detection and temporal tracking
            |
            +--> all-face policy
            +--> SFace reference-match policy
            +--> holistic child/adult policy
    |
    +--> merge and validate findings
    |
    +--> OpenCV visual redaction + FFmpeg audio muting
    |
    +--> redacted MP4 + JSON report + JSONL run log
```

## Face-redaction modes

### All faces

Every accepted YuNet face track is blurred.

### Reference face

A reference image is aligned with YuNet and embedded with SFace. Only tracks
whose similarity score exceeds the configured threshold are blurred. Reference
images and embeddings are held in memory for the current run.

### Visually apparent children

FrameGuard samples the strongest, temporally separated observations from each
YuNet face track. For every selected timestamp, it sends three complementary
views to the configured local Qwen2.5-Omni endpoint:

1. The full scene with the target face marked.
2. A generous target-person and surrounding-context crop.
3. A target-face crop.

Qwen classifies each timestamp as `child`, `adult`, or `uncertain`. FrameGuard
then applies its own track-level consensus policy rather than trusting one
single model response.

Default policy:

| Result | Action |
|---|---|
| At least 3 reliable child votes, at least 70% consensus, and no adult vote | Blur |
| At least 3 reliable adult votes, at least 70% consensus, and no child vote | Leave visible |
| Too few usable timestamps, mixed child/adult votes, or weak consensus | Leave visible |
| Classifier failure | Convert to uncertain and continue |

Enabling **Also blur uncertain classifications** applies a stricter privacy
policy but can blur adults. The classification describes visual appearance; it
does not prove legal age. Height, clothing, hairstyle, gender, ethnicity,
disability, and text visible in the scene are explicitly excluded as standalone
age evidence.

## Requirements

### System tools

- FFmpeg and FFprobe
- Tesseract OCR
- `uv`
- A ROCm-compatible vLLM environment for Qwen2.5-Omni

### Local models

```text
/workspace/persistent/Qwen2.5-Omni-3B
models/face_detection_yunet_2023mar.onnx
models/face_recognition_sface_2021dec.onnx
```

The Qwen path can be overridden with `FRAMEGUARD_MODEL_PATH`. The YuNet and
SFace paths can be overridden independently.

## Start

From the repository root:

```bash
chmod +x scripts/start.sh
./scripts/start.sh
```

The script:

1. Validates required tools and model files.
2. Reuses an existing healthy vLLM server or starts one.
3. Disables optional telemetry and remote model lookups.
4. Starts FrameGuard with the repository's `uv` environment.
5. Writes runtime logs under `outputs/runtime/`.

Default endpoints:

```text
FrameGuard: http://127.0.0.1:7860
vLLM:      http://127.0.0.1:8091
```

### Useful overrides

```bash
FRAMEGUARD_LOG_LEVEL=DEBUG ./scripts/start.sh
FRAMEGUARD_SYNC=1 ./scripts/start.sh
FRAMEGUARD_PORT=7861 ./scripts/start.sh
FRAMEGUARD_MODEL_PATH=/path/to/Qwen2.5-Omni-3B ./scripts/start.sh
```

| Variable | Default | Purpose |
|---|---|---|
| `FRAMEGUARD_MODEL_PATH` | Auto-detected | Local Qwen model directory |
| `FRAMEGUARD_FACE_MODEL` | `models/face_detection_yunet_2023mar.onnx` | YuNet model |
| `FRAMEGUARD_FACE_RECOGNITION_MODEL` | `models/face_recognition_sface_2021dec.onnx` | SFace model |
| `FRAMEGUARD_HOST` | `127.0.0.1` | Gradio bind address |
| `FRAMEGUARD_PORT` | `7860` | Gradio port |
| `FRAMEGUARD_API_PORT` | `8091` | Local vLLM port |
| `FRAMEGUARD_LOG_LEVEL` | `INFO` | Application log level |
| `FRAMEGUARD_SYNC` | `0` | Run `uv sync` before startup when set to `1` |

## Outputs

Each run creates:

```text
outputs/
├── <input>_redacted_<run-id>.mp4
├── <input>_report_<run-id>.json
└── logs/<input>_<run-id>.jsonl
```

The report contains configuration, metrics, findings, redaction decisions, and
per-track child/adult/uncertain decisions when that policy is enabled.

The JSONL log records stage timings, counts, model request metadata, and
redaction outcomes. It does not contain detected secret values, raw model
responses, media payloads, face crops, reference images, or face embeddings.

## Development

```bash
uv sync
uv run pytest -q
uv run ruff check .
uv run python -m compileall -q app.py frameguard scripts
```

Model validation:

```bash
uv run python -m scripts.check_yunet_model
uv run python -m scripts.check_sface_model
```

## Limitations

- Semantic model timestamps are approximate rather than word-aligned.
- OCR can miss small, moving, low-contrast, or partially obscured text.
- Face detection can degrade under occlusion, motion blur, profile views, and
  very small face sizes.
- SFace thresholds require calibration on representative footage.
- Visual child/adult classification can be inaccurate, especially for small,
  blurred, occluded, or boundary-age subjects, and should be treated as an
  experimental privacy heuristic.
- The current pipeline performs one analysis-and-redaction pass; it does not yet
  run an autonomous verification-and-retry loop.

## Hackathon track

FrameGuard's primary submission track is **Multimodal AI**. The core product
jointly reasons over video, audio, OCR, face imagery, and structured model
output to create an edited media artifact.

The project can be extended toward **Agentic AI** by adding a closed-loop stage
that inspects the rendered output, verifies that selected content was actually
redacted, adjusts thresholds or intervals, and reruns failed sections.

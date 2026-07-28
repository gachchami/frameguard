from __future__ import annotations

import json
import os
from pathlib import Path

import gradio as gr

from frameguard.pipeline import analyze_video, findings_table

OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

DEFAULT_API_BASE = os.environ.get("FRAMEGUARD_API_BASE", "http://127.0.0.1:8091/v1")
DEFAULT_MODEL = os.environ.get("FRAMEGUARD_MODEL", "Qwen/Qwen2.5-Omni-3B")
DEFAULT_API_KEY = os.environ.get("FRAMEGUARD_API_KEY", "EMPTY")
DEFAULT_CHUNK_SECONDS = float(os.environ.get("FRAMEGUARD_CHUNK_SECONDS", "5"))
DEFAULT_DETECTOR = os.environ.get("FRAMEGUARD_DETECTOR", "mock")

MODE_LABELS = {
    "Laptop smoke test — no LLM": "mock",
    "Qwen2.5-Omni model server": "qwen",
}


def run_pipeline(
    video_path: str | None,
    detector_label: str,
    api_base: str,
    model: str,
    chunk_seconds: float,
):
    if not video_path:
        raise gr.Error("Upload an MP4 video first.")
    detector_mode = MODE_LABELS[detector_label]
    try:
        result = analyze_video(
            video_path,
            api_base=api_base,
            model=model,
            api_key=DEFAULT_API_KEY,
            chunk_seconds=float(chunk_seconds),
            output_dir=OUTPUT_DIR,
            detector_mode=detector_mode,
        )
    except Exception as exc:
        raise gr.Error(str(exc)) from exc

    return (
        str(result.output_video),
        findings_table(result.findings),
        json.dumps(result.metrics, indent=2),
        str(result.report_path),
    )


def _default_label() -> str:
    return (
        "Qwen2.5-Omni model server"
        if DEFAULT_DETECTOR == "qwen"
        else "Laptop smoke test — no LLM"
    )


with gr.Blocks(title="FrameGuard") as frameguard_app:
    gr.Markdown(
        """
# FrameGuard
Upload a short screen recording. In production mode, a local Qwen2.5-Omni
server examines both the visible video and its embedded audio, identifies
sensitive information, and exports a redacted MP4.

**Laptop smoke-test mode is deterministic and does not run an LLM.** It exists
only to test the sample video, OCR localization, visual blur, audio mute, report,
and UI before moving to the AMD Linux machine.
"""
    )
    detector = gr.Radio(
        choices=list(MODE_LABELS),
        value=_default_label(),
        label="Detector",
    )
    with gr.Row():
        input_video = gr.Video(label="Original video", sources=["upload"], format="mp4")
        output_video = gr.Video(label="Redacted video")

    with gr.Accordion("Model server", open=False):
        api_base = gr.Textbox(label="vLLM-Omni API base", value=DEFAULT_API_BASE)
        model = gr.Textbox(label="Model", value=DEFAULT_MODEL)
        chunk_seconds = gr.Slider(
            minimum=3,
            maximum=10,
            step=1,
            value=DEFAULT_CHUNK_SECONDS,
            label="Chunk length in seconds",
        )

    analyze_button = gr.Button("Analyze video and audio, then redact", variant="primary")
    findings = gr.Dataframe(
        headers=[
            "Type",
            "Value",
            "Modality",
            "Confidence",
            "Start (s)",
            "End (s)",
            "OCR boxes",
            "Reason",
        ],
        interactive=False,
        label="Multimodal findings",
    )
    metrics = gr.Code(label="Metrics", language="json")
    report = gr.File(label="JSON audit report")

    analyze_button.click(
        fn=run_pipeline,
        inputs=[input_video, detector, api_base, model, chunk_seconds],
        outputs=[output_video, findings, metrics, report],
    )


if __name__ == "__main__":
    frameguard_app.launch(
        server_name="0.0.0.0",
        server_port=int(os.environ.get("FRAMEGUARD_PORT", "7860")),
    )

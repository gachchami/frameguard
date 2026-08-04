from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import gradio as gr

from frameguard.observability import configure_application_logging
from frameguard.pipeline import analyze_video, findings_table

configure_application_logging()
LOGGER = logging.getLogger("frameguard.app")

OUTPUT_DIR = Path(os.environ.get("FRAMEGUARD_OUTPUT_DIR", "outputs"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_API_BASE = os.environ.get("FRAMEGUARD_API_BASE", "http://127.0.0.1:8091/v1")
DEFAULT_MODEL = os.environ.get(
    "FRAMEGUARD_MODEL",
    "/workspace/persistent/Qwen2.5-Omni-3B",
)
DEFAULT_API_KEY = os.environ.get("FRAMEGUARD_API_KEY", "EMPTY")
DEFAULT_CHUNK_SECONDS = float(os.environ.get("FRAMEGUARD_CHUNK_SECONDS", "5"))
DEFAULT_DETECTOR_MODE = os.environ.get("FRAMEGUARD_DETECTOR", "qwen")
DEFAULT_FACE_MODEL = os.environ.get(
    "FRAMEGUARD_FACE_MODEL",
    "models/face_detection_yunet_2023mar.onnx",
)


def run_pipeline(
    video_path: str | None,
    api_base: str,
    model: str,
    chunk_seconds: float,
    deterministic_ocr: bool,
    detect_qr_codes: bool,
    deterministic_sample_interval_ms: int,
    redact_faces: bool,
    face_model_path: str,
    face_sample_interval_ms: int,
    face_score_threshold: float,
    face_max_track_gap_ms: int,
    face_min_track_observations: int,
    run_log_level: str,
    show_sensitive_values: bool,
    include_raw_model_output: bool,
):
    if not video_path:
        raise gr.Error("Upload an MP4 video first.")

    LOGGER.info("FrameGuard run requested")
    try:
        result = analyze_video(
            video_path,
            api_base=api_base,
            model=model,
            api_key=DEFAULT_API_KEY,
            chunk_seconds=float(chunk_seconds),
            output_dir=OUTPUT_DIR,
            detector_mode=DEFAULT_DETECTOR_MODE,
            deterministic_ocr=bool(deterministic_ocr),
            detect_qr_codes=bool(detect_qr_codes),
            deterministic_sample_interval_ms=int(deterministic_sample_interval_ms),
            redact_faces=bool(redact_faces),
            face_model_path=face_model_path,
            face_sample_interval_ms=int(face_sample_interval_ms),
            face_score_threshold=float(face_score_threshold),
            face_max_track_gap_ms=int(face_max_track_gap_ms),
            face_min_track_observations=int(face_min_track_observations),
            run_log_level=str(run_log_level),
            include_sensitive_values_in_report=bool(show_sensitive_values),
            include_raw_model_output=bool(include_raw_model_output),
        )
    except Exception as exc:
        LOGGER.exception("FrameGuard run failed")
        raise gr.Error(str(exc)) from exc

    report_payload = json.loads(result.report_path.read_text(encoding="utf-8"))
    report_preview = {
        "run_id": report_payload.get("run_id"),
        "privacy": report_payload.get("privacy"),
        "configuration": report_payload.get("configuration"),
        "metrics": report_payload.get("metrics"),
        "findings": report_payload.get("findings"),
        "model_responses": report_payload.get("model_responses"),
        "instrumentation": report_payload.get("instrumentation"),
    }

    return (
        str(result.output_video),
        str(result.output_video),
        findings_table(
            result.findings,
            show_sensitive_values=bool(show_sensitive_values),
        ),
        json.dumps(result.metrics, indent=2),
        json.dumps(report_preview, indent=2),
        str(result.report_path),
        str(result.log_path),
    )


with gr.Blocks(title="FrameGuard") as frameguard_app:
    gr.Markdown(
        """
# FrameGuard

Analyze a recording locally with Qwen2.5-Omni, deterministic OCR validation,
and QR detection. Review the redacted preview, audit report, and privacy-safe
per-run instrumentation before downloading the result.

**Logging policy:** INFO and DEBUG logs never contain detected values, raw Qwen
responses, prompts, credentials, or media data. DEBUG adds safe detail such as
request IDs, byte counts, response lengths, stage timings, and localization counts.
"""
    )

    with gr.Row():
        input_video = gr.Video(label="Original video", sources=["upload"], format="mp4")
        output_video = gr.Video(label="Redacted preview")

    with gr.Row():
        deterministic_ocr = gr.Checkbox(
            value=True,
            label="Deterministic OCR safety scan",
            info=(
                "Validates emails, IP addresses, known API-key formats, account IDs, "
                "phone numbers, and private URLs."
            ),
        )
        detect_qr_codes = gr.Checkbox(value=True, label="Redact QR codes")
        redact_faces = gr.Checkbox(
            value=True,
            label="Redact faces with YuNet",
            info=(
                "Runs a neural face detector on sampled frames, associates boxes into "
                "temporary tracks, and interpolates blur positions between samples."
            ),
        )

    with gr.Accordion("Advanced settings", open=False):
        api_base = gr.Textbox(label="vLLM API base", value=DEFAULT_API_BASE)
        model = gr.Textbox(label="Model", value=DEFAULT_MODEL)
        chunk_seconds = gr.Slider(
            minimum=3,
            maximum=10,
            step=1,
            value=DEFAULT_CHUNK_SECONDS,
            label="Qwen chunk length in seconds",
        )
        deterministic_sample_interval_ms = gr.Slider(
            minimum=200,
            maximum=1200,
            step=50,
            value=350,
            label="OCR/QR sampling interval in milliseconds",
        )
        face_model_path = gr.Textbox(
            label="YuNet model path",
            value=DEFAULT_FACE_MODEL,
        )
        face_sample_interval_ms = gr.Slider(
            minimum=100,
            maximum=1000,
            step=50,
            value=200,
            label="Face detection sampling interval in milliseconds",
            info="Lower values improve coverage but run more neural inference.",
        )
        face_score_threshold = gr.Slider(
            minimum=0.5,
            maximum=0.95,
            step=0.01,
            value=0.75,
            label="Face confidence threshold",
            info="Lower values favor privacy recall; higher values reduce false positives.",
        )
        face_max_track_gap_ms = gr.Slider(
            minimum=300,
            maximum=2000,
            step=100,
            value=900,
            label="Maximum face-track gap in milliseconds",
        )
        face_min_track_observations = gr.Slider(
            minimum=1,
            maximum=5,
            step=1,
            value=2,
            label="Minimum observations per face track",
        )
        run_log_level = gr.Dropdown(
            choices=["INFO", "DEBUG"],
            value="INFO",
            label="Run log detail",
            info="DEBUG remains privacy-safe; it records more timings and counts.",
        )
        show_sensitive_values = gr.Checkbox(
            value=False,
            label="Show exact detected values in UI and JSON report",
            info="Off by default. The report otherwise stores masked previews and fingerprints.",
        )
        include_raw_model_output = gr.Checkbox(
            value=False,
            label="Include raw Qwen output in JSON report",
            info=(
                "Sensitive: raw model output can contain every detected secret. "
                "It is never written to the run log."
            ),
        )

    analyze_button = gr.Button("Analyze and preview redaction", variant="primary")

    findings = gr.Dataframe(
        headers=[
            "Type",
            "Value",
            "Modality",
            "Confidence",
            "Start (s)",
            "End (s)",
            "Boxes",
            "Sources",
            "Action",
            "Reason",
        ],
        datatype=[
            "str",
            "str",
            "str",
            "number",
            "number",
            "number",
            "number",
            "str",
            "str",
            "str",
        ],
        interactive=False,
        label="Detected privacy findings",
    )

    with gr.Row():
        metrics = gr.Code(label="Processing metrics", language="json")
        report_preview = gr.Code(label="Audit report preview", language="json")

    with gr.Row():
        redacted_download = gr.File(label="Download redacted MP4")
        report_download = gr.File(label="Download JSON audit report")
        log_download = gr.File(label="Download privacy-safe run log")

    analyze_button.click(
        fn=run_pipeline,
        inputs=[
            input_video,
            api_base,
            model,
            chunk_seconds,
            deterministic_ocr,
            detect_qr_codes,
            deterministic_sample_interval_ms,
            redact_faces,
            face_model_path,
            face_sample_interval_ms,
            face_score_threshold,
            face_max_track_gap_ms,
            face_min_track_observations,
            run_log_level,
            show_sensitive_values,
            include_raw_model_output,
        ],
        outputs=[
            output_video,
            redacted_download,
            findings,
            metrics,
            report_preview,
            report_download,
            log_download,
        ],
    )


if __name__ == "__main__":
    frameguard_app.launch(
        server_name=os.environ.get("FRAMEGUARD_HOST", "127.0.0.1"),
        server_port=int(os.environ.get("FRAMEGUARD_PORT", "7860")),
    )

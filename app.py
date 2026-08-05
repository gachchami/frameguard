from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import gradio as gr

from frameguard.face_reference import DEFAULT_COSINE_THRESHOLD
from frameguard.minor_pipeline import analyze_video_with_face_policy
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
DEFAULT_FACE_RECOGNITION_MODEL = os.environ.get(
    "FRAMEGUARD_FACE_RECOGNITION_MODEL",
    "models/face_recognition_sface_2021dec.onnx",
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
    face_redaction_mode: str,
    reference_face_path: str | None,
    face_recognition_model_path: str,
    reference_match_threshold: float,
    child_minimum_confidence: float,
    child_minimum_usable_timestamps: int,
    child_consensus_fraction: float,
    child_max_samples_per_track: int,
    child_continue_on_error: bool,
    child_blur_uncertain: bool,
    run_log_level: str,
    show_sensitive_values: bool,
    include_raw_model_output: bool,
):
    if not video_path:
        raise gr.Error("Upload an MP4 video first.")

    normalized_face_mode = str(face_redaction_mode).strip().lower()
    if int(child_minimum_usable_timestamps) > int(child_max_samples_per_track):
        raise gr.Error(
            "Minimum usable timestamps cannot exceed maximum timestamps per track."
        )
    if redact_faces and normalized_face_mode == "reference" and not reference_face_path:
        raise gr.Error(
            "Upload one clear reference-face image when using reference-only face redaction."
        )

    common_kwargs = {
        "api_base": api_base,
        "model": model,
        "api_key": DEFAULT_API_KEY,
        "chunk_seconds": float(chunk_seconds),
        "output_dir": OUTPUT_DIR,
        "detector_mode": DEFAULT_DETECTOR_MODE,
        "deterministic_ocr": bool(deterministic_ocr),
        "detect_qr_codes": bool(detect_qr_codes),
        "deterministic_sample_interval_ms": int(deterministic_sample_interval_ms),
        "face_model_path": face_model_path,
        "face_sample_interval_ms": int(face_sample_interval_ms),
        "face_score_threshold": float(face_score_threshold),
        "face_max_track_gap_ms": int(face_max_track_gap_ms),
        "face_min_track_observations": int(face_min_track_observations),
        "run_log_level": str(run_log_level),
        "include_sensitive_values_in_report": bool(show_sensitive_values),
        "include_raw_model_output": bool(include_raw_model_output),
    }

    LOGGER.info("FrameGuard run requested")
    try:
        if redact_faces and normalized_face_mode == "likely_minors":
            result = analyze_video_with_face_policy(
                video_path,
                face_redaction_mode="likely_minors",
                child_minimum_confidence=float(child_minimum_confidence),
                child_minimum_usable_timestamps=int(child_minimum_usable_timestamps),
                child_consensus_fraction=float(child_consensus_fraction),
                child_max_samples_per_track=int(child_max_samples_per_track),
                child_continue_on_error=bool(child_continue_on_error),
                child_blur_uncertain=bool(child_blur_uncertain),
                **common_kwargs,
            )
        else:
            result = analyze_video(
                video_path,
                redact_faces=bool(redact_faces),
                face_redaction_mode=normalized_face_mode,
                reference_face_path=reference_face_path,
                face_recognition_model_path=face_recognition_model_path,
                reference_match_threshold=float(reference_match_threshold),
                **common_kwargs,
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
        "child_classification": report_payload.get("child_classification"),
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


with gr.Blocks(title="FrameGuard", analytics_enabled=False) as frameguard_app:
    gr.Markdown(
        """
# FrameGuard

FrameGuard analyzes video and audio for sensitive information, localizes visual
content, and produces a redacted MP4 with an audit report. Face protection
supports all-face redaction, reference-face matching, and holistic visual
child/adult classification.

The child-protection mode examines multiple timestamps for each YuNet track. At
each timestamp, Qwen receives the marked full scene, a person-context crop, and
a face crop. It classifies the person as visually appearing to be a child, an
adult, or uncertain. This is a privacy heuristic and does not establish legal
age.

By default, only tracks with multi-timestamp child consensus are blurred.
Uncertain tracks remain visible unless **Also blur uncertain classifications**
is enabled.

**Logging policy:** run logs exclude detected secret values, raw model output,
prompts, credentials, media payloads, reference images, face crops, and face
embeddings.
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

    with gr.Row():
        face_redaction_mode = gr.Radio(
            choices=[
                ("Blur every detected face", "all"),
                ("Blur only the uploaded reference face", "reference"),
                ("Blur visually apparent children only", "likely_minors"),
            ],
            value="all",
            label="Face redaction mode",
        )
        reference_face = gr.Image(
            label="Reference face image (reference-only mode)",
            type="filepath",
            sources=["upload"],
            height=220,
        )

    gr.Markdown(
        "Reference images and SFace embeddings are held in memory for one run. "
        "Child-protection mode sends selected full-scene, person-context, and "
        "face views to the configured local Omni endpoint."
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
        face_recognition_model_path = gr.Textbox(
            label="SFace recognition model path",
            value=DEFAULT_FACE_RECOGNITION_MODEL,
        )
        reference_match_threshold = gr.Slider(
            minimum=0.20,
            maximum=0.80,
            step=0.01,
            value=DEFAULT_COSINE_THRESHOLD,
            label="Reference-face cosine match threshold",
            info=(
                "Higher values are stricter. OpenCV's published SFace cosine "
                "threshold is 0.363; calibrate it on a representative validation set."
            ),
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

        gr.Markdown("### Visually apparent child policy")
        with gr.Row():
            child_minimum_confidence = gr.Slider(
                minimum=0.50,
                maximum=0.95,
                step=0.01,
                value=0.70,
                label="Minimum per-timestamp confidence",
                info="Lower-confidence timestamp judgments do not count toward consensus.",
            )
            child_max_samples_per_track = gr.Slider(
                minimum=3,
                maximum=8,
                step=1,
                value=5,
                label="Maximum timestamps per face track",
                info=(
                    "The best temporally separated timestamps are selected using face "
                    "size, detector confidence, and sharpness."
                ),
            )
        with gr.Row():
            child_minimum_usable_timestamps = gr.Slider(
                minimum=2,
                maximum=6,
                step=1,
                value=3,
                label="Minimum usable timestamps",
                info="At least this many reliable judgments are required before blurring.",
            )
            child_consensus_fraction = gr.Slider(
                minimum=0.50,
                maximum=1.00,
                step=0.05,
                value=0.70,
                label="Required child/adult consensus",
                info=(
                    "A track must reach this fraction without an opposite-class vote. "
                    "Mixed child/adult votes become uncertain."
                ),
            )
        child_blur_uncertain = gr.Checkbox(
            value=False,
            label="Also blur uncertain classifications",
            info=(
                "Off means only tracks with strong child consensus are blurred. Turning "
                "this on is privacy-conservative but can blur adults."
            ),
        )
        child_continue_on_error = gr.Checkbox(
            value=True,
            label="Continue when child classification fails",
            info=(
                "Converts classifier errors into an uncertain result instead of stopping "
                "the complete run. The uncertainty setting controls redaction."
            ),
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
                "Sensitive: raw secret-detection output can contain every detected secret. "
                "Child-classification prompts and raw responses are never included."
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
            face_redaction_mode,
            reference_face,
            face_recognition_model_path,
            reference_match_threshold,
            child_minimum_confidence,
            child_minimum_usable_timestamps,
            child_consensus_fraction,
            child_max_samples_per_track,
            child_continue_on_error,
            child_blur_uncertain,
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
        share=False,
    )

from __future__ import annotations

import json
import logging
import os
from pathlib import Path

import gradio as gr

from frameguard.face_gallery import (
    FaceGallerySession,
    analyze_video_with_face_gallery,
    match_uploaded_reference_photos,
    scan_face_gallery,
)
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


def _normalize_face_mode(value: object) -> str:
    raw = str(value or "").strip().lower()
    aliases = {
        "all": "all",
        "blur every detected face": "all",
        "reference": "reference",
        "blur only the uploaded reference face": "reference",
        "likely_minors": "likely_minors",
        "likely_minor": "likely_minors",
        "blur likely minors only": "likely_minors",
        "blur visually apparent children only": "likely_minors",
        "blur visually apparent children only (experimental)": "likely_minors",
    }
    normalized = aliases.get(raw)
    if normalized is None:
        raise gr.Error(f"Unexpected face-redaction mode received from the UI: {value!r}")
    return normalized


def _result_outputs(result, *, show_sensitive_values: bool):
    report_payload = json.loads(result.report_path.read_text(encoding="utf-8"))
    report_preview = {
        "run_id": report_payload.get("run_id"),
        "privacy": report_payload.get("privacy"),
        "configuration": report_payload.get("configuration"),
        "metrics": report_payload.get("metrics"),
        "face_gallery": report_payload.get("face_gallery"),
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

    normalized_face_mode = _normalize_face_mode(face_redaction_mode)
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

    LOGGER.info(
        "FrameGuard run requested raw_face_mode=%r normalized_face_mode=%s",
        face_redaction_mode,
        normalized_face_mode,
    )
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

    return _result_outputs(result, show_sensitive_values=bool(show_sensitive_values))


def extract_face_profiles(
    video_path: str | None,
    face_model_path: str,
    face_recognition_model_path: str,
    face_sample_interval_ms: int,
    face_score_threshold: float,
    face_max_track_gap_ms: int,
    face_min_track_observations: int,
    identity_similarity_threshold: float,
):
    if not video_path:
        raise gr.Error("Upload an MP4 video first.")

    try:
        session = scan_face_gallery(
            video_path,
            face_model_path=face_model_path,
            face_recognition_model_path=face_recognition_model_path,
            face_sample_interval_ms=int(face_sample_interval_ms),
            face_score_threshold=float(face_score_threshold),
            face_max_track_gap_ms=int(face_max_track_gap_ms),
            face_min_track_observations=int(face_min_track_observations),
            identity_similarity_threshold=float(identity_similarity_threshold),
        )
    except Exception as exc:
        LOGGER.exception("Face-gallery extraction failed")
        raise gr.Error(str(exc)) from exc

    status = (
        f"Extracted **{len(session.profiles)} profiles** from "
        f"**{len(session.findings)} face-track segments**. Review the gallery, "
        "choose the blur/keep policy, and render."
    )
    return (
        session,
        session.gallery_items(),
        gr.update(choices=session.labels, value=[]),
        json.dumps(session.public_summary(), indent=2),
        status,
    )


def preview_uploaded_photo_matches(
    session: FaceGallerySession | None,
    uploaded_photos,
    face_model_path: str,
    face_recognition_model_path: str,
    reference_match_threshold: float,
):
    if session is None:
        raise gr.Error("Extract face profiles before matching uploaded photos.")
    try:
        matches = match_uploaded_reference_photos(
            session,
            uploaded_photos,
            face_model_path=face_model_path,
            face_recognition_model_path=face_recognition_model_path,
            threshold=float(reference_match_threshold),
        )
    except Exception as exc:
        LOGGER.exception("Uploaded-photo matching failed")
        raise gr.Error(str(exc)) from exc
    return json.dumps(matches, indent=2)


def run_gallery_pipeline(
    video_path: str | None,
    session: FaceGallerySession | None,
    selected_labels: list[str] | None,
    gallery_action: str,
    uploaded_photos,
    uploaded_photo_action: str,
    api_base: str,
    model: str,
    chunk_seconds: float,
    deterministic_ocr: bool,
    detect_qr_codes: bool,
    deterministic_sample_interval_ms: int,
    face_model_path: str,
    face_recognition_model_path: str,
    reference_match_threshold: float,
    run_log_level: str,
    show_sensitive_values: bool,
    include_raw_model_output: bool,
):
    if not video_path:
        raise gr.Error("Upload an MP4 video first.")
    if session is None:
        raise gr.Error("Extract face profiles before rendering a manual selection.")

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
        "run_log_level": str(run_log_level),
        "include_sensitive_values_in_report": bool(show_sensitive_values),
        "include_raw_model_output": bool(include_raw_model_output),
    }

    try:
        result = analyze_video_with_face_gallery(
            video_path,
            session=session,
            selected_labels=selected_labels,
            gallery_action=gallery_action,  # type: ignore[arg-type]
            uploaded_photos=uploaded_photos,
            uploaded_photo_action=uploaded_photo_action,  # type: ignore[arg-type]
            face_model_path=face_model_path,
            face_recognition_model_path=face_recognition_model_path,
            reference_match_threshold=float(reference_match_threshold),
            **common_kwargs,
        )
    except Exception as exc:
        LOGGER.exception("Manual face-gallery render failed")
        raise gr.Error(str(exc)) from exc

    return _result_outputs(result, show_sensitive_values=bool(show_sensitive_values))


with gr.Blocks(title="FrameGuard", analytics_enabled=False) as frameguard_app:
    gallery_state = gr.State(value=None)

    gr.Markdown(
        """
# FrameGuard

FrameGuard analyzes video and audio for sensitive information and produces a
redacted MP4 with a JSON audit report.

For predictable face redaction, use **Manual face gallery**. FrameGuard extracts
representative face profiles, groups fragmented tracks with SFace, and lets the
user decide who is blurred or kept visible. Uploaded photos can also be matched
against the gallery. Profile crops and embeddings stay in server-side session
memory and are not written to the report or run log.
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
            label="Enable automatic face redaction",
            info="Used only by the automatic analysis button.",
        )

    with gr.Row():
        face_redaction_mode = gr.Radio(
            choices=[
                ("Blur every detected face", "all"),
                ("Blur only the uploaded reference face", "reference"),
                (
                    "Blur visually apparent children only (experimental)",
                    "likely_minors",
                ),
            ],
            value="all",
            type="value",
            label="Automatic face-redaction mode",
        )
        reference_face = gr.Image(
            label="Single reference face (automatic reference mode)",
            type="filepath",
            sources=["upload"],
            height=220,
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
            label="SFace uploaded-photo match threshold",
            info="Higher values are stricter.",
        )
        identity_similarity_threshold = gr.Slider(
            minimum=0.25,
            maximum=0.80,
            step=0.01,
            value=0.45,
            label="Face-gallery identity grouping threshold",
            info=(
                "Raise it when different people are merged. Lower it when the same "
                "person appears as several profiles."
            ),
        )
        face_sample_interval_ms = gr.Slider(
            minimum=100,
            maximum=1000,
            step=50,
            value=200,
            label="Face detection sampling interval in milliseconds",
        )
        face_score_threshold = gr.Slider(
            minimum=0.50,
            maximum=0.95,
            step=0.01,
            value=0.75,
            label="YuNet face confidence threshold",
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

        gr.Markdown("### Experimental child-classification policy")
        with gr.Row():
            child_minimum_confidence = gr.Slider(
                minimum=0.50,
                maximum=0.95,
                step=0.01,
                value=0.70,
                label="Minimum per-timestamp confidence",
            )
            child_max_samples_per_track = gr.Slider(
                minimum=3,
                maximum=8,
                step=1,
                value=5,
                label="Maximum timestamps per face track",
            )
        with gr.Row():
            child_minimum_usable_timestamps = gr.Slider(
                minimum=2,
                maximum=6,
                step=1,
                value=3,
                label="Minimum usable timestamps",
            )
            child_consensus_fraction = gr.Slider(
                minimum=0.50,
                maximum=1.00,
                step=0.05,
                value=0.70,
                label="Required child/adult consensus",
            )
        child_blur_uncertain = gr.Checkbox(
            value=False,
            label="Also blur uncertain classifications",
        )
        child_continue_on_error = gr.Checkbox(
            value=True,
            label="Continue when child classification fails",
        )

        run_log_level = gr.Dropdown(
            choices=["INFO", "DEBUG"],
            value="INFO",
            label="Run log detail",
        )
        show_sensitive_values = gr.Checkbox(
            value=False,
            label="Show exact detected values in UI and JSON report",
        )
        include_raw_model_output = gr.Checkbox(
            value=False,
            label="Include raw Qwen output in JSON report",
        )

    with gr.Accordion("Manual face gallery", open=True):
        gr.Markdown(
            """
1. Click **Extract face profiles**.
2. Review the profile gallery and select people.
3. Choose whether selected people are blurred or kept visible.
4. Optionally upload reference photos and choose whether their matches are
   blurred or kept visible.
5. Click **Render manual face selection**.
"""
        )
        extract_profiles_button = gr.Button("Extract face profiles")
        gallery_status = gr.Markdown()
        face_gallery = gr.Gallery(
            label="Detected face profiles",
            columns=5,
            rows=2,
            height="auto",
            object_fit="contain",
            preview=True,
        )
        gallery_choices = gr.CheckboxGroup(
            choices=[],
            label="Selected gallery people",
        )

        with gr.Row():
            gallery_action = gr.Radio(
                choices=[
                    ("Blur selected gallery people", "blur_selected"),
                    (
                        "Keep selected gallery people visible; blur everyone else",
                        "keep_selected_visible",
                    ),
                ],
                value="blur_selected",
                type="value",
                label="Gallery selection action",
            )
            uploaded_photo_action = gr.Radio(
                choices=[
                    ("Blur people matching uploaded photos", "blur"),
                    ("Keep people matching uploaded photos visible", "keep_visible"),
                ],
                value="blur",
                type="value",
                label="Uploaded-photo action",
            )

        uploaded_reference_faces = gr.File(
            label="Optional reference photos",
            file_count="multiple",
            file_types=["image"],
            type="filepath",
        )
        with gr.Row():
            preview_matches_button = gr.Button("Preview uploaded-photo matches")
            render_gallery_button = gr.Button(
                "Render manual face selection",
                variant="primary",
            )
        gallery_summary = gr.Code(label="Face-gallery summary", language="json")
        uploaded_match_preview = gr.Code(
            label="Uploaded-photo match preview",
            language="json",
        )

    automatic_button = gr.Button(
        "Run automatic analysis and redaction",
        variant="secondary",
    )

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

    extract_profiles_button.click(
        fn=extract_face_profiles,
        inputs=[
            input_video,
            face_model_path,
            face_recognition_model_path,
            face_sample_interval_ms,
            face_score_threshold,
            face_max_track_gap_ms,
            face_min_track_observations,
            identity_similarity_threshold,
        ],
        outputs=[
            gallery_state,
            face_gallery,
            gallery_choices,
            gallery_summary,
            gallery_status,
        ],
    )

    preview_matches_button.click(
        fn=preview_uploaded_photo_matches,
        inputs=[
            gallery_state,
            uploaded_reference_faces,
            face_model_path,
            face_recognition_model_path,
            reference_match_threshold,
        ],
        outputs=[uploaded_match_preview],
    )

    render_gallery_button.click(
        fn=run_gallery_pipeline,
        inputs=[
            input_video,
            gallery_state,
            gallery_choices,
            gallery_action,
            uploaded_reference_faces,
            uploaded_photo_action,
            api_base,
            model,
            chunk_seconds,
            deterministic_ocr,
            detect_qr_codes,
            deterministic_sample_interval_ms,
            face_model_path,
            face_recognition_model_path,
            reference_match_threshold,
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

    automatic_button.click(
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

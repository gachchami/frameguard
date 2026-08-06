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


def run_sensitive_content_pipeline(
    video_path: str | None,
    api_base: str,
    model: str,
    chunk_seconds: float,
    deterministic_ocr: bool,
    detect_qr_codes: bool,
    deterministic_sample_interval_ms: int,
    run_log_level: str,
    show_sensitive_values: bool,
    include_raw_model_output: bool,
):
    """Protect visual and spoken secrets without enabling face redaction."""

    if not video_path:
        raise gr.Error("Upload an MP4 video in the Sensitive Content section.")

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
            deterministic_sample_interval_ms=int(
                deterministic_sample_interval_ms
            ),
            redact_faces=False,
            run_log_level=str(run_log_level),
            include_sensitive_values_in_report=bool(show_sensitive_values),
            include_raw_model_output=bool(include_raw_model_output),
        )
    except Exception as exc:
        LOGGER.exception("Sensitive-content protection failed")
        raise gr.Error(str(exc)) from exc

    return _result_outputs(
        result,
        show_sensitive_values=bool(show_sensitive_values),
    )



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

    merged_count = max(0, len(session.findings) - len(session.profiles))
    status = (
        f"Found **{len(session.profiles)} unique people** from "
        f"**{len(session.findings)} face-track fragments**. "
        f"Merged **{merged_count} duplicate fragments**. "
        "Click a face card to select or deselect it."
    )
    return (
        session,
        session.gallery_items(selected_labels=[]),
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
        raise gr.Error("Detect people before creating the redacted video.")

    selected_labels = selected_labels or []
    has_uploaded_photos = bool(uploaded_photos)
    if not selected_labels and not has_uploaded_photos:
        raise gr.Error(
            "Choose at least one detected person or upload a reference photo "
            "before creating the redacted video."
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




def _ordered_selected_labels(
    session: FaceGallerySession,
    selected_labels: list[str] | None,
) -> list[str]:
    selected = set(selected_labels or [])
    return [
        profile.label
        for profile in session.profiles
        if profile.label in selected
    ]


def select_all_gallery_profiles(
    session: FaceGallerySession | None,
    gallery_action: str,
):
    if session is None:
        raise gr.Error("Detect people before selecting profiles.")
    selected = session.labels
    return (
        gr.update(value=selected),
        session.gallery_items(selected),
        describe_gallery_selection(session, selected, gallery_action),
    )


def clear_gallery_profiles(
    session: FaceGallerySession | None,
    gallery_action: str,
):
    if session is None:
        return (
            gr.update(value=[]),
            [],
            "No face profiles have been extracted yet.",
        )
    return (
        gr.update(value=[]),
        session.gallery_items([]),
        describe_gallery_selection(session, [], gallery_action),
    )


def toggle_gallery_profile(
    session: FaceGallerySession | None,
    selected_labels: list[str] | None,
    gallery_action: str,
    event: gr.SelectData,
):
    if session is None:
        raise gr.Error("Detect people before selecting profiles.")

    index = event.index[0] if isinstance(event.index, tuple) else int(event.index)
    if index < 0 or index >= len(session.profiles):
        raise gr.Error(f"Invalid face-gallery selection index: {index}")

    clicked_label = session.profiles[index].label
    selected = set(selected_labels or [])
    if clicked_label in selected:
        selected.remove(clicked_label)
    else:
        selected.add(clicked_label)

    ordered = _ordered_selected_labels(session, list(selected))
    return (
        gr.update(value=ordered),
        session.gallery_items(ordered),
        describe_gallery_selection(session, ordered, gallery_action),
    )


def describe_gallery_selection(
    session: FaceGallerySession | None,
    selected_labels: list[str] | None,
    gallery_action: str,
) -> str:
    if session is None:
        return "No face profiles have been extracted yet."

    selected = _ordered_selected_labels(session, selected_labels)
    selected_count = len(selected)
    total_count = len(session.profiles)
    selected_names = ", ".join(
        label.split("|", 1)[0].strip()
        for label in selected
    ) or "none"

    if gallery_action == "blur_selected":
        rule = (
            f"blur {selected_count} selected person"
            f"{'s' if selected_count != 1 else ''}; keep "
            f"{max(0, total_count - selected_count)} visible"
        )
    else:
        rule = (
            f"keep {selected_count} selected person"
            f"{'s' if selected_count != 1 else ''} visible; blur "
            f"{max(0, total_count - selected_count)}"
        )

    return (
        f"**Current rule:** {rule}.  \n"
        f"**Selected:** {selected_names}"
    )


def update_automatic_mode(mode: object):
    normalized = _normalize_face_mode(mode)
    return (
        gr.update(visible=normalized == "reference"),
        gr.update(visible=normalized == "likely_minors"),
    )

APP_CSS = """
.frameguard-hero {
    padding: 0.4rem 0 0.2rem 0;
}
.product-path {
    border: 1px solid var(--border-color-primary);
    border-radius: 14px;
    padding: 1rem;
    margin-bottom: 0.85rem;
}
.step-card {
    border: 1px solid var(--border-color-primary);
    border-radius: 12px;
    padding: 0.8rem;
    margin-bottom: 0.7rem;
}
.step-note {
    color: var(--body-text-color-subdued);
    font-size: 0.92rem;
}
.primary-path-note {
    border-left: 4px solid var(--color-accent);
    padding-left: 0.8rem;
}
.section-kicker {
    color: var(--body-text-color-subdued);
    margin-top: -0.35rem;
}
"""



with gr.Blocks(
    title="FrameGuard",
    analytics_enabled=False,
) as frameguard_app:
    gallery_state = gr.State(value=None)
    face_redaction_enabled = gr.State(value=True)
    face_secret_ocr_disabled = gr.State(value=False)
    face_secret_qr_disabled = gr.State(value=False)

    gr.Markdown(
        """
# FrameGuard

FrameGuard provides two independent video-privacy workflows:

1. **Sensitive Content** detects and redacts exposed secrets in video, text,
   QR codes, and audio.
2. **Face Privacy** detects people and blurs selected or automatically matched
   faces.
""",
        elem_classes=["frameguard-hero"],
    )

    with gr.Tabs():
        # ------------------------------------------------------------------
        # 1. Sensitive Content
        # ------------------------------------------------------------------
        with gr.Tab("Sensitive Content", id="sensitive-content"):
            gr.Markdown(
                """
## Protect secrets in video

Use this section for API keys, email addresses, IP addresses, account IDs,
phone numbers, private URLs, QR codes, and sensitive spoken information.
Face redaction is not enabled by this workflow.
""",
                elem_classes=["primary-path-note"],
            )

            with gr.Group(elem_classes=["product-path"]):
                gr.Markdown("### 1. Upload the source video")
                secrets_video = gr.Video(
                    label="Video containing sensitive content",
                    sources=["upload"],
                    format="mp4",
                )

            with gr.Group(elem_classes=["product-path"]):
                gr.Markdown("### 2. Choose the protection checks")
                with gr.Row():
                    secret_deterministic_ocr = gr.Checkbox(
                        value=True,
                        label="Detect visible sensitive text",
                        info=(
                            "Emails, IP addresses, API keys, account IDs, "
                            "phone numbers, and private URLs."
                        ),
                    )
                    secret_detect_qr_codes = gr.Checkbox(
                        value=True,
                        label="Detect and redact QR codes",
                    )
                gr.Markdown(
                    """
Qwen2.5-Omni analyzes the visual and audio streams for semantic privacy
findings. OCR and deterministic validators provide precise visible-text
localization.
""",
                    elem_classes=["step-note"],
                )

            with gr.Group(elem_classes=["product-path"]):
                gr.Markdown("### 3. Create the protected video")
                protect_secrets_button = gr.Button(
                    "Detect and redact sensitive content",
                    variant="primary",
                )

        # ------------------------------------------------------------------
        # 2. Face Privacy
        # ------------------------------------------------------------------
        with gr.Tab("Face Privacy", id="face-privacy"):
            gr.Markdown(
                """
## Blur faces in video

Use the reviewed face gallery for predictable control. FrameGuard detects faces
with YuNet, tracks appearances over time, and uses SFace to group fragmented
tracks that appear to belong to the same person.
""",
                elem_classes=["primary-path-note"],
            )

            with gr.Group(elem_classes=["product-path"]):
                gr.Markdown("### 1. Upload the source video")
                faces_video = gr.Video(
                    label="Video containing people",
                    sources=["upload"],
                    format="mp4",
                )

            with gr.Tabs():
                # ----------------------------------------------------------
                # Face Privacy: reviewed gallery
                # ----------------------------------------------------------
                with gr.Tab("Select People", id="face-select-people"):
                    gr.Markdown(
                        """
### Reviewed face selection

Detect unique people, click their profile cards, and choose whether selected
people should be blurred or kept visible.
"""
                    )

                    with gr.Group(elem_classes=["step-card"]):
                        gr.Markdown("### 2. Detect unique people")
                        extract_profiles_button = gr.Button(
                            "Detect people in this video",
                            variant="primary",
                        )
                        gallery_status = gr.Markdown(
                            "Upload a video, then detect people.",
                            elem_classes=["step-note"],
                        )

                    with gr.Group(elem_classes=["step-card"]):
                        gr.Markdown(
                            """
### 3. Select people

Click a face card to select or deselect it. Selected cards show a green border
and a visible **SELECTED** banner.
"""
                        )
                        face_gallery = gr.Gallery(
                            label=(
                                "Detected unique people — click cards to select"
                            ),
                            columns=4,
                            rows=2,
                            height=620,
                            object_fit="cover",
                            preview=True,
                            format="png",
                        )

                        gallery_action = gr.Radio(
                            choices=[
                                (
                                    "Blur the selected people",
                                    "blur_selected",
                                ),
                                (
                                    "Keep selected people visible and blur "
                                    "everyone else",
                                    "keep_selected_visible",
                                ),
                            ],
                            value="blur_selected",
                            type="value",
                            label="Face privacy rule",
                        )

                        gallery_choices = gr.CheckboxGroup(
                            choices=[],
                            label="Selected people",
                            visible=False,
                        )
                        with gr.Row():
                            select_all_button = gr.Button("Select all")
                            clear_selection_button = gr.Button(
                                "Clear selection"
                            )

                        selection_summary = gr.Markdown(
                            "No face profiles have been extracted yet.",
                            elem_classes=["step-note"],
                        )

                    with gr.Group(elem_classes=["step-card"]):
                        gr.Markdown(
                            """
### 4. Optional reference photos

Upload one or more clear photos. FrameGuard matches them against the detected
people and can either add the matches to the blur list or keep them visible.
"""
                        )
                        uploaded_reference_faces = gr.File(
                            label="Reference photos",
                            file_count="multiple",
                            file_types=["image"],
                            type="filepath",
                        )
                        uploaded_photo_action = gr.Radio(
                            choices=[
                                (
                                    "Blur people matching uploaded photos",
                                    "blur",
                                ),
                                (
                                    "Keep people matching uploaded photos "
                                    "visible",
                                    "keep_visible",
                                ),
                            ],
                            value="blur",
                            type="value",
                            label="Uploaded-photo rule",
                        )
                        preview_matches_button = gr.Button(
                            "Preview photo matches"
                        )
                        uploaded_match_preview = gr.Code(
                            label="Photo matches",
                            language="json",
                        )

                    with gr.Group(elem_classes=["step-card"]):
                        gr.Markdown("### 5. Create the face-protected video")
                        render_gallery_button = gr.Button(
                            "Blur selected faces",
                            variant="primary",
                        )

                    with gr.Accordion(
                        "Face-detection details",
                        open=False,
                    ):
                        gallery_summary = gr.Code(
                            label="Detected profile summary",
                            language="json",
                        )

                # ----------------------------------------------------------
                # Face Privacy: automatic rules
                # ----------------------------------------------------------
                with gr.Tab("Automatic Rules", id="face-automatic-rules"):
                    gr.Markdown(
                        """
### Automatic face redaction

Use these shortcuts when reviewed profile selection is unnecessary. The
child-only option remains experimental and is not a legal-age determination.
"""
                    )

                    face_redaction_mode = gr.Radio(
                        choices=[
                            ("Blur every detected face", "all"),
                            (
                                "Blur one uploaded reference face",
                                "reference",
                            ),
                            (
                                "Blur visually apparent children "
                                "(experimental)",
                                "likely_minors",
                            ),
                        ],
                        value="all",
                        type="value",
                        label="Automatic face rule",
                    )

                    with gr.Column(visible=False) as reference_face_panel:
                        reference_face = gr.Image(
                            label="Reference face",
                            type="filepath",
                            sources=["upload"],
                            height=240,
                        )

                    with gr.Column(visible=False) as child_policy_panel:
                        gr.Markdown(
                            """
**Experimental:** this mode makes a visual child/adult judgment from sampled
video evidence. Review its audit output carefully.
"""
                        )
                        with gr.Row():
                            child_minimum_confidence = gr.Slider(
                                minimum=0.50,
                                maximum=0.95,
                                step=0.01,
                                value=0.70,
                                label="Minimum timestamp confidence",
                            )
                            child_max_samples_per_track = gr.Slider(
                                minimum=3,
                                maximum=8,
                                step=1,
                                value=5,
                                label="Maximum timestamps per track",
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
                                label="Required temporal consensus",
                            )
                        child_blur_uncertain = gr.Checkbox(
                            value=False,
                            label="Also blur uncertain classifications",
                        )
                        child_continue_on_error = gr.Checkbox(
                            value=True,
                            label="Continue when classification fails",
                        )

                    automatic_button = gr.Button(
                        "Apply automatic face redaction",
                        variant="primary",
                    )

        # ------------------------------------------------------------------
        # 3. Results
        # ------------------------------------------------------------------
        with gr.Tab("Results", id="results"):
            gr.Markdown(
                """
## Protected output

The latest completed workflow appears here with its redacted video, findings,
audit report, metrics, and privacy-safe run log.
"""
            )
            with gr.Row():
                output_video = gr.Video(label="Protected video preview")
                with gr.Column():
                    redacted_download = gr.File(
                        label="Download protected MP4"
                    )
                    report_download = gr.File(
                        label="Download JSON audit report"
                    )
                    log_download = gr.File(
                        label="Download privacy-safe run log"
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
                label="Privacy findings",
            )

            with gr.Accordion("Metrics and audit preview", open=False):
                with gr.Row():
                    metrics = gr.Code(
                        label="Processing metrics",
                        language="json",
                    )
                    report_preview = gr.Code(
                        label="Audit report preview",
                        language="json",
                    )

        # ------------------------------------------------------------------
        # 4. Settings
        # ------------------------------------------------------------------
        with gr.Tab("Settings", id="settings"):
            gr.Markdown(
                """
## Processing settings

These settings are shared by the two product workflows. The defaults are
appropriate for the supplied models.
"""
            )

            with gr.Accordion("Qwen model connection", open=True):
                api_base = gr.Textbox(
                    label="vLLM API base",
                    value=DEFAULT_API_BASE,
                )
                model = gr.Textbox(
                    label="Model",
                    value=DEFAULT_MODEL,
                )
                chunk_seconds = gr.Slider(
                    minimum=3,
                    maximum=10,
                    step=1,
                    value=DEFAULT_CHUNK_SECONDS,
                    label="Qwen chunk length in seconds",
                )

            with gr.Accordion("Sensitive-content detection", open=True):
                deterministic_sample_interval_ms = gr.Slider(
                    minimum=200,
                    maximum=1200,
                    step=50,
                    value=350,
                    label="OCR and QR scan interval (ms)",
                )

            with gr.Accordion("Face detection and matching", open=True):
                face_model_path = gr.Textbox(
                    label="YuNet model path",
                    value=DEFAULT_FACE_MODEL,
                )
                face_recognition_model_path = gr.Textbox(
                    label="SFace model path",
                    value=DEFAULT_FACE_RECOGNITION_MODEL,
                )
                with gr.Row():
                    reference_match_threshold = gr.Slider(
                        minimum=0.20,
                        maximum=0.80,
                        step=0.01,
                        value=DEFAULT_COSINE_THRESHOLD,
                        label="Uploaded-photo match threshold",
                        info="Higher values require a closer SFace match.",
                    )
                    identity_similarity_threshold = gr.Slider(
                        minimum=0.25,
                        maximum=0.80,
                        step=0.01,
                        value=0.40,
                        label="Gallery deduplication threshold",
                        info=(
                            "Raise when different people merge; lower when "
                            "one person splits."
                        ),
                    )
                with gr.Row():
                    face_sample_interval_ms = gr.Slider(
                        minimum=100,
                        maximum=1000,
                        step=50,
                        value=200,
                        label="Face scan interval (ms)",
                    )
                    face_score_threshold = gr.Slider(
                        minimum=0.50,
                        maximum=0.95,
                        step=0.01,
                        value=0.75,
                        label="YuNet confidence threshold",
                    )
                with gr.Row():
                    face_max_track_gap_ms = gr.Slider(
                        minimum=300,
                        maximum=2000,
                        step=100,
                        value=900,
                        label="Maximum track gap (ms)",
                    )
                    face_min_track_observations = gr.Slider(
                        minimum=1,
                        maximum=5,
                        step=1,
                        value=2,
                        label="Minimum observations per track",
                    )

            with gr.Accordion("Reporting and diagnostics", open=False):
                run_log_level = gr.Dropdown(
                    choices=["INFO", "DEBUG"],
                    value="INFO",
                    label="Run-log detail",
                )
                show_sensitive_values = gr.Checkbox(
                    value=False,
                    label="Show exact detected values in UI and report",
                )
                include_raw_model_output = gr.Checkbox(
                    value=False,
                    label="Include raw Qwen output in the audit report",
                )

    # ----------------------------------------------------------------------
    # Sensitive-content events
    # ----------------------------------------------------------------------
    protect_secrets_button.click(
        fn=run_sensitive_content_pipeline,
        inputs=[
            secrets_video,
            api_base,
            model,
            chunk_seconds,
            secret_deterministic_ocr,
            secret_detect_qr_codes,
            deterministic_sample_interval_ms,
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

    # ----------------------------------------------------------------------
    # Face-gallery events
    # ----------------------------------------------------------------------
    extract_profiles_button.click(
        fn=extract_face_profiles,
        inputs=[
            faces_video,
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
    ).then(
        fn=describe_gallery_selection,
        inputs=[gallery_state, gallery_choices, gallery_action],
        outputs=[selection_summary],
    )

    face_gallery.select(
        fn=toggle_gallery_profile,
        inputs=[gallery_state, gallery_choices, gallery_action],
        outputs=[
            gallery_choices,
            face_gallery,
            selection_summary,
        ],
    )

    select_all_button.click(
        fn=select_all_gallery_profiles,
        inputs=[gallery_state, gallery_action],
        outputs=[
            gallery_choices,
            face_gallery,
            selection_summary,
        ],
    )

    clear_selection_button.click(
        fn=clear_gallery_profiles,
        inputs=[gallery_state, gallery_action],
        outputs=[
            gallery_choices,
            face_gallery,
            selection_summary,
        ],
    )

    gallery_action.change(
        fn=describe_gallery_selection,
        inputs=[gallery_state, gallery_choices, gallery_action],
        outputs=[selection_summary],
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
            faces_video,
            gallery_state,
            gallery_choices,
            gallery_action,
            uploaded_reference_faces,
            uploaded_photo_action,
            api_base,
            model,
            chunk_seconds,
            face_secret_ocr_disabled,
            face_secret_qr_disabled,
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

    # ----------------------------------------------------------------------
    # Automatic face events
    # ----------------------------------------------------------------------
    face_redaction_mode.change(
        fn=update_automatic_mode,
        inputs=[face_redaction_mode],
        outputs=[reference_face_panel, child_policy_panel],
    )

    automatic_button.click(
        fn=run_pipeline,
        inputs=[
            faces_video,
            api_base,
            model,
            chunk_seconds,
            face_secret_ocr_disabled,
            face_secret_qr_disabled,
            deterministic_sample_interval_ms,
            face_redaction_enabled,
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
        css=APP_CSS,
    )

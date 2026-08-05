from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .face_tracking import DEFAULT_FACE_MODEL, scan_face_tracks
from .minor_protection import (
    FaceRedactionMode,
    QwenAgeEstimator,
    classify_minor_face_tracks,
)
from .observability import RunEventRecorder
from .pipeline import PipelineResult, analyze_video, merge_findings
from .redact import render_redacted_video
from .schemas import Finding

_VALID_MODES = {"off", "all", "likely_minors"}


def normalize_face_redaction_mode(value: str | None) -> FaceRedactionMode:
    normalized = (value or "all").strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "none": "off",
        "disabled": "off",
        "blur_all": "all",
        "all_faces": "all",
        "likely_minor": "likely_minors",
        "minors": "likely_minors",
        "children": "likely_minors",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in _VALID_MODES:
        raise ValueError(
            f"Unknown face redaction mode {value!r}; expected off, all, or likely_minors"
        )
    return normalized  # type: ignore[return-value]


def _face_report(finding: Finding) -> dict[str, object]:
    return {
        "id": finding.id,
        "type": finding.type,
        "value": finding.value,
        "modality": finding.modality,
        "start_ms": finding.start_ms,
        "end_ms": finding.end_ms,
        "confidence": finding.confidence,
        "reason": finding.reason,
        "visual_location": finding.visual_location,
        "action": finding.action,
        "sources": finding.sources,
        "observations": [
            {
                "time_ms": item.time_ms,
                "x": item.x,
                "y": item.y,
                "width": item.width,
                "height": item.height,
                "confidence": item.confidence,
            }
            for item in finding.observations
        ],
    }


def _update_report(
    report_path: Path,
    *,
    mode: FaceRedactionMode,
    selected_face_findings: list[Finding],
    age_decisions: list[dict[str, object]],
    metrics: dict[str, object],
    configuration: dict[str, object],
) -> None:
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        payload = {}

    config_payload = payload.setdefault("configuration", {})
    if isinstance(config_payload, dict):
        config_payload.update(configuration)
        config_payload["face_redaction_mode"] = mode

    metrics_payload = payload.setdefault("metrics", {})
    if isinstance(metrics_payload, dict):
        metrics_payload.update(metrics)

    payload["age_estimation"] = {
        "purpose": "privacy-protective apparent-age triage",
        "warning": (
            "Apparent age is probabilistic and is not proof of legal age. "
            "Uncertain tracks are blurred by default."
        ),
        "decisions": age_decisions,
    }

    findings_payload = payload.setdefault("findings", [])
    if isinstance(findings_payload, list):
        findings_payload.extend(_face_report(item) for item in selected_face_findings)

    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def analyze_video_with_face_policy(
    input_path: str | Path,
    *,
    face_redaction_mode: str = "all",
    age_minor_boundary: int = 18,
    age_confident_adult_age: int = 22,
    age_minimum_confidence: float = 0.65,
    age_max_samples_per_track: int = 5,
    age_fail_closed: bool = True,
    age_blur_uncertain: bool = False,
    **pipeline_kwargs: Any,
) -> PipelineResult:
    """Run FrameGuard with off/all/likely-minors face-redaction policies.

    The existing pipeline remains the source of truth for semantic-secret, OCR,
    QR, audio, rendering, reporting, and normal all-face redaction behavior.
    Only the likely-minors mode adds a post-pipeline face-age triage stage.
    """

    mode = normalize_face_redaction_mode(face_redaction_mode)
    base_kwargs = dict(pipeline_kwargs)

    face_model_path = Path(base_kwargs.get("face_model_path", DEFAULT_FACE_MODEL))
    face_sample_interval_ms = max(50, int(base_kwargs.get("face_sample_interval_ms", 200)))
    face_score_threshold = float(base_kwargs.get("face_score_threshold", 0.75))
    face_max_track_gap_ms = max(100, int(base_kwargs.get("face_max_track_gap_ms", 900)))
    face_min_track_observations = max(
        1,
        int(base_kwargs.get("face_min_track_observations", 2)),
    )
    run_log_level = str(base_kwargs.get("run_log_level", "INFO"))

    if mode == "all":
        base_kwargs["redact_faces"] = True
        return analyze_video(input_path, **base_kwargs)

    base_kwargs["redact_faces"] = False
    result = analyze_video(input_path, **base_kwargs)
    if mode == "off":
        return result

    recorder = RunEventRecorder(
        result.log_path,
        run_id=result.run_id,
        level=run_log_level,
    )
    recorder.info(
        "minor_protection.started",
        face_model=face_model_path.name,
        minor_boundary=int(age_minor_boundary),
        confident_adult_age=int(age_confident_adult_age),
        minimum_confidence=float(age_minimum_confidence),
        max_samples_per_track=int(age_max_samples_per_track),
        fail_closed=bool(age_fail_closed),
        blur_uncertain=bool(age_blur_uncertain),
    )

    face_scan = scan_face_tracks(
        input_path,
        model_path=face_model_path,
        sample_interval_ms=face_sample_interval_ms,
        score_threshold=face_score_threshold,
        max_track_gap_ms=face_max_track_gap_ms,
        min_track_observations=face_min_track_observations,
        recorder=recorder,
    )

    estimator = QwenAgeEstimator(
        api_base=str(base_kwargs["api_base"]),
        model=str(base_kwargs["model"]),
        api_key=str(base_kwargs.get("api_key", "EMPTY")),
        minor_boundary=int(age_minor_boundary),
        confident_adult_age=int(age_confident_adult_age),
        minimum_confidence=float(age_minimum_confidence),
        fail_closed=bool(age_fail_closed),
        blur_uncertain=bool(age_blur_uncertain),
        recorder=recorder,
    )
    selected_face_findings, decisions = classify_minor_face_tracks(
        input_path,
        face_scan.findings,
        estimator=estimator,
        max_samples_per_track=max(1, int(age_max_samples_per_track)),
        recorder=recorder,
    )

    combined_findings = merge_findings([*result.findings, *selected_face_findings])
    render_redacted_video(
        input_path,
        result.output_video,
        combined_findings,
        recorder=recorder,
    )

    metrics = dict(result.metrics)
    metrics.update(
        {
            "face_redaction_mode": mode,
            "age_face_tracks": face_scan.tracks,
            "age_tracks_blurred": len(selected_face_findings),
            "age_likely_minor_tracks": sum(
                item.category == "likely_minor" for item in decisions
            ),
            "age_uncertain_tracks": sum(item.category == "uncertain" for item in decisions),
            "age_likely_adult_tracks": sum(
                item.category == "likely_adult" for item in decisions
            ),
            "age_face_scan_seconds": round(face_scan.elapsed_seconds, 4),
            "age_estimation_seconds": round(
                sum(item.elapsed_seconds for item in decisions),
                4,
            ),
            "output_bytes": result.output_video.stat().st_size,
        }
    )

    configuration = {
        "age_minor_boundary": int(age_minor_boundary),
        "age_confident_adult_age": int(age_confident_adult_age),
        "age_minimum_confidence": float(age_minimum_confidence),
        "age_max_samples_per_track": int(age_max_samples_per_track),
        "age_fail_closed": bool(age_fail_closed),
        "age_blur_uncertain": bool(age_blur_uncertain),
    }
    _update_report(
        result.report_path,
        mode=mode,
        selected_face_findings=selected_face_findings,
        age_decisions=[item.to_dict() for item in decisions],
        metrics=metrics,
        configuration=configuration,
    )
    recorder.info(
        "minor_protection.rendered",
        face_tracks=face_scan.tracks,
        blurred_tracks=len(selected_face_findings),
        output=result.output_video.name,
        report=result.report_path.name,
    )

    return PipelineResult(
        run_id=result.run_id,
        output_video=result.output_video,
        report_path=result.report_path,
        log_path=result.log_path,
        findings=combined_findings,
        metrics=metrics,
    )

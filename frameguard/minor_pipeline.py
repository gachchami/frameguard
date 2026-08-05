from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from .face_tracking import DEFAULT_FACE_MODEL, scan_face_tracks
from .minor_protection import (
    FaceRedactionMode,
    QwenChildClassifier,
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
        "visually_apparent_children": "likely_minors",
    }
    normalized = aliases.get(normalized, normalized)
    if normalized not in _VALID_MODES:
        raise ValueError(
            f"Unknown face redaction mode {value!r}; expected off, all, or likely_minors"
        )
    return cast(FaceRedactionMode, normalized)


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
    child_decisions: list[dict[str, object]],
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

    uncertain_action = (
        "blurred" if bool(configuration.get("child_blur_uncertain")) else "left visible"
    )
    payload["child_classification"] = {
        "purpose": "holistic visual child/adult privacy triage",
        "warning": (
            "Visual classification is probabilistic and does not establish legal age. "
            "Each decision combines marked full-scene, person-context, and face views "
            f"from multiple timestamps. Uncertain tracks are {uncertain_action}."
        ),
        "decisions": child_decisions,
    }

    findings_payload = payload.setdefault("findings", [])
    if isinstance(findings_payload, list):
        findings_payload.extend(_face_report(item) for item in selected_face_findings)

    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def analyze_video_with_face_policy(
    input_path: str | Path,
    *,
    face_redaction_mode: str = "all",
    child_minimum_confidence: float = 0.70,
    child_minimum_usable_timestamps: int = 3,
    child_consensus_fraction: float = 0.70,
    child_max_samples_per_track: int = 5,
    child_continue_on_error: bool = True,
    child_blur_uncertain: bool = False,
    **pipeline_kwargs: Any,
) -> PipelineResult:
    """Run FrameGuard with all-face or visually-apparent-child redaction."""

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
        "child_protection.started",
        face_model=face_model_path.name,
        minimum_confidence=float(child_minimum_confidence),
        minimum_usable_timestamps=int(child_minimum_usable_timestamps),
        consensus_fraction=float(child_consensus_fraction),
        max_samples_per_track=int(child_max_samples_per_track),
        continue_on_error=bool(child_continue_on_error),
        blur_uncertain=bool(child_blur_uncertain),
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

    classifier = QwenChildClassifier(
        api_base=str(base_kwargs["api_base"]),
        model=str(base_kwargs["model"]),
        api_key=str(base_kwargs.get("api_key", "EMPTY")),
        minimum_confidence=float(child_minimum_confidence),
        minimum_usable_timestamps=int(child_minimum_usable_timestamps),
        consensus_fraction=float(child_consensus_fraction),
        continue_on_error=bool(child_continue_on_error),
        blur_uncertain=bool(child_blur_uncertain),
        recorder=recorder,
    )
    selected_face_findings, decisions = classify_minor_face_tracks(
        input_path,
        face_scan.findings,
        estimator=classifier,
        max_samples_per_track=max(1, int(child_max_samples_per_track)),
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
            "child_face_tracks": face_scan.tracks,
            "child_tracks_blurred": len(selected_face_findings),
            "child_likely_child_tracks": sum(
                item.category == "likely_minor" for item in decisions
            ),
            "child_uncertain_tracks": sum(
                item.category == "uncertain" for item in decisions
            ),
            "child_likely_adult_tracks": sum(
                item.category == "likely_adult" for item in decisions
            ),
            "child_face_scan_seconds": round(face_scan.elapsed_seconds, 4),
            "child_classification_seconds": round(
                sum(item.elapsed_seconds for item in decisions),
                4,
            ),
            "child_total_usable_timestamps": sum(
                item.usable_timestamps for item in decisions
            ),
            "output_bytes": result.output_video.stat().st_size,
        }
    )

    configuration = {
        "child_minimum_confidence": float(child_minimum_confidence),
        "child_minimum_usable_timestamps": int(child_minimum_usable_timestamps),
        "child_consensus_fraction": float(child_consensus_fraction),
        "child_max_samples_per_track": int(child_max_samples_per_track),
        "child_continue_on_error": bool(child_continue_on_error),
        "child_blur_uncertain": bool(child_blur_uncertain),
    }
    _update_report(
        result.report_path,
        mode=mode,
        selected_face_findings=selected_face_findings,
        child_decisions=[item.to_dict() for item in decisions],
        metrics=metrics,
        configuration=configuration,
    )
    recorder.info(
        "child_protection.rendered",
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

from __future__ import annotations

import json
import tempfile
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .deterministic_detectors import DeterministicScanResult, scan_deterministic_findings
from .face_tracking import DEFAULT_FACE_MODEL, FaceScanResult, scan_face_tracks
from .multimodal_llm import DemoMockClient, QwenOmniClient
from .observability import RunEventRecorder, mask_value
from .redact import render_redacted_video
from .schemas import BoxObservation, Finding, ModelFinding
from .video import probe_video, split_video
from .visual_locator import localize_visual_finding, normalize_for_search


@dataclass(frozen=True, slots=True)
class PipelineResult:
    run_id: str
    output_video: Path
    report_path: Path
    log_path: Path
    findings: list[Finding]
    metrics: dict[str, object]


def _action_for_modality(modality: str) -> str:
    if modality == "both":
        return "blur_and_mute"
    if modality == "audio":
        return "mute"
    return "blur"


def _to_global_finding(model_finding: ModelFinding, offset_seconds: float) -> Finding:
    shifted = model_finding.shifted(offset_seconds)
    start_ms = int(shifted.start_seconds * 1000)
    end_ms = int(shifted.end_seconds * 1000)
    if shifted.modality in {"audio", "both"}:
        start_ms = max(0, start_ms - 200)
        end_ms += 250
    return Finding(
        id=f"finding_{uuid.uuid4().hex[:8]}",
        type=shifted.type,
        value=shifted.value,
        modality=shifted.modality,
        start_ms=start_ms,
        end_ms=max(start_ms, end_ms),
        confidence=shifted.confidence,
        reason=shifted.reason,
        visual_location=shifted.visual_location,
        action=_action_for_modality(shifted.modality),
        sources=["qwen"],
    )


def _merge_key(finding: Finding) -> tuple[str, str]:
    return finding.type.lower(), normalize_for_search(finding.value)


def _combined_modality(first: str, second: str) -> str:
    if first == second:
        return first
    return "both"


def _deduplicate_observations(observations: list[BoxObservation]) -> list[BoxObservation]:
    unique: dict[tuple[int, int, int, int, int], BoxObservation] = {}
    for observation in observations:
        key = (
            observation.time_ms,
            observation.x,
            observation.y,
            observation.width,
            observation.height,
        )
        previous = unique.get(key)
        if previous is None or observation.confidence > previous.confidence:
            unique[key] = observation
    return sorted(unique.values(), key=lambda item: item.time_ms)


def merge_findings(findings: list[Finding], max_gap_ms: int = 1200) -> list[Finding]:
    merged: list[Finding] = []
    for finding in sorted(findings, key=lambda item: (item.start_ms, item.end_ms)):
        candidate: Finding | None = None
        for existing in reversed(merged):
            if _merge_key(existing) != _merge_key(finding):
                continue
            if finding.start_ms <= existing.end_ms + max_gap_ms:
                candidate = existing
                break

        if candidate is None:
            finding.observations = _deduplicate_observations(finding.observations)
            finding.sources = sorted(set(finding.sources))
            merged.append(finding)
            continue

        candidate.start_ms = min(candidate.start_ms, finding.start_ms)
        candidate.end_ms = max(candidate.end_ms, finding.end_ms)
        candidate.confidence = max(candidate.confidence, finding.confidence)
        candidate.modality = _combined_modality(  # type: ignore[assignment]
            candidate.modality,
            finding.modality,
        )
        candidate.action = _action_for_modality(candidate.modality)
        candidate.sources = sorted(set([*candidate.sources, *finding.sources]))
        candidate.observations = _deduplicate_observations(
            [*candidate.observations, *finding.observations]
        )
        if finding.reason and finding.reason not in candidate.reason:
            candidate.reason = "; ".join(
                part for part in [candidate.reason, finding.reason] if part
            )
    return merged


def _empty_deterministic_scan() -> DeterministicScanResult:
    return DeterministicScanResult(
        findings=[],
        sampled_frames=0,
        elapsed_seconds=0.0,
        qr_observations=0,
        pattern_observations=0,
    )


def _finding_ref(finding: Finding, recorder: RunEventRecorder) -> dict[str, object]:
    return {
        "id": finding.id,
        "type": finding.type,
        "modality": finding.modality,
        "start_ms": finding.start_ms,
        "end_ms": finding.end_ms,
        "confidence": round(finding.confidence, 4),
        "sources": finding.sources,
        "value": recorder.value_ref(finding.value, kind=finding.type),
        "observations": len(finding.observations),
    }


def _report_finding(
    finding: Finding,
    recorder: RunEventRecorder,
    *,
    include_sensitive_values: bool,
) -> dict[str, object]:
    payload = finding.to_dict()
    payload["value_fingerprint"] = recorder.fingerprint(finding.value)
    payload["value_length"] = len(finding.value)
    payload["value_preview"] = mask_value(finding.value, finding.type)
    if not include_sensitive_values:
        payload.pop("value", None)
    return payload


def _model_response_report(
    *,
    chunk_name: str,
    start_seconds: float,
    duration_seconds: float,
    response,
    recorder: RunEventRecorder,
    include_raw_model_output: bool,
) -> dict[str, object]:
    result: dict[str, object] = {
        "chunk": chunk_name,
        "start_seconds": start_seconds,
        "duration_seconds": duration_seconds,
        "metadata": response.metadata,
        "raw_output_characters": len(response.raw_text),
        "parsed_findings": [
            {
                "type": item.type,
                "modality": item.modality,
                "start_seconds": item.start_seconds,
                "end_seconds": item.end_seconds,
                "confidence": item.confidence,
                "value_length": len(item.value),
                "value_fingerprint": recorder.fingerprint(item.value),
                "value_preview": mask_value(item.value, item.type),
            }
            for item in response.findings
        ],
    }
    if include_raw_model_output:
        result["raw_text"] = response.raw_text
    return result


def analyze_video(
    input_path: str | Path,
    *,
    api_base: str,
    model: str,
    api_key: str = "EMPTY",
    chunk_seconds: float = 5.0,
    output_dir: str | Path = "outputs",
    detector_mode: str = "qwen",
    deterministic_ocr: bool = True,
    detect_qr_codes: bool = True,
    deterministic_sample_interval_ms: int = 350,
    redact_faces: bool = True,
    face_model_path: str | Path = DEFAULT_FACE_MODEL,
    face_sample_interval_ms: int = 200,
    face_score_threshold: float = 0.75,
    face_max_track_gap_ms: int = 900,
    face_min_track_observations: int = 2,
    run_log_level: str = "INFO",
    include_sensitive_values_in_report: bool = False,
    include_raw_model_output: bool = False,
) -> PipelineResult:
    started = time.perf_counter()
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_id = uuid.uuid4().hex[:10]
    stem = input_path.stem.replace(" ", "_") or "video"
    log_path = output_dir / "logs" / f"{stem}_{run_id}.jsonl"
    recorder = RunEventRecorder(log_path, run_id=run_id, level=run_log_level)

    recorder.info(
        "pipeline.started",
        input=input_path.name,
        detector_mode=detector_mode,
        model=model,
        chunk_seconds=chunk_seconds,
        deterministic_ocr=deterministic_ocr,
        detect_qr_codes=detect_qr_codes,
        deterministic_sample_interval_ms=deterministic_sample_interval_ms,
        redact_faces=redact_faces,
        face_model=Path(face_model_path).name,
        face_sample_interval_ms=face_sample_interval_ms,
        face_score_threshold=face_score_threshold,
        face_max_track_gap_ms=face_max_track_gap_ms,
        face_min_track_observations=face_min_track_observations,
        include_sensitive_values_in_report=include_sensitive_values_in_report,
        include_raw_model_output=include_raw_model_output,
    )

    try:
        with recorder.stage("video.probe"):
            video_info = probe_video(input_path)
        recorder.info(
            "video.metadata",
            width=video_info.width,
            height=video_info.height,
            fps=round(video_info.fps, 4),
            frame_count=video_info.frame_count,
            duration_ms=video_info.duration_ms,
            input_bytes=input_path.stat().st_size,
        )

        if detector_mode == "mock":
            client = DemoMockClient(recorder=recorder)
            effective_model = "demo-mock-no-llm"
        elif detector_mode == "qwen":
            client = QwenOmniClient(
                api_base=api_base,
                model=model,
                api_key=api_key,
                recorder=recorder,
            )
            effective_model = model
        else:
            raise ValueError(f"Unknown detector mode: {detector_mode}")

        model_seconds = 0.0
        localization_seconds = 0.0
        raw_response_reports: list[dict[str, object]] = []
        global_findings: list[Finding] = []

        with tempfile.TemporaryDirectory(prefix="frameguard-chunks-") as temp_dir:
            with recorder.stage("video.chunking", chunk_seconds=chunk_seconds) as timer:
                chunks = split_video(input_path, temp_dir, chunk_seconds)
            chunking_seconds = timer.elapsed_seconds
            recorder.info(
                "video.chunks.created",
                chunk_count=len(chunks),
                durations_seconds=[round(chunk.duration_seconds, 3) for chunk in chunks],
            )

            for index, chunk in enumerate(chunks):
                recorder.info(
                    "chunk.analysis.started",
                    chunk_index=index,
                    chunk=chunk.path.name,
                    start_seconds=round(chunk.start_seconds, 3),
                    duration_seconds=round(chunk.duration_seconds, 3),
                )
                model_started = time.perf_counter()
                response = client.analyze_clip(chunk.path, chunk.duration_seconds)
                elapsed = time.perf_counter() - model_started
                model_seconds += elapsed

                converted = [
                    _to_global_finding(item, chunk.start_seconds) for item in response.findings
                ]
                global_findings.extend(converted)
                raw_response_reports.append(
                    _model_response_report(
                        chunk_name=chunk.path.name,
                        start_seconds=chunk.start_seconds,
                        duration_seconds=chunk.duration_seconds,
                        response=response,
                        recorder=recorder,
                        include_raw_model_output=include_raw_model_output,
                    )
                )
                recorder.info(
                    "chunk.analysis.completed",
                    chunk_index=index,
                    elapsed_seconds=round(elapsed, 4),
                    finding_count=len(converted),
                    findings=[_finding_ref(item, recorder) for item in converted],
                )

        deterministic_scan = _empty_deterministic_scan()
        if deterministic_ocr or detect_qr_codes:
            deterministic_scan = scan_deterministic_findings(
                input_path,
                sample_interval_ms=max(100, int(deterministic_sample_interval_ms)),
                deterministic_ocr=deterministic_ocr,
                detect_qr_codes=detect_qr_codes,
                recorder=recorder,
            )
            global_findings.extend(deterministic_scan.findings)
            recorder.info(
                "deterministic_findings.collected",
                finding_count=len(deterministic_scan.findings),
                findings=[
                    _finding_ref(item, recorder) for item in deterministic_scan.findings
                ],
            )

        face_scan: FaceScanResult | None = None
        if redact_faces:
            face_scan = scan_face_tracks(
                input_path,
                model_path=face_model_path,
                sample_interval_ms=max(50, int(face_sample_interval_ms)),
                score_threshold=float(face_score_threshold),
                max_track_gap_ms=max(100, int(face_max_track_gap_ms)),
                min_track_observations=max(1, int(face_min_track_observations)),
                recorder=recorder,
            )
            global_findings.extend(face_scan.findings)
            recorder.info(
                "face_findings.collected",
                finding_count=len(face_scan.findings),
                findings=[_finding_ref(item, recorder) for item in face_scan.findings],
            )

        premerge_count = len(global_findings)
        findings = merge_findings(global_findings)
        recorder.info(
            "findings.merged",
            before=premerge_count,
            after=len(findings),
            findings=[_finding_ref(item, recorder) for item in findings],
        )

        localization_started = time.perf_counter()
        for finding in findings:
            item_started = time.perf_counter()
            existing_observations = len(finding.observations)
            if finding.modality in {"visual", "both"} and not finding.observations:
                finding.observations = localize_visual_finding(str(input_path), finding)

            if finding.modality in {"visual", "both"} and finding.observations:
                finding.observations = _deduplicate_observations(finding.observations)
                finding.start_ms = max(
                    0,
                    min(finding.start_ms, finding.observations[0].time_ms - 250),
                )
                finding.end_ms = min(
                    video_info.duration_ms,
                    max(finding.end_ms, finding.observations[-1].time_ms + 500),
                )

            item_elapsed = time.perf_counter() - item_started
            recorder.debug(
                "finding.localization.completed",
                finding=_finding_ref(finding, recorder),
                preexisting_observations=existing_observations,
                elapsed_seconds=round(item_elapsed, 4),
            )
            if finding.modality in {"visual", "both"} and not finding.observations:
                recorder.warning(
                    "finding.localization.missed",
                    finding=_finding_ref(finding, recorder),
                )

        localization_seconds = time.perf_counter() - localization_started
        recorder.info(
            "localization.completed",
            elapsed_seconds=round(localization_seconds, 4),
            localized=sum(
                item.modality in {"visual", "both"} and bool(item.observations)
                for item in findings
            ),
            unlocalized=sum(
                item.modality in {"visual", "both"} and not item.observations
                for item in findings
            ),
        )

        output_video = output_dir / f"{stem}_redacted_{run_id}.mp4"
        report_path = output_dir / f"{stem}_report_{run_id}.json"

        render_started = time.perf_counter()
        render_redacted_video(input_path, output_video, findings, recorder=recorder)
        render_seconds = time.perf_counter() - render_started
        total_seconds = time.perf_counter() - started

        source_counts = Counter(source for item in findings for source in item.sources)
        type_counts = Counter(item.type for item in findings)
        localized_visual = sum(
            item.modality in {"visual", "both"} and bool(item.observations)
            for item in findings
        )
        unlocalized_visual = sum(
            item.modality in {"visual", "both"} and not item.observations
            for item in findings
        )

        metrics: dict[str, object] = {
            "run_id": run_id,
            "detector_mode": detector_mode,
            "model": effective_model,
            "video_duration_seconds": round(video_info.duration_ms / 1000.0, 2),
            "chunks_analyzed": len(raw_response_reports),
            "findings": len(findings),
            "finding_types": dict(sorted(type_counts.items())),
            "finding_sources": dict(sorted(source_counts.items())),
            "visual_findings": sum(item.modality in {"visual", "both"} for item in findings),
            "audio_findings": sum(item.modality in {"audio", "both"} for item in findings),
            "localized_visual_findings": localized_visual,
            "unlocalized_visual_findings": unlocalized_visual,
            "redaction_ready_visual_findings": localized_visual,
            "audio_redaction_intervals": sum(
                item.modality in {"audio", "both"} for item in findings
            ),
            "deterministic_frames_sampled": deterministic_scan.sampled_frames,
            "deterministic_pattern_observations": deterministic_scan.pattern_observations,
            "qr_observations": deterministic_scan.qr_observations,
            "face_detection_enabled": redact_faces,
            "face_frames_sampled": face_scan.sampled_frames if face_scan else 0,
            "face_detections": face_scan.detections if face_scan else 0,
            "face_tracks": face_scan.tracks if face_scan else 0,
            "face_rejected_tracks": face_scan.rejected_tracks if face_scan else 0,
            "face_scan_seconds": round(face_scan.elapsed_seconds, 2) if face_scan else 0.0,
            "chunking_seconds": round(chunking_seconds, 2),
            "model_seconds": round(model_seconds, 2),
            "deterministic_scan_seconds": round(deterministic_scan.elapsed_seconds, 2),
            "localization_seconds": round(localization_seconds, 2),
            "render_seconds": round(render_seconds, 2),
            "total_seconds": round(total_seconds, 2),
            "processing_ratio": round(
                (video_info.duration_ms / 1000.0) / total_seconds if total_seconds else 0.0,
                2,
            ),
        }

        report = {
            "schema_version": "1.1",
            "run_id": run_id,
            "input_video": input_path.name,
            "output_video": output_video.name,
            "privacy": {
                "contains_sensitive_values": include_sensitive_values_in_report,
                "contains_raw_model_output": include_raw_model_output,
                "log_contains_sensitive_values": False,
                "log_policy": (
                    "INFO and DEBUG logs contain counts, timings, types, lengths, and "
                    "per-run fingerprints only. They never contain detected values, prompts, "
                    "media data URIs, credentials, or raw model responses."
                ),
            },
            "configuration": {
                "detector_mode": detector_mode,
                "model": effective_model,
                "chunk_seconds": chunk_seconds,
                "deterministic_ocr": deterministic_ocr,
                "detect_qr_codes": detect_qr_codes,
                "deterministic_sample_interval_ms": deterministic_sample_interval_ms,
                "redact_faces": redact_faces,
                "face_model": Path(face_model_path).name,
                "face_sample_interval_ms": face_sample_interval_ms,
                "face_score_threshold": face_score_threshold,
                "face_max_track_gap_ms": face_max_track_gap_ms,
                "face_min_track_observations": face_min_track_observations,
                "run_log_level": run_log_level.upper(),
            },
            "metrics": metrics,
            "findings": [
                _report_finding(
                    item,
                    recorder,
                    include_sensitive_values=include_sensitive_values_in_report,
                )
                for item in findings
            ],
            "model_responses": raw_response_reports,
            "instrumentation": {
                "run_log": log_path.name,
                "event_count_at_report_write": recorder.event_count,
            },
        }
        report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

        recorder.info(
            "report.written",
            report=report_path.name,
            report_bytes=report_path.stat().st_size,
            contains_sensitive_values=include_sensitive_values_in_report,
            contains_raw_model_output=include_raw_model_output,
        )
        recorder.info(
            "pipeline.completed",
            total_seconds=round(total_seconds, 4),
            output=output_video.name,
            report=report_path.name,
            log=log_path.name,
            findings=len(findings),
            localized_visual_findings=localized_visual,
            unlocalized_visual_findings=unlocalized_visual,
        )

        return PipelineResult(
            run_id=run_id,
            output_video=output_video,
            report_path=report_path,
            log_path=log_path,
            findings=findings,
            metrics=metrics,
        )
    except Exception as exc:
        recorder.error(
            "pipeline.failed",
            exception_type=type(exc).__name__,
            error=exc,
            elapsed_seconds=round(time.perf_counter() - started, 4),
        )
        raise


def findings_table(
    findings: list[Finding],
    *,
    show_sensitive_values: bool = False,
) -> list[list[object]]:
    return [
        [
            item.type,
            item.value if show_sensitive_values else mask_value(item.value, item.type),
            item.modality,
            round(item.confidence, 2),
            round(item.start_ms / 1000.0, 2),
            round(item.end_ms / 1000.0, 2),
            len(item.observations),
            ", ".join(item.sources),
            item.action,
            item.reason,
        ]
        for item in findings
    ]

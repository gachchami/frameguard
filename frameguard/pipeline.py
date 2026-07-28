from __future__ import annotations

import json
import tempfile
import time
import uuid
from dataclasses import dataclass
from pathlib import Path

from .multimodal_llm import DemoMockClient, QwenOmniClient
from .redact import render_redacted_video
from .schemas import Finding, ModelFinding
from .video import probe_video, split_video
from .visual_locator import localize_visual_finding, normalize_for_search


@dataclass(frozen=True, slots=True)
class PipelineResult:
    output_video: Path
    report_path: Path
    findings: list[Finding]
    metrics: dict[str, object]


def _to_global_finding(model_finding: ModelFinding, offset_seconds: float) -> Finding:
    shifted = model_finding.shifted(offset_seconds)
    start_ms = int(shifted.start_seconds * 1000)
    end_ms = int(shifted.end_seconds * 1000)
    # Add conservative timing padding. Exact localization is a later refinement.
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
    )


def _merge_key(finding: Finding) -> tuple[str, str, str]:
    return (
        finding.type.lower(),
        finding.modality,
        normalize_for_search(finding.value),
    )


def merge_findings(findings: list[Finding], max_gap_ms: int = 1200) -> list[Finding]:
    merged: list[Finding] = []
    for finding in sorted(findings, key=lambda item: (item.start_ms, item.end_ms)):
        candidate = None
        for existing in reversed(merged):
            if _merge_key(existing) != _merge_key(finding):
                continue
            if finding.start_ms <= existing.end_ms + max_gap_ms:
                candidate = existing
                break
        if candidate is None:
            merged.append(finding)
            continue
        candidate.start_ms = min(candidate.start_ms, finding.start_ms)
        candidate.end_ms = max(candidate.end_ms, finding.end_ms)
        candidate.confidence = max(candidate.confidence, finding.confidence)
        if finding.reason and finding.reason not in candidate.reason:
            candidate.reason = "; ".join(part for part in [candidate.reason, finding.reason] if part)
    return merged


def analyze_video(
    input_path: str | Path,
    *,
    api_base: str,
    model: str,
    api_key: str = "EMPTY",
    chunk_seconds: float = 5.0,
    output_dir: str | Path = "outputs",
    detector_mode: str = "qwen",
) -> PipelineResult:
    started = time.perf_counter()
    input_path = Path(input_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    video_info = probe_video(input_path)
    if detector_mode == "mock":
        client = DemoMockClient()
        effective_model = "demo-mock-no-llm"
    elif detector_mode == "qwen":
        client = QwenOmniClient(api_base=api_base, model=model, api_key=api_key)
        effective_model = model
    else:
        raise ValueError(f"Unknown detector mode: {detector_mode}")

    model_seconds = 0.0
    raw_responses: list[dict[str, object]] = []
    global_findings: list[Finding] = []

    with tempfile.TemporaryDirectory(prefix="frameguard-chunks-") as temp_dir:
        chunks = split_video(input_path, temp_dir, chunk_seconds)
        for chunk in chunks:
            model_started = time.perf_counter()
            response = client.analyze_clip(chunk.path, chunk.duration_seconds)
            model_seconds += time.perf_counter() - model_started
            raw_responses.append(
                {
                    "chunk": chunk.path.name,
                    "start_seconds": chunk.start_seconds,
                    "duration_seconds": chunk.duration_seconds,
                    "raw_text": response.raw_text,
                }
            )
            global_findings.extend(
                _to_global_finding(item, chunk.start_seconds) for item in response.findings
            )

    findings = merge_findings(global_findings)

    localization_started = time.perf_counter()
    localization_interval_ms = 900 if detector_mode == "mock" else 350
    for finding in findings:
        finding.observations = localize_visual_finding(
            str(input_path),
            finding,
            sample_interval_ms=localization_interval_ms,
        )
        if finding.modality in {"visual", "both"} and finding.observations:
            # Expand the active period to cover every localized appearance.
            finding.start_ms = min(finding.start_ms, finding.observations[0].time_ms - 250)
            finding.end_ms = max(finding.end_ms, finding.observations[-1].time_ms + 500)
            finding.start_ms = max(0, finding.start_ms)
    localization_seconds = time.perf_counter() - localization_started

    stem = input_path.stem.replace(" ", "_")
    run_id = uuid.uuid4().hex[:8]
    output_video = output_dir / f"{stem}_redacted_{run_id}.mp4"
    report_path = output_dir / f"{stem}_report_{run_id}.json"

    render_started = time.perf_counter()
    render_redacted_video(input_path, output_video, findings)
    render_seconds = time.perf_counter() - render_started
    total_seconds = time.perf_counter() - started

    metrics: dict[str, object] = {
        "detector_mode": detector_mode,
        "model": effective_model,
        "video_duration_seconds": round(video_info.duration_ms / 1000.0, 2),
        "chunks_analyzed": len(raw_responses),
        "findings": len(findings),
        "visual_findings": sum(item.modality in {"visual", "both"} for item in findings),
        "audio_findings": sum(item.modality in {"audio", "both"} for item in findings),
        "localized_visual_findings": sum(bool(item.observations) for item in findings),
        "model_seconds": round(model_seconds, 2),
        "localization_seconds": round(localization_seconds, 2),
        "render_seconds": round(render_seconds, 2),
        "total_seconds": round(total_seconds, 2),
        "processing_ratio": round(
            (video_info.duration_ms / 1000.0) / total_seconds if total_seconds else 0.0,
            2,
        ),
    }
    report = {
        "input_video": str(input_path),
        "output_video": str(output_video),
        "metrics": metrics,
        "findings": [item.to_dict() for item in findings],
        "model_responses": raw_responses,
    }
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return PipelineResult(output_video, report_path, findings, metrics)


def findings_table(findings: list[Finding]) -> list[list[object]]:
    return [
        [
            item.type,
            item.value,
            item.modality,
            round(item.confidence, 2),
            round(item.start_ms / 1000.0, 2),
            round(item.end_ms / 1000.0, 2),
            len(item.observations),
            item.reason,
        ]
        for item in findings
    ]

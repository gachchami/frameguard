from __future__ import annotations

import json
import re
from typing import Any

from .schemas import ModelFinding

_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


def _extract_json_text(text: str) -> str:
    stripped = text.strip()
    fenced = _JSON_FENCE.search(stripped)
    if fenced:
        stripped = fenced.group(1).strip()

    array_start = stripped.find("[")
    object_start = stripped.find("{")
    starts = [position for position in (array_start, object_start) if position != -1]
    if not starts:
        raise ValueError("The model response did not contain JSON")

    start = min(starts)
    end = stripped.rfind("]" if stripped[start] == "[" else "}")
    if end == -1 or end < start:
        raise ValueError("The model response contained incomplete JSON")
    return stripped[start : end + 1]


def _number(value: Any, *, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _extract_raw_findings(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]

    if not isinstance(payload, dict):
        raise ValueError("The model JSON must be an object or array")

    wrapped = payload.get("findings")
    if isinstance(wrapped, list):
        return [item for item in wrapped if isinstance(item, dict)]

    if "type" in payload and "value" in payload:
        return [payload]

    return []


def _normalize_times(
    start_seconds: Any,
    end_seconds: Any,
    clip_duration_seconds: float,
) -> tuple[float, float]:
    duration = max(0.0, float(clip_duration_seconds))
    start = max(0.0, min(_number(start_seconds), duration))
    end = max(0.0, min(_number(end_seconds, default=duration), duration))

    # Qwen commonly emits 0 -> 0 when it detects an item but cannot estimate
    # frame-level timing. For privacy redaction, use the full current chunk.
    if end <= start:
        return 0.0, duration
    return start, end


def parse_model_findings(text: str, clip_duration_seconds: float) -> list[ModelFinding]:
    """Parse top-level arrays, wrapped objects, and fenced Qwen JSON safely."""

    payload = json.loads(_extract_json_text(text))
    raw_findings = _extract_raw_findings(payload)
    findings: list[ModelFinding] = []

    for raw in raw_findings:
        value = str(raw.get("value", "")).strip()
        if not value:
            continue

        kind = str(raw.get("type", "other")).strip().lower() or "other"
        modality = str(raw.get("modality", "visual")).strip().lower()
        if modality not in {"visual", "audio", "both"}:
            modality = "visual"

        start, end = _normalize_times(
            raw.get("start_seconds"),
            raw.get("end_seconds"),
            clip_duration_seconds,
        )
        confidence = min(1.0, max(0.0, _number(raw.get("confidence"), default=0.5)))
        location = raw.get("visual_location")

        findings.append(
            ModelFinding(
                type=kind,
                value=value,
                modality=modality,  # type: ignore[arg-type]
                start_seconds=start,
                end_seconds=end,
                confidence=confidence,
                reason=str(raw.get("reason", "")).strip(),
                visual_location=None if location is None else str(location).strip(),
            )
        )

    return findings

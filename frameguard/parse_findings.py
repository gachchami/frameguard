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

    first = stripped.find("{")
    last = stripped.rfind("}")
    if first == -1 or last == -1 or last < first:
        raise ValueError("The model response did not contain a JSON object")
    return stripped[first : last + 1]


def _number(value: Any, *, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def parse_model_findings(text: str, clip_duration_seconds: float) -> list[ModelFinding]:
    """Parse and defensively validate the model's JSON response."""

    payload = json.loads(_extract_json_text(text))
    raw_findings = payload.get("findings", [])
    if not isinstance(raw_findings, list):
        raise ValueError("The model JSON field 'findings' must be a list")

    findings: list[ModelFinding] = []
    for raw in raw_findings:
        if not isinstance(raw, dict):
            continue

        value = str(raw.get("value", "")).strip()
        kind = str(raw.get("type", "other")).strip().lower() or "other"
        modality = str(raw.get("modality", "visual")).strip().lower()
        if modality not in {"visual", "audio", "both"}:
            modality = "visual"

        start = max(0.0, _number(raw.get("start_seconds")))
        end = max(start, _number(raw.get("end_seconds"), default=start))
        start = min(start, clip_duration_seconds)
        end = min(max(start, end), clip_duration_seconds)
        confidence = min(1.0, max(0.0, _number(raw.get("confidence"), default=0.5)))

        # A secret without a value cannot be localized reliably in this MVP.
        if not value:
            continue

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

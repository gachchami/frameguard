from __future__ import annotations

import base64
import json
import math
import re
import time
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Literal, Protocol

import cv2
import httpx
import numpy as np

from .observability import RunEventRecorder
from .schemas import BoxObservation, Finding

AgeCategory = Literal["likely_minor", "uncertain", "likely_adult"]
FaceRedactionMode = Literal["off", "all", "likely_minors"]

_AGE_PROMPT = """
You are performing privacy-protective age-band estimation from multiple face crops
belonging to the same temporary video track. Do not identify the person. Do not
infer a name, ethnicity, health condition, or any other attribute.

Estimate one conservative apparent-age interval across all supplied images.
Return exactly one JSON object and no prose:
{
  "estimated_age_low": integer from 0 to 100,
  "estimated_age_high": integer from 0 to 100,
  "confidence": number from 0.0 to 1.0,
  "quality": "good" | "limited" | "poor",
  "reason_code": "clear_consistent" | "mixed_views" | "small_face" |
                 "occluded" | "blurred" | "insufficient_evidence"
}

Use a wider interval when uncertain. Age is an estimate, not proof. If the face
is too small, blurred, occluded, stylized, or inconsistent, set quality to poor
and confidence below 0.5.
""".strip()

_JSON_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.IGNORECASE | re.DOTALL)


@dataclass(frozen=True, slots=True)
class AgeDecision:
    track_id: str
    estimated_age_low: int | None
    estimated_age_high: int | None
    confidence: float
    quality: str
    category: AgeCategory
    blur: bool
    reason: str
    sample_count: int
    elapsed_seconds: float = 0.0

    @property
    def age_band(self) -> str:
        if self.estimated_age_low is None or self.estimated_age_high is None:
            return "unknown"
        return f"{self.estimated_age_low}-{self.estimated_age_high}"

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["age_band"] = self.age_band
        return payload


class AgeEstimator(Protocol):
    def estimate(self, crops: list[np.ndarray], *, track_id: str) -> AgeDecision: ...


def _message_content_text(content: object) -> str:
    """Normalize OpenAI-compatible message content to plain text.

    vLLM normally returns a string, but some compatible servers return a list
    of typed content blocks. Supporting both keeps the fail-closed policy from
    blurring every track merely because the response envelope changed.
    """

    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                text = item.get("text")
                if isinstance(text, str):
                    parts.append(text)
        if parts:
            return "\n".join(parts)
    raise ValueError("Age estimator response did not contain textual content")


def _extract_json_object(text: str) -> dict[str, object]:
    candidate = text.strip()
    fenced = _JSON_FENCE.search(candidate)
    if fenced:
        candidate = fenced.group(1).strip()

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        start = candidate.find("{")
        end = candidate.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("Age estimator did not return a JSON object")
        parsed = json.loads(candidate[start : end + 1])

    if not isinstance(parsed, dict):
        raise ValueError("Age estimator JSON must be an object")
    return parsed


def _bounded_int(value: object, *, low: int = 0, high: int = 100) -> int:
    number = int(round(float(value)))
    return max(low, min(high, number))


def _bounded_float(value: object, *, low: float = 0.0, high: float = 1.0) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError("Expected a finite confidence value")
    return max(low, min(high, number))


def decide_age_policy(
    *,
    track_id: str,
    estimated_age_low: int | None,
    estimated_age_high: int | None,
    confidence: float,
    quality: str,
    sample_count: int,
    minor_boundary: int = 18,
    confident_adult_age: int = 22,
    minimum_confidence: float = 0.65,
    elapsed_seconds: float = 0.0,
    failure_reason: str | None = None,
) -> AgeDecision:
    """Convert an uncertain age interval into a privacy-safe blur decision.

    Only a high-confidence interval whose lower edge is at least
    ``confident_adult_age`` is allowed to remain visible. Everything else is
    blurred. This is intentionally conservative around the 18-year boundary.
    """

    quality_normalized = str(quality or "poor").strip().lower()
    if quality_normalized not in {"good", "limited", "poor"}:
        quality_normalized = "poor"

    if failure_reason:
        return AgeDecision(
            track_id=track_id,
            estimated_age_low=None,
            estimated_age_high=None,
            confidence=0.0,
            quality="poor",
            category="uncertain",
            blur=True,
            reason=failure_reason,
            sample_count=sample_count,
            elapsed_seconds=elapsed_seconds,
        )

    if estimated_age_low is None or estimated_age_high is None:
        return AgeDecision(
            track_id=track_id,
            estimated_age_low=None,
            estimated_age_high=None,
            confidence=confidence,
            quality=quality_normalized,
            category="uncertain",
            blur=True,
            reason="missing_age_interval",
            sample_count=sample_count,
            elapsed_seconds=elapsed_seconds,
        )

    low = max(0, min(int(estimated_age_low), int(estimated_age_high)))
    high = min(100, max(int(estimated_age_low), int(estimated_age_high)))

    if quality_normalized == "poor" or confidence < minimum_confidence:
        category: AgeCategory = "uncertain"
        blur = True
        reason = "low_quality_or_confidence"
    elif high < minor_boundary:
        category = "likely_minor"
        blur = True
        reason = "estimated_interval_below_minor_boundary"
    elif low >= confident_adult_age:
        category = "likely_adult"
        blur = False
        reason = "high_confidence_interval_above_adult_safety_margin"
    else:
        category = "uncertain"
        blur = True
        reason = "interval_overlaps_minor_or_safety_margin"

    return AgeDecision(
        track_id=track_id,
        estimated_age_low=low,
        estimated_age_high=high,
        confidence=confidence,
        quality=quality_normalized,
        category=category,
        blur=blur,
        reason=reason,
        sample_count=sample_count,
        elapsed_seconds=elapsed_seconds,
    )


class QwenAgeEstimator:
    """Use the existing local Qwen2.5-Omni vLLM endpoint for age bands.

    Face crops are sent only to the local endpoint and are never persisted by
    this class. The response is reduced to a coarse age interval and a
    conservative blur decision.
    """

    def __init__(
        self,
        *,
        api_base: str,
        model: str,
        api_key: str = "EMPTY",
        timeout_seconds: float = 90.0,
        minor_boundary: int = 18,
        confident_adult_age: int = 22,
        minimum_confidence: float = 0.65,
        fail_closed: bool = True,
        recorder: RunEventRecorder | None = None,
    ) -> None:
        self.api_base = api_base.rstrip("/")
        self.model = model
        self.api_key = api_key
        self.timeout_seconds = max(5.0, float(timeout_seconds))
        self.minor_boundary = int(minor_boundary)
        self.confident_adult_age = int(confident_adult_age)
        self.minimum_confidence = float(minimum_confidence)
        self.fail_closed = bool(fail_closed)
        self.recorder = recorder

    @staticmethod
    def _image_part(crop: np.ndarray) -> dict[str, object]:
        ok, encoded = cv2.imencode(
            ".jpg",
            crop,
            [cv2.IMWRITE_JPEG_QUALITY, 88],
        )
        if not ok:
            raise ValueError("Could not encode a face crop")
        payload = base64.b64encode(encoded.tobytes()).decode("ascii")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{payload}"},
        }

    def estimate(self, crops: list[np.ndarray], *, track_id: str) -> AgeDecision:
        started = time.perf_counter()
        usable = [crop for crop in crops if crop.size and min(crop.shape[:2]) >= 32]
        if not usable:
            return decide_age_policy(
                track_id=track_id,
                estimated_age_low=None,
                estimated_age_high=None,
                confidence=0.0,
                quality="poor",
                sample_count=0,
                minor_boundary=self.minor_boundary,
                confident_adult_age=self.confident_adult_age,
                minimum_confidence=self.minimum_confidence,
                elapsed_seconds=time.perf_counter() - started,
                failure_reason="no_usable_face_crops",
            )

        content: list[dict[str, object]] = [self._image_part(crop) for crop in usable]
        content.append({"type": "text", "text": _AGE_PROMPT})
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": 0.0,
            "top_p": 1.0,
            "max_tokens": 220,
            "seed": 42,
            "repetition_penalty": 1.02,
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        if self.recorder:
            self.recorder.info(
                "age_estimation.requested",
                track_id=track_id,
                sample_count=len(usable),
                model=self.model,
            )

        try:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                response = client.post(
                    f"{self.api_base}/chat/completions",
                    headers=headers,
                    json=payload,
                )
                response.raise_for_status()
                body = response.json()

            content_value = body["choices"][0]["message"]["content"]
            text = _message_content_text(content_value)
            parsed = _extract_json_object(text)
            low = _bounded_int(parsed["estimated_age_low"])
            high = _bounded_int(parsed["estimated_age_high"])
            confidence = _bounded_float(parsed.get("confidence", 0.0))
            quality = str(parsed.get("quality", "poor"))
            decision = decide_age_policy(
                track_id=track_id,
                estimated_age_low=low,
                estimated_age_high=high,
                confidence=confidence,
                quality=quality,
                sample_count=len(usable),
                minor_boundary=self.minor_boundary,
                confident_adult_age=self.confident_adult_age,
                minimum_confidence=self.minimum_confidence,
                elapsed_seconds=time.perf_counter() - started,
            )
        except Exception as exc:
            if not self.fail_closed:
                raise
            decision = decide_age_policy(
                track_id=track_id,
                estimated_age_low=None,
                estimated_age_high=None,
                confidence=0.0,
                quality="poor",
                sample_count=len(usable),
                minor_boundary=self.minor_boundary,
                confident_adult_age=self.confident_adult_age,
                minimum_confidence=self.minimum_confidence,
                elapsed_seconds=time.perf_counter() - started,
                failure_reason=f"estimator_failure:{type(exc).__name__}",
            )

        if self.recorder:
            self.recorder.info(
                "age_estimation.completed",
                track_id=track_id,
                age_band=decision.age_band,
                confidence=round(decision.confidence, 4),
                quality=decision.quality,
                category=decision.category,
                blur=decision.blur,
                reason=decision.reason,
                sample_count=decision.sample_count,
                elapsed_seconds=round(decision.elapsed_seconds, 4),
            )
        return decision


def _select_observations(
    observations: list[BoxObservation],
    *,
    max_samples: int,
) -> list[BoxObservation]:
    ordered = sorted(observations, key=lambda item: item.time_ms)
    if len(ordered) <= max_samples:
        return ordered

    # Split the track into temporal windows, selecting the clearest/largest box
    # from each. This gives diversity without blindly using adjacent frames.
    selected: list[BoxObservation] = []
    window_size = len(ordered) / max_samples
    for index in range(max_samples):
        start = int(round(index * window_size))
        end = int(round((index + 1) * window_size))
        window = ordered[start : max(start + 1, end)]
        selected.append(
            max(
                window,
                key=lambda item: (
                    item.confidence,
                    item.width * item.height,
                ),
            )
        )
    return selected


def _crop_with_padding(
    frame: np.ndarray,
    observation: BoxObservation,
    *,
    padding_ratio: float,
    max_edge: int,
) -> np.ndarray | None:
    frame_height, frame_width = frame.shape[:2]
    pad_x = int(round(observation.width * padding_ratio))
    pad_y = int(round(observation.height * padding_ratio))
    x1 = max(0, observation.x - pad_x)
    y1 = max(0, observation.y - pad_y)
    x2 = min(frame_width, observation.x + observation.width + pad_x)
    y2 = min(frame_height, observation.y + observation.height + pad_y)
    if x2 <= x1 or y2 <= y1:
        return None

    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    height, width = crop.shape[:2]
    longest = max(height, width)
    if longest > max_edge:
        scale = max_edge / longest
        crop = cv2.resize(
            crop,
            (max(1, int(round(width * scale))), max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    return crop


def extract_track_crops(
    video_path: str | Path,
    finding: Finding,
    *,
    max_samples: int = 5,
    padding_ratio: float = 0.35,
    max_edge: int = 448,
) -> list[np.ndarray]:
    observations = _select_observations(
        finding.observations,
        max_samples=max(1, int(max_samples)),
    )
    if not observations:
        return []

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video for age estimation: {video_path}")

    crops: list[np.ndarray] = []
    try:
        for observation in observations:
            capture.set(cv2.CAP_PROP_POS_MSEC, float(observation.time_ms))
            ok, frame = capture.read()
            if not ok:
                continue
            crop = _crop_with_padding(
                frame,
                observation,
                padding_ratio=max(0.0, float(padding_ratio)),
                max_edge=max(96, int(max_edge)),
            )
            if crop is not None:
                crops.append(crop)
    finally:
        capture.release()
    return crops


def classify_minor_face_tracks(
    video_path: str | Path,
    face_findings: list[Finding],
    *,
    estimator: AgeEstimator,
    max_samples_per_track: int = 5,
    recorder: RunEventRecorder | None = None,
) -> tuple[list[Finding], list[AgeDecision]]:
    """Return only face tracks that must be blurred under the minor policy."""

    selected: list[Finding] = []
    decisions: list[AgeDecision] = []

    for finding in face_findings:
        crops = extract_track_crops(
            video_path,
            finding,
            max_samples=max_samples_per_track,
        )
        decision = estimator.estimate(crops, track_id=finding.value)
        decisions.append(decision)
        if not decision.blur:
            continue

        finding_type = (
            "likely_minor_face"
            if decision.category == "likely_minor"
            else "age_uncertain_face"
        )
        selected.append(
            replace(
                finding,
                type=finding_type,
                reason=(
                    "Face track selected by privacy-safe apparent-age policy; "
                    f"category={decision.category}; age_band={decision.age_band}; "
                    f"reason={decision.reason}"
                ),
                sources=sorted(set([*finding.sources, "qwen_age_estimation"])),
            )
        )

    if recorder:
        recorder.info(
            "minor_protection.completed",
            face_tracks=len(face_findings),
            blurred_tracks=len(selected),
            likely_minors=sum(item.category == "likely_minor" for item in decisions),
            uncertain=sum(item.category == "uncertain" for item in decisions),
            likely_adults=sum(item.category == "likely_adult" for item in decisions),
        )
    return selected, decisions

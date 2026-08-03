from __future__ import annotations

import ipaddress
import re
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pytesseract
from pytesseract import Output

from .observability import RunEventRecorder
from .schemas import BoxObservation, Finding
from .video import iter_frames_between, probe_video
from .visual_locator import normalize_for_search


@dataclass(frozen=True, slots=True)
class DeterministicScanResult:
    findings: list[Finding]
    sampled_frames: int
    elapsed_seconds: float
    qr_observations: int
    pattern_observations: int


@dataclass(frozen=True, slots=True)
class TextPatternMatch:
    type: str
    value: str
    start: int
    end: int
    confidence: float


@dataclass(slots=True)
class OCRToken:
    text: str
    confidence: float
    x: int
    y: int
    width: int
    height: int


_EMAIL = re.compile(
    r"(?<![A-Za-z0-9._%+-])"
    r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
    r"(?![A-Za-z0-9._%+-])"
)
_IPV4 = re.compile(r"(?<!\d)(?:\d{1,3}\.){3}\d{1,3}(?!\d)")
_PHONE = re.compile(r"(?<!\w)(?:\+?\d[\d\s().-]{6,}\d)(?!\w)")
_URL = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)
_ACCOUNT_ID = re.compile(
    r"\b(?:CUST|CUSTOMER|ACC|ACCOUNT|CLIENT|USER|EMP|EMPLOYEE|TICKET|CASE)-[A-Z0-9]{4,}\b",
    re.IGNORECASE,
)
_API_KEY_PATTERNS = (
    re.compile(r"\bsk-[A-Za-z0-9_-]{8,}\b"),
    re.compile(r"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(r"\bxox[baprs]-[A-Za-z0-9-]{10,}\b"),
)


def _valid_ipv4(value: str) -> bool:
    try:
        return isinstance(ipaddress.ip_address(value), ipaddress.IPv4Address)
    except ValueError:
        return False


def _looks_private_url(value: str) -> bool:
    lowered = value.lower().rstrip(".,;:)")
    host_match = re.match(r"https?://([^/:?#]+)", lowered)
    if host_match is None:
        return False
    host = host_match.group(1)
    if host in {"localhost", "127.0.0.1"}:
        return True
    if host.endswith((".local", ".internal", ".corp", ".lan")):
        return True
    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        return False
    return address.is_private or address.is_loopback or address.is_link_local


def _reasonable_phone(value: str) -> bool:
    stripped = value.strip()
    if _valid_ipv4(stripped):
        return False
    digits = re.sub(r"\D", "", stripped)
    return 8 <= len(digits) <= 15


def detect_pattern_matches(text: str) -> list[TextPatternMatch]:
    matches: list[TextPatternMatch] = []

    for match in _EMAIL.finditer(text):
        matches.append(TextPatternMatch("email", match.group(0), match.start(), match.end(), 0.99))

    for match in _IPV4.finditer(text):
        value = match.group(0)
        if _valid_ipv4(value):
            matches.append(TextPatternMatch("ip_address", value, match.start(), match.end(), 0.99))

    for pattern in _API_KEY_PATTERNS:
        for match in pattern.finditer(text):
            matches.append(TextPatternMatch("api_key", match.group(0), match.start(), match.end(), 0.99))

    for match in _ACCOUNT_ID.finditer(text):
        matches.append(TextPatternMatch("account_id", match.group(0), match.start(), match.end(), 0.97))

    for match in _URL.finditer(text):
        value = match.group(0).rstrip(".,;:")
        if _looks_private_url(value):
            matches.append(
                TextPatternMatch(
                    "private_url",
                    value,
                    match.start(),
                    match.start() + len(value),
                    0.96,
                )
            )

    for match in _PHONE.finditer(text):
        value = match.group(0).strip()
        if _reasonable_phone(value):
            matches.append(TextPatternMatch("phone", value, match.start(), match.end(), 0.93))

    deduplicated: dict[tuple[str, str, int, int], TextPatternMatch] = {}
    for item in matches:
        key = (item.type, normalize_for_search(item.value), item.start, item.end)
        deduplicated[key] = item
    return list(deduplicated.values())


def _preprocess_for_ocr(frame: np.ndarray) -> tuple[np.ndarray, float]:
    scale = 1.5
    enlarged = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB), scale


def _ocr_lines(
    frame: np.ndarray,
    *,
    min_confidence: float = 20.0,
) -> tuple[list[list[OCRToken]], float]:
    image, scale = _preprocess_for_ocr(frame)
    data = pytesseract.image_to_data(
        image,
        output_type=Output.DICT,
        config="--oem 3 --psm 11",
    )

    lines: dict[tuple[int, int, int], list[OCRToken]] = defaultdict(list)
    for index, raw_text in enumerate(data["text"]):
        text = str(raw_text).strip()
        try:
            confidence = float(data["conf"][index])
        except (TypeError, ValueError):
            confidence = -1.0
        if not text or confidence < min_confidence:
            continue
        key = (
            int(data["block_num"][index]),
            int(data["par_num"][index]),
            int(data["line_num"][index]),
        )
        lines[key].append(
            OCRToken(
                text=text,
                confidence=confidence,
                x=int(data["left"][index]),
                y=int(data["top"][index]),
                width=int(data["width"][index]),
                height=int(data["height"][index]),
            )
        )

    ordered: list[list[OCRToken]] = []
    for tokens in lines.values():
        tokens.sort(key=lambda token: token.x)
        ordered.append(tokens)
    return ordered, scale


def _text_and_spans(
    tokens: list[OCRToken],
    separator: str,
) -> tuple[str, list[tuple[int, int, OCRToken]]]:
    parts: list[str] = []
    spans: list[tuple[int, int, OCRToken]] = []
    cursor = 0
    for index, token in enumerate(tokens):
        if index:
            parts.append(separator)
            cursor += len(separator)
        start = cursor
        parts.append(token.text)
        cursor += len(token.text)
        spans.append((start, cursor, token))
    return "".join(parts), spans


def _box_for_match(
    match: TextPatternMatch,
    spans: list[tuple[int, int, OCRToken]],
    scale: float,
    time_ms: int,
) -> BoxObservation | None:
    overlapping = [
        token
        for start, end, token in spans
        if not (end <= match.start or start >= match.end)
    ]
    if not overlapping:
        return None

    x1 = min(token.x for token in overlapping)
    y1 = min(token.y for token in overlapping)
    x2 = max(token.x + token.width for token in overlapping)
    y2 = max(token.y + token.height for token in overlapping)
    confidence = sum(token.confidence for token in overlapping) / len(overlapping) / 100.0
    return BoxObservation(
        time_ms=time_ms,
        x=int(x1 / scale),
        y=int(y1 / scale),
        width=max(1, int((x2 - x1) / scale)),
        height=max(1, int((y2 - y1) / scale)),
        confidence=min(1.0, max(0.0, confidence)),
    )


def _detect_ocr_patterns(
    frame: np.ndarray,
    time_ms: int,
) -> list[tuple[TextPatternMatch, BoxObservation]]:
    lines, scale = _ocr_lines(frame)
    results: list[tuple[TextPatternMatch, BoxObservation]] = []
    seen: set[tuple[str, str, int, int, int, int]] = set()

    for tokens in lines:
        # The compact form catches punctuation split into separate OCR tokens.
        for separator in (" ", ""):
            text, spans = _text_and_spans(tokens, separator)
            for match in detect_pattern_matches(text):
                observation = _box_for_match(match, spans, scale, time_ms)
                if observation is None:
                    continue
                key = (
                    match.type,
                    normalize_for_search(match.value),
                    observation.x,
                    observation.y,
                    observation.width,
                    observation.height,
                )
                if key in seen:
                    continue
                seen.add(key)
                results.append((match, observation))
    return results


def _points_to_box(points: np.ndarray, time_ms: int) -> BoxObservation | None:
    array = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if array.size == 0:
        return None
    x1 = int(np.floor(array[:, 0].min()))
    y1 = int(np.floor(array[:, 1].min()))
    x2 = int(np.ceil(array[:, 0].max()))
    y2 = int(np.ceil(array[:, 1].max()))
    if x2 <= x1 or y2 <= y1:
        return None
    return BoxObservation(time_ms, x1, y1, x2 - x1, y2 - y1, 0.98)


def _detect_qr_codes(
    frame: np.ndarray,
    time_ms: int,
    detector: cv2.QRCodeDetector,
) -> list[tuple[str, BoxObservation]]:
    results: list[tuple[str, BoxObservation]] = []
    try:
        detected, decoded_values, points, _ = detector.detectAndDecodeMulti(frame)
    except (cv2.error, ValueError):
        detected, decoded_values, points = False, (), None

    if detected and points is not None:
        values = list(decoded_values)
        for index, polygon in enumerate(points):
            observation = _points_to_box(polygon, time_ms)
            if observation is None:
                continue
            decoded = values[index].strip() if index < len(values) else ""
            results.append((decoded or "QR code", observation))
        return results

    try:
        decoded, points, _ = detector.detectAndDecode(frame)
    except cv2.error:
        return results
    if points is not None:
        observation = _points_to_box(points, time_ms)
        if observation is not None:
            results.append((decoded.strip() or "QR code", observation))
    return results


def _iou(first: BoxObservation, second: BoxObservation) -> float:
    x1 = max(first.x, second.x)
    y1 = max(first.y, second.y)
    x2 = min(first.x + first.width, second.x + second.width)
    y2 = min(first.y + first.height, second.y + second.height)
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    if not intersection:
        return 0.0
    union = first.width * first.height + second.width * second.height - intersection
    return intersection / union if union else 0.0


def _merge_observation(
    findings: list[Finding],
    *,
    kind: str,
    value: str,
    observation: BoxObservation,
    confidence: float,
    reason: str,
    sample_interval_ms: int,
    spatial_tracking: bool = False,
) -> None:
    normalized = normalize_for_search(value)
    candidate: Finding | None = None

    for existing in reversed(findings):
        if existing.type != kind:
            continue
        if observation.time_ms - existing.end_ms > sample_interval_ms * 3:
            continue
        if existing.observations and existing.observations[-1].time_ms == observation.time_ms:
            continue

        same_value = normalize_for_search(existing.value) == normalized
        overlaps = bool(existing.observations) and _iou(existing.observations[-1], observation) >= 0.15
        if same_value or (spatial_tracking and overlaps):
            candidate = existing
            break

    window_ms = max(300, sample_interval_ms // 2 + 100)
    start_ms = max(0, observation.time_ms - window_ms)
    end_ms = observation.time_ms + window_ms

    if candidate is None:
        label = value
        if spatial_tracking and value == "QR code":
            sequence = 1 + sum(item.type == kind for item in findings)
            label = f"QR code {sequence}"
        findings.append(
            Finding(
                id=f"finding_{uuid.uuid4().hex[:8]}",
                type=kind,
                value=label,
                modality="visual",
                start_ms=start_ms,
                end_ms=end_ms,
                confidence=confidence,
                reason=reason,
                visual_location="deterministic detector",
                observations=[observation],
                action="blur",
                sources=["ocr_regex" if kind != "qr_code" else "qr_detector"],
            )
        )
        return

    candidate.start_ms = min(candidate.start_ms, start_ms)
    candidate.end_ms = max(candidate.end_ms, end_ms)
    candidate.confidence = max(candidate.confidence, confidence)
    candidate.observations.append(observation)


def scan_deterministic_findings(
    video_path: str | Path,
    *,
    sample_interval_ms: int = 350,
    deterministic_ocr: bool = True,
    detect_qr_codes: bool = True,
    recorder: RunEventRecorder | None = None,
) -> DeterministicScanResult:
    started = time.perf_counter()
    info = probe_video(video_path)
    findings: list[Finding] = []
    sampled_frames = 0
    qr_observations = 0
    pattern_observations = 0
    qr_detector = cv2.QRCodeDetector() if detect_qr_codes else None

    if recorder:
        recorder.info(
            "deterministic_scan.started",
            sample_interval_ms=sample_interval_ms,
            deterministic_ocr=deterministic_ocr,
            detect_qr_codes=detect_qr_codes,
        )

    for time_ms, frame in iter_frames_between(
        video_path,
        0,
        info.duration_ms,
        interval_ms=max(100, int(sample_interval_ms)),
    ):
        sampled_frames += 1
        frame_pattern_count = 0
        frame_qr_count = 0

        if deterministic_ocr:
            for match, observation in _detect_ocr_patterns(frame, time_ms):
                pattern_observations += 1
                frame_pattern_count += 1
                _merge_observation(
                    findings,
                    kind=match.type,
                    value=match.value,
                    observation=observation,
                    confidence=min(match.confidence, max(0.5, observation.confidence)),
                    reason="Detected by deterministic OCR pattern scan",
                    sample_interval_ms=sample_interval_ms,
                )

        if qr_detector is not None:
            for decoded, observation in _detect_qr_codes(frame, time_ms, qr_detector):
                qr_observations += 1
                frame_qr_count += 1
                _merge_observation(
                    findings,
                    kind="qr_code",
                    value=decoded,
                    observation=observation,
                    confidence=observation.confidence,
                    reason="QR code selected for privacy redaction",
                    sample_interval_ms=sample_interval_ms,
                    spatial_tracking=decoded == "QR code",
                )

        if recorder:
            recorder.debug(
                "deterministic_scan.frame",
                time_ms=time_ms,
                pattern_matches=frame_pattern_count,
                qr_matches=frame_qr_count,
            )

    for finding in findings:
        finding.observations.sort(key=lambda item: item.time_ms)

    elapsed = time.perf_counter() - started
    result = DeterministicScanResult(
        findings=findings,
        sampled_frames=sampled_frames,
        elapsed_seconds=elapsed,
        qr_observations=qr_observations,
        pattern_observations=pattern_observations,
    )

    if recorder:
        recorder.info(
            "deterministic_scan.completed",
            elapsed_seconds=round(elapsed, 4),
            sampled_frames=sampled_frames,
            finding_count=len(findings),
            finding_types=sorted({item.type for item in findings}),
            pattern_observations=pattern_observations,
            qr_observations=qr_observations,
        )

    return result

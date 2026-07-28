from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
import re

import cv2
import numpy as np
import pytesseract
from pytesseract import Output

from .schemas import BoxObservation, Finding
from .video import iter_frames_between


@dataclass(slots=True)
class OCRToken:
    text: str
    confidence: float
    x: int
    y: int
    width: int
    height: int


def normalize_for_search(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.lower())


def _preprocess(frame: np.ndarray) -> tuple[np.ndarray, float]:
    scale = 1.5
    enlarged = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
    gray = cv2.cvtColor(enlarged, cv2.COLOR_BGR2GRAY)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB), scale


def _tokens_by_line(frame: np.ndarray, min_confidence: float = 20.0) -> tuple[list[list[OCRToken]], float]:
    image, scale = _preprocess(frame)
    data = pytesseract.image_to_data(image, output_type=Output.DICT, config="--oem 3 --psm 11")
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
    ordered = []
    for tokens in lines.values():
        tokens.sort(key=lambda token: token.x)
        ordered.append(tokens)
    return ordered, scale


def _locate_in_tokens(tokens: list[OCRToken], target: str, scale: float) -> BoxObservation | None:
    normalized_target = normalize_for_search(target)
    if len(normalized_target) < 3:
        return None

    joined = ""
    spans: list[tuple[int, int, OCRToken]] = []
    for token in tokens:
        normalized = normalize_for_search(token.text)
        if not normalized:
            continue
        start = len(joined)
        joined += normalized
        spans.append((start, len(joined), token))

    start_index = joined.find(normalized_target)
    if start_index == -1:
        return None
    end_index = start_index + len(normalized_target)
    overlapping = [
        token for start, end, token in spans if not (end <= start_index or start >= end_index)
    ]
    if not overlapping:
        return None

    x1 = min(token.x for token in overlapping)
    y1 = min(token.y for token in overlapping)
    x2 = max(token.x + token.width for token in overlapping)
    y2 = max(token.y + token.height for token in overlapping)
    confidence = sum(token.confidence for token in overlapping) / len(overlapping) / 100.0
    return BoxObservation(
        time_ms=0,
        x=int(x1 / scale),
        y=int(y1 / scale),
        width=max(1, int((x2 - x1) / scale)),
        height=max(1, int((y2 - y1) / scale)),
        confidence=min(1.0, max(0.0, confidence)),
    )


def locate_value_in_frame(frame: np.ndarray, value: str) -> BoxObservation | None:
    lines, scale = _tokens_by_line(frame)
    for tokens in lines:
        located = _locate_in_tokens(tokens, value, scale)
        if located is not None:
            return located
    return None


def localize_visual_finding(
    video_path: str,
    finding: Finding,
    *,
    sample_interval_ms: int = 350,
    padding_ms: int = 700,
) -> list[BoxObservation]:
    if finding.modality not in {"visual", "both"}:
        return []

    observations: list[BoxObservation] = []
    start_ms = max(0, finding.start_ms - padding_ms)
    end_ms = finding.end_ms + padding_ms
    for time_ms, frame in iter_frames_between(
        video_path,
        start_ms,
        end_ms,
        interval_ms=sample_interval_ms,
    ):
        observation = locate_value_in_frame(frame, finding.value)
        if observation is None:
            continue
        observation.time_ms = time_ms
        observations.append(observation)
    return observations

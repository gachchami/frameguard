from __future__ import annotations

import math
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from .observability import RunEventRecorder
from .schemas import BoxObservation, Finding
from .video import probe_video

DEFAULT_FACE_MODEL = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "face_detection_yunet_2023mar.onnx"
)


@dataclass(frozen=True, slots=True)
class FaceDetection:
    x: int
    y: int
    width: int
    height: int
    confidence: float

    def to_observation(self, time_ms: int) -> BoxObservation:
        return BoxObservation(
            time_ms=time_ms,
            x=self.x,
            y=self.y,
            width=self.width,
            height=self.height,
            confidence=self.confidence,
        )


@dataclass(frozen=True, slots=True)
class FaceScanResult:
    findings: list[Finding]
    sampled_frames: int
    detections: int
    tracks: int
    rejected_tracks: int
    elapsed_seconds: float
    model_path: Path


class FaceDetector(Protocol):
    def detect(self, frame: np.ndarray) -> list[FaceDetection]: ...


class YuNetFaceDetector:
    """Thin OpenCV-DNN wrapper around the pretrained YuNet face detector."""

    def __init__(
        self,
        model_path: str | Path,
        *,
        score_threshold: float = 0.75,
        nms_threshold: float = 0.3,
        top_k: int = 5000,
    ) -> None:
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise FileNotFoundError(
                "YuNet face model is missing: "
                f"{self.model_path}. Run scripts/download_yunet_model.py on an "
                "internet-connected machine, commit models/face_detection_yunet_2023mar.onnx, "
                "and pull it into the target container."
            )
        if not hasattr(cv2, "FaceDetectorYN"):
            raise RuntimeError(
                "This OpenCV build does not provide cv2.FaceDetectorYN. "
                "Install opencv-python-headless>=4.10."
            )

        self._detector = cv2.FaceDetectorYN.create(
            model=str(self.model_path),
            config="",
            input_size=(320, 320),
            score_threshold=float(score_threshold),
            nms_threshold=float(nms_threshold),
            top_k=int(top_k),
        )
        self._input_size = (320, 320)

    def detect(self, frame: np.ndarray) -> list[FaceDetection]:
        height, width = frame.shape[:2]
        input_size = (width, height)
        if input_size != self._input_size:
            self._detector.setInputSize(input_size)
            self._input_size = input_size

        _, raw_faces = self._detector.detect(frame)
        if raw_faces is None:
            return []

        detections: list[FaceDetection] = []
        for row in raw_faces:
            x, y, box_width, box_height = (float(value) for value in row[:4])
            confidence = float(row[-1])

            x1 = max(0, min(width - 1, int(round(x))))
            y1 = max(0, min(height - 1, int(round(y))))
            x2 = max(x1 + 1, min(width, int(round(x + box_width))))
            y2 = max(y1 + 1, min(height, int(round(y + box_height))))

            detections.append(
                FaceDetection(
                    x=x1,
                    y=y1,
                    width=x2 - x1,
                    height=y2 - y1,
                    confidence=max(0.0, min(1.0, confidence)),
                )
            )
        return detections


def _box_iou(first: FaceDetection, second: FaceDetection) -> float:
    first_x2 = first.x + first.width
    first_y2 = first.y + first.height
    second_x2 = second.x + second.width
    second_y2 = second.y + second.height

    intersection_x1 = max(first.x, second.x)
    intersection_y1 = max(first.y, second.y)
    intersection_x2 = min(first_x2, second_x2)
    intersection_y2 = min(first_y2, second_y2)

    intersection_width = max(0, intersection_x2 - intersection_x1)
    intersection_height = max(0, intersection_y2 - intersection_y1)
    intersection = intersection_width * intersection_height
    if intersection == 0:
        return 0.0

    union = (
        first.width * first.height
        + second.width * second.height
        - intersection
    )
    return intersection / union if union else 0.0


def _normalized_center_distance(
    first: FaceDetection,
    second: FaceDetection,
    frame_width: int,
    frame_height: int,
) -> float:
    first_center = (first.x + first.width / 2.0, first.y + first.height / 2.0)
    second_center = (second.x + second.width / 2.0, second.y + second.height / 2.0)
    distance = math.hypot(
        first_center[0] - second_center[0],
        first_center[1] - second_center[1],
    )
    frame_diagonal = max(1.0, math.hypot(frame_width, frame_height))
    return distance / frame_diagonal


def _size_ratio(first: FaceDetection, second: FaceDetection) -> float:
    first_area = max(1, first.width * first.height)
    second_area = max(1, second.width * second.height)
    return max(first_area, second_area) / min(first_area, second_area)


@dataclass(slots=True)
class _FaceTrack:
    track_number: int
    observations: list[BoxObservation] = field(default_factory=list)

    @property
    def last_observation(self) -> BoxObservation:
        return self.observations[-1]

    @property
    def last_detection(self) -> FaceDetection:
        observation = self.last_observation
        return FaceDetection(
            x=observation.x,
            y=observation.y,
            width=observation.width,
            height=observation.height,
            confidence=observation.confidence,
        )

    def add(self, detection: FaceDetection, time_ms: int) -> None:
        self.observations.append(detection.to_observation(time_ms))


class FaceTracker:
    """Associate sampled-frame detections into stable, non-biometric tracks.

    This tracker uses geometry only. It does not compute face embeddings and does
    not identify people. A track ID means "the same moving box", not identity.
    """

    def __init__(
        self,
        *,
        frame_width: int,
        frame_height: int,
        max_gap_ms: int = 900,
        iou_threshold: float = 0.08,
        center_distance_threshold: float = 0.18,
        max_size_ratio: float = 3.0,
    ) -> None:
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.max_gap_ms = max(1, int(max_gap_ms))
        self.iou_threshold = float(iou_threshold)
        self.center_distance_threshold = float(center_distance_threshold)
        self.max_size_ratio = float(max_size_ratio)
        self._tracks: list[_FaceTrack] = []
        self._next_track_number = 1

    @property
    def tracks(self) -> list[_FaceTrack]:
        return self._tracks

    def update(self, time_ms: int, detections: list[FaceDetection]) -> None:
        active_tracks = [
            track
            for track in self._tracks
            if time_ms - track.last_observation.time_ms <= self.max_gap_ms
        ]

        candidates: list[tuple[float, int, int]] = []
        for track_index, track in enumerate(active_tracks):
            previous = track.last_detection
            for detection_index, detection in enumerate(detections):
                iou = _box_iou(previous, detection)
                center_distance = _normalized_center_distance(
                    previous,
                    detection,
                    self.frame_width,
                    self.frame_height,
                )
                size_ratio = _size_ratio(previous, detection)
                if size_ratio > self.max_size_ratio:
                    continue
                if (
                    iou < self.iou_threshold
                    and center_distance > self.center_distance_threshold
                ):
                    continue

                center_score = max(
                    0.0,
                    1.0 - center_distance / max(self.center_distance_threshold, 1e-6),
                )
                score = iou * 0.75 + center_score * 0.25
                candidates.append((score, track_index, detection_index))

        candidates.sort(reverse=True)
        matched_tracks: set[int] = set()
        matched_detections: set[int] = set()

        for _, track_index, detection_index in candidates:
            if track_index in matched_tracks or detection_index in matched_detections:
                continue
            active_tracks[track_index].add(detections[detection_index], time_ms)
            matched_tracks.add(track_index)
            matched_detections.add(detection_index)

        for detection_index, detection in enumerate(detections):
            if detection_index in matched_detections:
                continue
            track = _FaceTrack(track_number=self._next_track_number)
            self._next_track_number += 1
            track.add(detection, time_ms)
            self._tracks.append(track)


def _track_to_finding(
    track: _FaceTrack,
    *,
    duration_ms: int,
    sample_interval_ms: int,
) -> Finding:
    observations = sorted(track.observations, key=lambda item: item.time_ms)
    confidence = sum(item.confidence for item in observations) / len(observations)
    return Finding(
        id=f"finding_{uuid.uuid4().hex[:8]}",
        type="face",
        value=f"face_{track.track_number:03d}",
        modality="visual",
        start_ms=max(0, observations[0].time_ms - sample_interval_ms),
        end_ms=min(duration_ms, observations[-1].time_ms + sample_interval_ms),
        confidence=confidence,
        reason="Detected by YuNet neural face detector and associated across sampled frames",
        visual_location="tracked face bounding box",
        observations=observations,
        action="blur",
        sources=["yunet"],
    )


def scan_face_tracks(
    video_path: str | Path,
    *,
    model_path: str | Path = DEFAULT_FACE_MODEL,
    sample_interval_ms: int = 200,
    score_threshold: float = 0.75,
    max_track_gap_ms: int = 900,
    min_track_observations: int = 2,
    keep_single_detection_threshold: float = 0.93,
    recorder: RunEventRecorder | None = None,
    detector: FaceDetector | None = None,
) -> FaceScanResult:
    """Detect faces on sampled frames and associate them into temporal tracks."""

    started = time.perf_counter()
    video_path = Path(video_path)
    model_path = Path(model_path)
    info = probe_video(video_path)
    interval_ms = max(50, int(sample_interval_ms))
    frame_step = max(1, round(info.fps * interval_ms / 1000.0))

    if recorder:
        recorder.info(
            "face_scan.started",
            detector="yunet",
            model=model_path.name,
            sample_interval_ms=interval_ms,
            score_threshold=score_threshold,
            max_track_gap_ms=max_track_gap_ms,
            min_track_observations=min_track_observations,
        )

    effective_detector = detector or YuNetFaceDetector(
        model_path,
        score_threshold=score_threshold,
    )
    tracker = FaceTracker(
        frame_width=info.width,
        frame_height=info.height,
        max_gap_ms=max_track_gap_ms,
    )

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video for face detection: {video_path}")

    sampled_frames = 0
    detection_count = 0
    frame_index = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_index % frame_step != 0:
                frame_index += 1
                continue

            time_ms = min(
                info.duration_ms,
                int(round(frame_index / info.fps * 1000.0)),
            )
            detections = effective_detector.detect(frame)
            sampled_frames += 1
            detection_count += len(detections)
            tracker.update(time_ms, detections)

            if recorder:
                recorder.debug(
                    "face_scan.frame",
                    frame_index=frame_index,
                    time_ms=time_ms,
                    detections=len(detections),
                    active_tracks=sum(
                        time_ms - track.last_observation.time_ms <= max_track_gap_ms
                        for track in tracker.tracks
                    ),
                    total_tracks=len(tracker.tracks),
                )
            frame_index += 1
    finally:
        capture.release()

    accepted_tracks: list[_FaceTrack] = []
    rejected_tracks = 0
    for track in tracker.tracks:
        if len(track.observations) >= max(1, int(min_track_observations)):
            accepted_tracks.append(track)
            continue
        peak_confidence = max(item.confidence for item in track.observations)
        if peak_confidence >= keep_single_detection_threshold:
            accepted_tracks.append(track)
        else:
            rejected_tracks += 1

    findings = [
        _track_to_finding(
            track,
            duration_ms=info.duration_ms,
            sample_interval_ms=interval_ms,
        )
        for track in accepted_tracks
    ]
    elapsed = time.perf_counter() - started

    if recorder:
        recorder.info(
            "face_scan.completed",
            elapsed_seconds=round(elapsed, 4),
            sampled_frames=sampled_frames,
            detections=detection_count,
            tracks=len(findings),
            rejected_tracks=rejected_tracks,
            track_observation_counts=[len(item.observations) for item in findings],
        )

    return FaceScanResult(
        findings=findings,
        sampled_frames=sampled_frames,
        detections=detection_count,
        tracks=len(findings),
        rejected_tracks=rejected_tracks,
        elapsed_seconds=elapsed,
        model_path=model_path,
    )

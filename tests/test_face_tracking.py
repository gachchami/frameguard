from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from frameguard.face_tracking import (
    FaceDetection,
    FaceTracker,
    scan_face_tracks,
)
from frameguard.schemas import BoxObservation, Finding


def test_face_tracker_keeps_one_moving_face_in_one_track() -> None:
    tracker = FaceTracker(
        frame_width=640,
        frame_height=360,
        max_gap_ms=700,
    )

    tracker.update(0, [FaceDetection(100, 80, 90, 110, 0.96)])
    tracker.update(200, [FaceDetection(108, 82, 92, 111, 0.95)])
    tracker.update(400, [FaceDetection(118, 84, 91, 112, 0.94)])

    assert len(tracker.tracks) == 1
    assert len(tracker.tracks[0].observations) == 3


def test_face_tracker_separates_two_faces() -> None:
    tracker = FaceTracker(frame_width=640, frame_height=360)

    tracker.update(
        0,
        [
            FaceDetection(40, 60, 80, 100, 0.95),
            FaceDetection(430, 65, 82, 102, 0.94),
        ],
    )
    tracker.update(
        200,
        [
            FaceDetection(48, 62, 81, 101, 0.96),
            FaceDetection(422, 66, 84, 104, 0.93),
        ],
    )

    assert len(tracker.tracks) == 2
    assert sorted(len(track.observations) for track in tracker.tracks) == [2, 2]


def test_face_tracker_does_not_reuse_track_across_scene_cut() -> None:
    tracker = FaceTracker(frame_width=640, frame_height=360)

    tracker.update(0, [FaceDetection(100, 80, 80, 100, 0.96)])
    tracker.update(200, [FaceDetection(108, 82, 80, 100, 0.95)])
    tracker.start_new_scene(400)
    tracker.update(400, [FaceDetection(110, 84, 80, 100, 0.97)])

    assert len(tracker.tracks) == 2
    assert [len(track.observations) for track in tracker.tracks] == [2, 1]


def test_face_tracker_rejects_impossible_jump_during_gradual_transition() -> None:
    tracker = FaceTracker(frame_width=1280, frame_height=720)

    tracker.update(4167, [FaceDetection(641, 249, 33, 58, 0.92)])
    tracker.update(4583, [FaceDetection(537, 180, 42, 55, 0.92)])

    assert len(tracker.tracks) == 2


def test_finding_interpolates_observations() -> None:
    finding = Finding(
        id="finding_test",
        type="face",
        value="face_001",
        modality="visual",
        start_ms=0,
        end_ms=1000,
        confidence=0.9,
        observations=[
            BoxObservation(0, 100, 50, 80, 100, 0.9),
            BoxObservation(1000, 200, 100, 100, 120, 1.0),
        ],
    )

    observation = finding.observation_at(500)

    assert observation is not None
    assert observation.x == 150
    assert observation.y == 75
    assert observation.width == 90
    assert observation.height == 110


class _MovingFaceDetector:
    def __init__(self) -> None:
        self.calls = 0

    def detect(self, frame: np.ndarray) -> list[FaceDetection]:
        del frame
        offset = self.calls * 4
        self.calls += 1
        return [FaceDetection(40 + offset, 30, 50, 60, 0.97)]


def _write_test_video(path: Path) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        10.0,
        (160, 120),
    )
    assert writer.isOpened()
    try:
        for _ in range(10):
            writer.write(np.zeros((120, 160, 3), dtype=np.uint8))
    finally:
        writer.release()


def test_scan_face_tracks_accepts_injected_detector(tmp_path: Path) -> None:
    video_path = tmp_path / "video.mp4"
    _write_test_video(video_path)

    result = scan_face_tracks(
        video_path,
        detector=_MovingFaceDetector(),
        sample_interval_ms=200,
        min_track_observations=2,
    )

    assert result.sampled_frames == 5
    assert result.detections == 5
    assert result.tracks == 1
    assert len(result.findings) == 1
    assert result.findings[0].type == "face"
    assert result.findings[0].value == "face_001"
    assert len(result.findings[0].observations) == 5


class _TwoFaceDetector:
    def detect(self, frame: np.ndarray) -> list[FaceDetection]:
        del frame
        return [
            FaceDetection(20, 30, 40, 50, 0.98),
            FaceDetection(100, 30, 40, 50, 0.97),
        ]


class _LeftFaceMatcher:
    def matches(
        self,
        frame: np.ndarray,
        detection: FaceDetection,
    ) -> tuple[bool, float]:
        del frame
        return detection.x < 50, 0.8 if detection.x < 50 else 0.1


def test_reference_mode_tracks_only_matching_face(tmp_path: Path) -> None:
    video_path = tmp_path / "video.mp4"
    _write_test_video(video_path)

    result = scan_face_tracks(
        video_path,
        detector=_TwoFaceDetector(),
        matcher=_LeftFaceMatcher(),
        redaction_mode="reference",
        sample_interval_ms=200,
        min_track_observations=2,
    )

    assert result.detections == 10
    assert result.reference_candidates == 10
    assert result.reference_matches == 5
    assert result.reference_rejections == 5
    assert result.tracks == 1
    assert result.findings[0].value == "reference_face_001"
    assert result.findings[0].sources == ["yunet", "sface_reference"]

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from frameguard.face_gallery import (
    FaceGallerySession,
    FaceProfile,
    _TrackProfile,
    _cluster_track_profiles,
    resolve_blur_person_ids,
)


@dataclass
class DummyFinding:
    value: str


def embedding(*values: float) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float32)
    return vector / np.linalg.norm(vector)


def profile(person_id: str, label: str) -> FaceProfile:
    return FaceProfile(
        person_id=person_id,
        label=label,
        track_ids=[f"{person_id}_track"],
        portrait_rgb=np.zeros((8, 8, 3), dtype=np.uint8),
        embedding=embedding(1.0, 0.0),
        first_seen_ms=0,
        last_seen_ms=1000,
        observation_count=3,
        mean_detector_confidence=0.9,
    )


def gallery_session() -> FaceGallerySession:
    profiles = [
        profile("person_01", "Person 01"),
        profile("person_02", "Person 02"),
        profile("person_03", "Person 03"),
    ]
    return FaceGallerySession(
        session_id="gallery_test",
        video_path="/tmp/video.mp4",
        profiles=profiles,
        findings=[],
        metrics={},
    )


def test_cluster_groups_similar_embeddings() -> None:
    first = _TrackProfile(
        finding=DummyFinding("face_001"),  # type: ignore[arg-type]
        portrait_rgb=np.zeros((4, 4, 3), dtype=np.uint8),
        embedding=embedding(1.0, 0.0),
        quality_score=10.0,
    )
    second = _TrackProfile(
        finding=DummyFinding("face_002"),  # type: ignore[arg-type]
        portrait_rgb=np.zeros((4, 4, 3), dtype=np.uint8),
        embedding=embedding(0.99, 0.05),
        quality_score=9.0,
    )
    third = _TrackProfile(
        finding=DummyFinding("face_003"),  # type: ignore[arg-type]
        portrait_rgb=np.zeros((4, 4, 3), dtype=np.uint8),
        embedding=embedding(0.0, 1.0),
        quality_score=8.0,
    )

    clusters = _cluster_track_profiles(
        [first, second, third],
        similarity_threshold=0.8,
    )

    assert sorted(len(cluster) for cluster in clusters) == [1, 2]


def test_blur_selected_gallery_people() -> None:
    value = resolve_blur_person_ids(
        gallery_session(),
        ["Person 01", "Person 03"],
        gallery_action="blur_selected",
    )
    assert value == {"person_01", "person_03"}


def test_keep_selected_visible_blurs_everyone_else() -> None:
    value = resolve_blur_person_ids(
        gallery_session(),
        ["Person 02"],
        gallery_action="keep_selected_visible",
    )
    assert value == {"person_01", "person_03"}


def test_uploaded_blur_match_adds_identity() -> None:
    value = resolve_blur_person_ids(
        gallery_session(),
        [],
        gallery_action="blur_selected",
        uploaded_matches=[
            {
                "accepted": True,
                "matched_person_id": "person_02",
            }
        ],
        uploaded_photo_action="blur",
    )
    assert value == {"person_02"}


def test_uploaded_keep_visible_removes_identity() -> None:
    value = resolve_blur_person_ids(
        gallery_session(),
        [],
        gallery_action="keep_selected_visible",
        uploaded_matches=[
            {
                "accepted": True,
                "matched_person_id": "person_02",
            }
        ],
        uploaded_photo_action="keep_visible",
    )
    assert value == {"person_01", "person_03"}

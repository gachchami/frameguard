from __future__ import annotations

import numpy as np

from frameguard.minor_protection import (
    AgeDecision,
    _message_content_text,
    classify_minor_face_tracks,
    decide_age_policy,
)
from frameguard.schemas import BoxObservation, Finding


def test_clear_minor_interval_is_blurred() -> None:
    decision = decide_age_policy(
        track_id="face_001",
        estimated_age_low=11,
        estimated_age_high=16,
        confidence=0.91,
        quality="good",
        sample_count=5,
    )
    assert decision.category == "likely_minor"
    assert decision.blur is True


def test_boundary_overlap_is_uncertain_and_visible_by_default() -> None:
    decision = decide_age_policy(
        track_id="face_002",
        estimated_age_low=17,
        estimated_age_high=21,
        confidence=0.88,
        quality="good",
        sample_count=5,
    )
    assert decision.category == "uncertain"
    assert decision.blur is False


def test_boundary_overlap_can_be_blurred_in_strict_mode() -> None:
    decision = decide_age_policy(
        track_id="face_002b",
        estimated_age_low=17,
        estimated_age_high=21,
        confidence=0.88,
        quality="good",
        sample_count=5,
        blur_uncertain=True,
    )
    assert decision.category == "uncertain"
    assert decision.blur is True


def test_high_confidence_adult_interval_can_remain_visible() -> None:
    decision = decide_age_policy(
        track_id="face_003",
        estimated_age_low=24,
        estimated_age_high=31,
        confidence=0.86,
        quality="good",
        sample_count=5,
    )
    assert decision.category == "likely_adult"
    assert decision.blur is False


def test_low_confidence_adult_estimate_is_uncertain_and_visible_by_default() -> None:
    decision = decide_age_policy(
        track_id="face_004",
        estimated_age_low=30,
        estimated_age_high=38,
        confidence=0.40,
        quality="limited",
        sample_count=3,
    )
    assert decision.category == "uncertain"
    assert decision.blur is False


def test_estimator_failure_is_uncertain_and_visible_by_default() -> None:
    decision = decide_age_policy(
        track_id="face_005",
        estimated_age_low=None,
        estimated_age_high=None,
        confidence=0.0,
        quality="poor",
        sample_count=0,
        failure_reason="no_usable_face_crops",
    )
    assert decision.category == "uncertain"
    assert decision.blur is False


class _FakeEstimator:
    def __init__(self, decision: AgeDecision) -> None:
        self.decision = decision

    def estimate(self, crops: list[np.ndarray], *, track_id: str) -> AgeDecision:
        assert track_id == self.decision.track_id
        return self.decision


def test_classifier_excludes_confident_adult_track(monkeypatch) -> None:
    finding = Finding(
        id="finding_1",
        type="face",
        value="face_001",
        modality="visual",
        start_ms=0,
        end_ms=1000,
        confidence=0.95,
        observations=[
            BoxObservation(
                time_ms=0,
                x=10,
                y=10,
                width=100,
                height=100,
                confidence=0.95,
            )
        ],
        action="blur",
        sources=["yunet"],
    )
    decision = decide_age_policy(
        track_id="face_001",
        estimated_age_low=25,
        estimated_age_high=32,
        confidence=0.9,
        quality="good",
        sample_count=1,
    )
    monkeypatch.setattr(
        "frameguard.minor_protection.extract_track_crops",
        lambda *args, **kwargs: [np.zeros((128, 128, 3), dtype=np.uint8)],
    )
    selected, decisions = classify_minor_face_tracks(
        "unused.mp4",
        [finding],
        estimator=_FakeEstimator(decision),
    )
    assert selected == []
    assert decisions == [decision]


def test_message_content_accepts_typed_text_blocks() -> None:
    assert _message_content_text(
        [{"type": "text", "text": '{"estimated_age_low": 12}'}]
    ) == '{"estimated_age_low": 12}'

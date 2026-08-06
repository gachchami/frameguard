from __future__ import annotations

import json

import numpy as np

from frameguard.minor_protection import (
    AgeDecision,
    TimestampAssessment,
    TrackEvidence,
    _message_content_text,
    _parse_timestamp_assessments,
    classify_minor_face_tracks,
    decide_child_policy,
)
from frameguard.schemas import BoxObservation, Finding


def assessment(
    index: int,
    classification: str,
    *,
    confidence: float = 0.9,
    quality: str = "good",
    reason_codes: tuple[str, ...] = (),
) -> TimestampAssessment:
    return TimestampAssessment(
        index=index,
        classification=classification,  # type: ignore[arg-type]
        confidence=confidence,
        quality=quality,
        reason_codes=reason_codes,
    )


def test_three_consistent_child_timestamps_are_blurred() -> None:
    decision = decide_child_policy(
        track_id="face_001",
        assessments=[
            assessment(1, "child", reason_codes=("childlike_face", "childlike_body_proportions")),
            assessment(2, "child", reason_codes=("childlike_face", "childlike_body_proportions")),
            assessment(3, "child", reason_codes=("childlike_face", "childlike_body_proportions")),
            assessment(4, "uncertain", confidence=0.8),
        ],
        sample_count=4,
        minimum_usable_timestamps=3,
        consensus_fraction=0.70,
    )
    assert decision.category == "likely_minor"
    assert decision.blur is True
    assert decision.child_votes == 3


def test_three_consistent_adult_timestamps_remain_visible() -> None:
    decision = decide_child_policy(
        track_id="face_002",
        assessments=[
            assessment(1, "adult", reason_codes=("mature_face",)),
            assessment(2, "adult", reason_codes=("adult_body_proportions",)),
            assessment(3, "adult", reason_codes=("mature_face", "adult_body_proportions")),
            assessment(4, "uncertain", confidence=0.8),
        ],
        sample_count=4,
        minimum_usable_timestamps=3,
        consensus_fraction=0.70,
    )
    assert decision.category == "likely_adult"
    assert decision.blur is False
    assert decision.adult_votes == 3


def test_child_and_adult_votes_are_uncertain() -> None:
    decision = decide_child_policy(
        track_id="face_003",
        assessments=[
            assessment(1, "child", reason_codes=("childlike_face", "childlike_body_proportions")),
            assessment(2, "child", reason_codes=("childlike_face", "childlike_body_proportions")),
            assessment(3, "adult", reason_codes=("mature_face",)),
            assessment(4, "uncertain"),
        ],
        sample_count=4,
        minimum_usable_timestamps=3,
        consensus_fraction=0.60,
    )
    assert decision.category == "uncertain"
    assert decision.blur is False
    assert decision.reason == "conflicting_child_adult_votes"


def test_too_few_usable_timestamps_are_uncertain() -> None:
    decision = decide_child_policy(
        track_id="face_004",
        assessments=[
            assessment(1, "child", quality="poor"),
            assessment(2, "child", confidence=0.40),
            assessment(3, "adult", confidence=0.95),
        ],
        sample_count=3,
        minimum_confidence=0.70,
        minimum_usable_timestamps=3,
    )
    assert decision.category == "uncertain"
    assert decision.usable_timestamps == 1
    assert decision.blur is False


def test_uncertain_can_be_blurred_in_strict_mode() -> None:
    decision = decide_child_policy(
        track_id="face_005",
        assessments=[assessment(1, "uncertain")],
        sample_count=1,
        minimum_usable_timestamps=3,
        blur_uncertain=True,
    )
    assert decision.category == "uncertain"
    assert decision.blur is True


def test_parser_accepts_timestamp_array() -> None:
    parsed, reasons = _parse_timestamp_assessments(
        {
            "timestamps": [
                {
                    "index": 1,
                    "classification": "kid",
                    "confidence": 0.91,
                    "quality": "good",
                    "reason_codes": ["childlike_face"],
                },
                {
                    "index": 2,
                    "classification": "grown-up",
                    "confidence": 0.88,
                    "quality": "mixed",
                },
            ],
            "overall_reason_codes": ["mixed_views"],
        },
        expected_count=2,
    )
    assert [item.classification for item in parsed] == ["child", "adult"]
    assert parsed[1].quality == "limited"
    assert reasons == ("mixed_views",)


class _FakeEstimator:
    def __init__(self, decision: AgeDecision) -> None:
        self.decision = decision

    def estimate(
        self,
        evidence: list[TrackEvidence],
        *,
        track_id: str,
    ) -> AgeDecision:
        assert evidence
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
    decision = decide_child_policy(
        track_id="face_001",
        assessments=[
            assessment(1, "adult", reason_codes=("mature_face",)),
            assessment(2, "adult", reason_codes=("adult_body_proportions",)),
            assessment(3, "adult", reason_codes=("mature_face", "adult_body_proportions")),
        ],
        sample_count=3,
    )
    evidence = TrackEvidence(
        time_ms=0,
        full_frame=np.zeros((200, 300, 3), dtype=np.uint8),
        person_crop=np.zeros((180, 100, 3), dtype=np.uint8),
        face_crop=np.zeros((100, 100, 3), dtype=np.uint8),
        face_width_px=100,
        face_height_px=100,
        detector_confidence=0.95,
        face_sharpness=100.0,
        quality_score=0.95,
        quality_hint="good",
    )
    monkeypatch.setattr(
        "frameguard.minor_protection.extract_track_evidence",
        lambda *args, **kwargs: [evidence],
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
        [{"type": "text", "text": '{"timestamps": []}'}]
    ) == '{"timestamps": []}'


def test_qwen_classifier_uses_three_views_and_aggregates_adult_votes(monkeypatch) -> None:
    from frameguard.minor_protection import QwenChildClassifier

    evidence = [
        TrackEvidence(
            time_ms=index * 500,
            full_frame=np.zeros((180, 320, 3), dtype=np.uint8),
            person_crop=np.zeros((180, 100, 3), dtype=np.uint8),
            face_crop=np.zeros((96, 96, 3), dtype=np.uint8),
            face_width_px=96,
            face_height_px=96,
            detector_confidence=0.94,
            face_sharpness=90.0,
            quality_score=0.9,
            quality_hint="good",
        )
        for index in range(3)
    ]
    captured: dict[str, object] = {}

    class _Response:
        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, object]:
            return {
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "timestamps": [
                                        {
                                            "index": index,
                                            "classification": "adult",
                                            "confidence": 0.92,
                                            "quality": "good",
                                            "reason_codes": ["mature_face"],
                                        }
                                        for index in range(1, 4)
                                    ]
                                }
                            )
                        }
                    }
                ]
            }

    class _Client:
        def __init__(self, *, timeout: float) -> None:
            captured["timeout"] = timeout

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc, tb) -> None:
            return None

        def post(self, url: str, *, headers, json):
            captured["url"] = url
            captured["payload"] = json
            return _Response()

    monkeypatch.setattr("frameguard.minor_protection.httpx.Client", _Client)
    classifier = QwenChildClassifier(
        api_base="http://127.0.0.1:8091/v1",
        model="local-model",
        minimum_usable_timestamps=3,
    )
    decision = classifier.estimate(evidence, track_id="face_009")

    assert decision.category == "likely_adult"
    assert decision.blur is False
    payload = captured["payload"]
    assert isinstance(payload, dict)
    content = payload["messages"][0]["content"]  # type: ignore[index]
    image_parts = [item for item in content if item.get("type") == "image_url"]
    assert len(image_parts) == 9


def test_child_labels_without_holistic_evidence_do_not_blur() -> None:
    decision = decide_child_policy(
        track_id="face_weak",
        assessments=[
            assessment(1, "child"),
            assessment(2, "child"),
            assessment(3, "child"),
        ],
        sample_count=3,
        median_face_width_px=120,
    )
    assert decision.category == "uncertain"
    assert decision.blur is False


def test_limited_quality_child_votes_do_not_blur() -> None:
    reasons = ("childlike_face", "childlike_body_proportions")
    decision = decide_child_policy(
        track_id="face_limited",
        assessments=[
            assessment(1, "child", quality="limited", reason_codes=reasons),
            assessment(2, "child", quality="limited", reason_codes=reasons),
            assessment(3, "child", quality="limited", reason_codes=reasons),
        ],
        sample_count=3,
        median_face_width_px=120,
    )
    assert decision.category == "uncertain"
    assert decision.blur is False


def test_small_face_child_consensus_does_not_blur() -> None:
    reasons = ("childlike_face", "childlike_body_proportions")
    decision = decide_child_policy(
        track_id="face_small",
        assessments=[
            assessment(1, "child", reason_codes=reasons),
            assessment(2, "child", reason_codes=reasons),
            assessment(3, "child", reason_codes=reasons),
        ],
        sample_count=3,
        median_face_width_px=42,
    )
    assert decision.category == "uncertain"
    assert decision.reason == "face_too_small_for_reliable_child_classification"
    assert decision.blur is False


def test_one_clear_child_cue_per_timestamp_is_sufficient() -> None:
    decision = decide_child_policy(
        track_id="face_balanced_1",
        assessments=[
            assessment(1, "child", reason_codes=("childlike_face",)),
            assessment(2, "child", reason_codes=("childlike_body_proportions",)),
            assessment(3, "child", reason_codes=("childlike_face",)),
        ],
        sample_count=3,
        minimum_usable_timestamps=3,
        consensus_fraction=0.70,
        median_face_width_px=96,
    )
    assert decision.category == "likely_minor"
    assert decision.blur is True


def test_uncertain_timestamps_do_not_dilute_decisive_child_consensus() -> None:
    decision = decide_child_policy(
        track_id="face_balanced_2",
        assessments=[
            assessment(1, "child", reason_codes=("childlike_face",)),
            assessment(2, "child", reason_codes=("childlike_body_proportions",)),
            assessment(3, "child", reason_codes=("childlike_face",)),
            assessment(4, "uncertain", reason_codes=("insufficient_detail",)),
            assessment(5, "uncertain", reason_codes=("profile_view",)),
        ],
        sample_count=5,
        minimum_usable_timestamps=3,
        consensus_fraction=0.70,
        median_face_width_px=96,
    )
    assert decision.category == "likely_minor"
    assert decision.blur is True
    assert decision.child_votes == 3
    assert decision.uncertain_votes == 2

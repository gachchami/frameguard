from __future__ import annotations

import json
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Literal

import cv2
import numpy as np

from .face_tracking import DEFAULT_FACE_MODEL, FaceScanResult, scan_face_tracks
from .observability import RunEventRecorder
from .pipeline import PipelineResult, analyze_video, merge_findings
from .redact import render_redacted_video
from .schemas import BoxObservation, Finding

GallerySelectionAction = Literal["blur_selected", "keep_selected_visible"]
UploadedPhotoAction = Literal["blur", "keep_visible"]


@dataclass(slots=True)
class _TrackProfile:
    finding: Finding
    portrait_rgb: np.ndarray
    embedding: np.ndarray | None
    quality_score: float
    preview_rgbs: list[np.ndarray] = field(default_factory=list)


@dataclass(slots=True)
class FaceProfile:
    person_id: str
    label: str
    track_ids: list[str]
    portrait_rgb: np.ndarray
    embedding: np.ndarray | None
    first_seen_ms: int
    last_seen_ms: int
    observation_count: int
    mean_detector_confidence: float
    preview_rgbs: list[np.ndarray] = field(default_factory=list)

    def public_summary(self) -> dict[str, object]:
        return {
            "person_id": self.person_id,
            "label": self.label,
            "track_ids": list(self.track_ids),
            "track_segments": len(self.track_ids),
            "first_seen_ms": self.first_seen_ms,
            "last_seen_ms": self.last_seen_ms,
            "observation_count": self.observation_count,
            "mean_detector_confidence": round(self.mean_detector_confidence, 4),
        }


@dataclass(slots=True)
class FaceGallerySession:
    session_id: str
    video_path: str
    profiles: list[FaceProfile]
    findings: list[Finding]
    metrics: dict[str, object]

    @property
    def labels(self) -> list[str]:
        return [profile.label for profile in self.profiles]

    @property
    def label_to_person_id(self) -> dict[str, str]:
        return {profile.label: profile.person_id for profile in self.profiles}

    @property
    def all_person_ids(self) -> set[str]:
        return {profile.person_id for profile in self.profiles}

    def gallery_items(
        self,
        selected_labels: Sequence[str] | None = None,
    ) -> list[tuple[np.ndarray, str]]:
        selected = set(selected_labels or [])
        return [
            (
                _selection_card(
                    profile.portrait_rgb,
                    selected=profile.label in selected,
                ),
                profile.label,
            )
            for profile in self.profiles
        ]

    def public_summary(self) -> dict[str, object]:
        return {
            "session_id": self.session_id,
            "video": Path(self.video_path).name,
            "identities": len(self.profiles),
            "face_track_segments": len(self.findings),
            "profiles": [profile.public_summary() for profile in self.profiles],
            "metrics": dict(self.metrics),
            "privacy": (
                "Profile crops and SFace embeddings remain in the server-side "
                "web session and are not written to the report or JSONL log."
            ),
        }


class FaceIdentityEncoder:
    """Detect, align, and embed a face with YuNet and SFace.

    This is a user-assistance feature for grouping gallery profiles. It is not
    identity verification and its threshold must be checked on representative
    footage.
    """

    def __init__(
        self,
        *,
        face_model_path: str | Path,
        recognition_model_path: str | Path,
        detector_threshold: float = 0.60,
    ) -> None:
        face_model = Path(face_model_path)
        recognition_model = Path(recognition_model_path)
        if not face_model.is_file():
            raise FileNotFoundError(f"YuNet model is missing: {face_model}")
        if not recognition_model.is_file():
            raise FileNotFoundError(f"SFace model is missing: {recognition_model}")
        if not hasattr(cv2, "FaceDetectorYN"):
            raise RuntimeError("OpenCV does not provide cv2.FaceDetectorYN.")
        if not hasattr(cv2, "FaceRecognizerSF"):
            raise RuntimeError("OpenCV does not provide cv2.FaceRecognizerSF.")

        self._detector = cv2.FaceDetectorYN.create(
            str(face_model),
            "",
            (320, 320),
            float(detector_threshold),
            0.3,
            5000,
        )
        self._recognizer = cv2.FaceRecognizerSF.create(str(recognition_model), "")
        self._input_size = (320, 320)

    def encode(self, image_bgr: np.ndarray) -> np.ndarray:
        if image_bgr is None or image_bgr.size == 0:
            raise ValueError("Cannot embed an empty image.")

        height, width = image_bgr.shape[:2]
        input_size = (max(1, width), max(1, height))
        if input_size != self._input_size:
            self._detector.setInputSize(input_size)
            self._input_size = input_size

        _, raw_faces = self._detector.detect(image_bgr)
        if raw_faces is not None and len(raw_faces):
            face = max(
                raw_faces,
                key=lambda row: float(row[2]) * float(row[3]) * float(row[-1]),
            )
            aligned = self._recognizer.alignCrop(image_bgr, face)
        else:
            # A padded crop from a tiny video face may not survive a second
            # detection pass. The fallback keeps the workflow usable, but the
            # gallery remains user-reviewed rather than fully automatic.
            aligned = cv2.resize(image_bgr, (112, 112), interpolation=cv2.INTER_CUBIC)

        raw_feature = self._recognizer.feature(aligned)
        vector = np.asarray(raw_feature, dtype=np.float32).reshape(-1)
        norm = float(np.linalg.norm(vector))
        if not np.isfinite(norm) or norm <= 1e-8:
            raise ValueError("SFace returned an invalid feature vector.")
        return vector / norm


def cosine_similarity(first: np.ndarray, second: np.ndarray) -> float:
    first_vector = np.asarray(first, dtype=np.float32).reshape(-1)
    second_vector = np.asarray(second, dtype=np.float32).reshape(-1)
    denominator = float(np.linalg.norm(first_vector) * np.linalg.norm(second_vector))
    if denominator <= 1e-8:
        return -1.0
    return float(np.dot(first_vector, second_vector) / denominator)


def _square_face_crop(
    frame_bgr: np.ndarray,
    observation: BoxObservation,
    *,
    margin: float = 0.55,
) -> np.ndarray:
    frame_height, frame_width = frame_bgr.shape[:2]
    center_x = observation.x + observation.width / 2.0
    center_y = observation.y + observation.height / 2.0
    side = max(observation.width, observation.height) * (1.0 + 2.0 * margin)

    x1 = max(0, int(round(center_x - side / 2.0)))
    y1 = max(0, int(round(center_y - side / 2.0)))
    x2 = min(frame_width, int(round(center_x + side / 2.0)))
    y2 = min(frame_height, int(round(center_y + side / 2.0)))
    if x2 <= x1 or y2 <= y1:
        return np.empty((0, 0, 3), dtype=np.uint8)
    return frame_bgr[y1:y2, x1:x2].copy()


def _observation_priority(observation: BoxObservation) -> float:
    return float(
        np.sqrt(max(1, observation.width * observation.height))
        * max(0.01, observation.confidence)
    )


def _candidate_observations(
    finding: Finding,
    *,
    maximum: int = 8,
    minimum_spacing_ms: int = 350,
) -> list[BoxObservation]:
    """Choose strong, temporally diverse observations before decoding frames."""

    if not finding.observations:
        raise ValueError(f"Face track {finding.value!r} has no observations.")

    selected: list[BoxObservation] = []
    for observation in sorted(
        finding.observations,
        key=_observation_priority,
        reverse=True,
    ):
        if any(
            abs(observation.time_ms - existing.time_ms) < minimum_spacing_ms
            for existing in selected
        ):
            continue
        selected.append(observation)
        if len(selected) >= maximum:
            break

    if not selected:
        selected.append(max(finding.observations, key=_observation_priority))
    return sorted(selected, key=lambda item: item.time_ms)


def _read_frames_sequentially(
    video_path: str | Path,
    requested_times_ms: Sequence[int],
) -> dict[int, np.ndarray]:
    """Decode the video once instead of repeatedly seeking inside H.264.

    Repeated ``CAP_PROP_POS_MSEC`` seeks can trigger FFmpeg ``mmco`` warnings
    and can return damaged reference frames. Sequential decoding is slower than
    a single seek but more reliable for a gallery extraction pass.
    """

    targets = sorted({max(0, int(value)) for value in requested_times_ms})
    if not targets:
        return {}

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video for face-gallery extraction: {video_path}")

    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0

    results: dict[int, np.ndarray] = {}
    target_index = 0
    frame_index = 0
    try:
        while target_index < len(targets):
            ok, frame = capture.read()
            if not ok or frame is None:
                break

            position_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
            if not np.isfinite(position_ms) or position_ms <= 0:
                position_ms = frame_index * 1000.0 / fps

            while target_index < len(targets) and position_ms >= targets[target_index]:
                results[targets[target_index]] = frame.copy()
                target_index += 1
            frame_index += 1
    finally:
        capture.release()

    return results


def _portrait_card(crop_bgr: np.ndarray, *, size: int = 512) -> np.ndarray:
    """Create a high-resolution gallery image without over-enlarging the face."""

    if crop_bgr.size == 0:
        return np.zeros((size, size, 3), dtype=np.uint8)

    height, width = crop_bgr.shape[:2]
    scale = max(size / max(width, 1), size / max(height, 1))
    resized_width = max(size, int(round(width * scale)))
    resized_height = max(size, int(round(height * scale)))
    resized = cv2.resize(
        crop_bgr,
        (resized_width, resized_height),
        interpolation=(
            cv2.INTER_LANCZOS4
            if scale > 1.0
            else cv2.INTER_AREA
        ),
    )

    # A restrained unsharp mask restores edge contrast lost when a small video
    # face is enlarged. It cannot invent detail, but it avoids the smeared look
    # produced by a large browser-scaled thumbnail.
    softened = cv2.GaussianBlur(resized, (0, 0), 0.8)
    resized = cv2.addWeighted(resized, 1.18, softened, -0.18, 0)

    x1 = max(0, (resized_width - size) // 2)
    y1 = max(0, (resized_height - size) // 2)
    card = resized[y1 : y1 + size, x1 : x1 + size].copy()
    return cv2.cvtColor(card, cv2.COLOR_BGR2RGB)


def _selection_card(image_rgb: np.ndarray, *, selected: bool) -> np.ndarray:
    """Add an unmistakable visual selection state directly to a gallery card."""

    card = np.asarray(image_rgb, dtype=np.uint8).copy()
    if not selected:
        return card

    height, width = card.shape[:2]
    border = max(6, min(height, width) // 30)
    selected_rgb = (34, 197, 94)
    cv2.rectangle(
        card,
        (border // 2, border // 2),
        (width - border // 2 - 1, height - border // 2 - 1),
        selected_rgb,
        border,
    )

    banner_height = max(32, height // 7)
    overlay = card.copy()
    cv2.rectangle(
        overlay,
        (0, height - banner_height),
        (width, height),
        selected_rgb,
        thickness=-1,
    )
    card = cv2.addWeighted(overlay, 0.88, card, 0.12, 0)
    cv2.putText(
        card,
        "SELECTED",
        (max(8, width // 18), height - max(9, banner_height // 4)),
        cv2.FONT_HERSHEY_SIMPLEX,
        max(0.55, width / 420.0),
        (255, 255, 255),
        max(1, border // 3),
        cv2.LINE_AA,
    )
    return card


def _sharpness(crop_bgr: np.ndarray) -> float:
    if crop_bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _candidate_quality(
    observation: BoxObservation,
    crop_bgr: np.ndarray,
) -> float:
    """Combine source resolution, sharpness, and YuNet confidence."""

    minimum_face_dimension = max(1, min(observation.width, observation.height))
    size_score = min(1.0, minimum_face_dimension / 96.0)
    sharpness_score = min(1.0, _sharpness(crop_bgr) / 150.0)
    detector_score = min(1.0, max(0.0, observation.confidence))
    return float(
        0.40 * size_score
        + 0.38 * sharpness_score
        + 0.22 * detector_score
    )


def _normalized_embedding_mean(
    weighted_embeddings: Sequence[tuple[np.ndarray, float]],
) -> np.ndarray | None:
    if not weighted_embeddings:
        return None
    vectors = np.stack([item[0] for item in weighted_embeddings])
    weights = np.asarray(
        [max(0.05, float(item[1])) for item in weighted_embeddings],
        dtype=np.float32,
    )
    centroid = np.average(vectors, axis=0, weights=weights).astype(np.float32)
    norm = float(np.linalg.norm(centroid))
    if not np.isfinite(norm) or norm <= 1e-8:
        return None
    return centroid / norm


def _profile_quality(
    finding: Finding,
    observation: BoxObservation,
    crop_bgr: np.ndarray,
) -> float:
    return float(
        _candidate_quality(observation, crop_bgr)
        * np.log1p(max(1, len(finding.observations)))
    )


def _track_times(item: _TrackProfile) -> tuple[int, int]:
    return (
        int(getattr(item.finding, "start_ms", 0)),
        int(getattr(item.finding, "end_ms", 0)),
    )


def _clusters_temporally_compatible(
    first: Sequence[_TrackProfile],
    second: Sequence[_TrackProfile],
    *,
    maximum_overlap_ms: int = 350,
) -> bool:
    """Two simultaneously visible tracks must not become one identity."""

    for first_item in first:
        first_start, first_end = _track_times(first_item)
        for second_item in second:
            second_start, second_end = _track_times(second_item)
            overlap = min(first_end, second_end) - max(first_start, second_start)
            if overlap > maximum_overlap_ms:
                return False
    return True


def _cluster_centroid(cluster: Sequence[_TrackProfile]) -> np.ndarray | None:
    embeddings = [
        item.embedding
        for item in cluster
        if item.embedding is not None
    ]
    if not embeddings:
        return None
    centroid = np.mean(np.stack(embeddings), axis=0).astype(np.float32)
    norm = float(np.linalg.norm(centroid))
    if not np.isfinite(norm) or norm <= 1e-8:
        return None
    return centroid / norm


def _cluster_track_profiles(
    track_profiles: Sequence[_TrackProfile],
    *,
    similarity_threshold: float,
) -> list[list[_TrackProfile]]:
    """Merge duplicate track fragments into unique people.

    The highest-similarity compatible cluster pair is merged repeatedly.
    Temporal overlap prevents two concurrently visible people from being
    collapsed into one profile.
    """

    threshold = max(-1.0, min(1.0, float(similarity_threshold)))
    clusters: list[list[_TrackProfile]] = [
        [item]
        for item in sorted(
            track_profiles,
            key=lambda item: item.quality_score,
            reverse=True,
        )
    ]

    while True:
        best_pair: tuple[int, int] | None = None
        best_similarity = threshold

        centroids = [_cluster_centroid(cluster) for cluster in clusters]
        for first_index in range(len(clusters)):
            first_centroid = centroids[first_index]
            if first_centroid is None:
                continue

            for second_index in range(first_index + 1, len(clusters)):
                second_centroid = centroids[second_index]
                if second_centroid is None:
                    continue
                if not _clusters_temporally_compatible(
                    clusters[first_index],
                    clusters[second_index],
                ):
                    continue

                similarity = cosine_similarity(first_centroid, second_centroid)
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_pair = (first_index, second_index)

        if best_pair is None:
            break

        first_index, second_index = best_pair
        clusters[first_index].extend(clusters[second_index])
        del clusters[second_index]

    return clusters


def _profile_from_cluster(
    cluster: Sequence[_TrackProfile],
    *,
    person_number: int,
) -> FaceProfile:
    representative = max(cluster, key=lambda item: item.quality_score)
    findings = [item.finding for item in cluster]
    observations = [
        observation
        for finding in findings
        for observation in finding.observations
    ]
    centroid = _normalized_embedding_mean(
        [
            (item.embedding, item.quality_score)
            for item in cluster
            if item.embedding is not None
        ]
    )

    first_seen = min(item.start_ms for item in findings)
    last_seen = max(item.end_ms for item in findings)
    track_word = "segment" if len(findings) == 1 else "segments"
    label = (
        f"Person {person_number:02d} | {len(findings)} track {track_word} | "
        f"{first_seen / 1000.0:.1f}s–{last_seen / 1000.0:.1f}s"
    )
    return FaceProfile(
        person_id=f"person_{person_number:02d}",
        label=label,
        track_ids=sorted(item.value for item in findings),
        portrait_rgb=representative.portrait_rgb,
        embedding=centroid,
        first_seen_ms=first_seen,
        last_seen_ms=last_seen,
        observation_count=len(observations),
        mean_detector_confidence=(
            sum(item.confidence for item in observations) / len(observations)
            if observations
            else 0.0
        ),
        preview_rgbs=[
            image
            for item in sorted(cluster, key=lambda row: row.quality_score, reverse=True)
            for image in (item.preview_rgbs or [item.portrait_rgb])
        ][:3],
    )


def scan_face_gallery(
    video_path: str | Path,
    *,
    face_model_path: str | Path = DEFAULT_FACE_MODEL,
    face_recognition_model_path: str | Path,
    face_sample_interval_ms: int = 200,
    face_score_threshold: float = 0.75,
    face_max_track_gap_ms: int = 900,
    face_min_track_observations: int = 2,
    identity_similarity_threshold: float = 0.40,
    recorder: RunEventRecorder | None = None,
) -> FaceGallerySession:
    """Extract representative faces and group fragmented temporal tracks."""

    started = time.perf_counter()
    video_path = Path(video_path).resolve()
    scan: FaceScanResult = scan_face_tracks(
        video_path,
        model_path=face_model_path,
        sample_interval_ms=int(face_sample_interval_ms),
        score_threshold=float(face_score_threshold),
        max_track_gap_ms=int(face_max_track_gap_ms),
        min_track_observations=int(face_min_track_observations),
        recorder=recorder,
    )

    encoder = FaceIdentityEncoder(
        face_model_path=face_model_path,
        recognition_model_path=face_recognition_model_path,
    )

    candidates_by_track = {
        finding.value: _candidate_observations(finding)
        for finding in scan.findings
    }
    requested_times = [
        observation.time_ms
        for observations in candidates_by_track.values()
        for observation in observations
    ]
    frames_by_time = _read_frames_sequentially(video_path, requested_times)

    track_profiles: list[_TrackProfile] = []
    embedding_failures = 0
    tracks_without_frames = 0

    for finding in scan.findings:
        candidate_rows: list[
            tuple[float, BoxObservation, np.ndarray, np.ndarray]
        ] = []
        for observation in candidates_by_track[finding.value]:
            frame = frames_by_time.get(observation.time_ms)
            if frame is None:
                continue

            identity_crop = _square_face_crop(
                frame,
                observation,
                margin=0.42,
            )
            display_crop = _square_face_crop(
                frame,
                observation,
                # Include hair, shoulders, and nearby clothing. Tiny faces are
                # easier to distinguish with context than as extreme closeups.
                margin=0.70,
            )
            if identity_crop.size == 0 or display_crop.size == 0:
                continue

            quality = _profile_quality(
                finding,
                observation,
                identity_crop,
            )
            candidate_rows.append(
                (
                    quality,
                    observation,
                    identity_crop,
                    display_crop,
                )
            )

        if not candidate_rows:
            tracks_without_frames += 1
            track_profiles.append(
                _TrackProfile(
                    finding=finding,
                    portrait_rgb=np.zeros((512, 512, 3), dtype=np.uint8),
                    embedding=None,
                    quality_score=0.0,
                    preview_rgbs=[],
                )
            )
            continue

        candidate_rows.sort(key=lambda item: item[0], reverse=True)
        best_quality, _, _, best_display_crop = candidate_rows[0]
        portrait = _portrait_card(best_display_crop)
        previews = [
            _portrait_card(display_crop)
            for _, _, _, display_crop in candidate_rows[:3]
        ]

        weighted_embeddings: list[tuple[np.ndarray, float]] = []
        for quality, _, identity_crop, _ in candidate_rows[:3]:
            try:
                weighted_embeddings.append(
                    (encoder.encode(identity_crop), quality)
                )
            except (ValueError, cv2.error):
                embedding_failures += 1

        track_profiles.append(
            _TrackProfile(
                finding=finding,
                portrait_rgb=portrait,
                embedding=_normalized_embedding_mean(weighted_embeddings),
                quality_score=best_quality,
                preview_rgbs=previews,
            )
        )

    clusters = _cluster_track_profiles(
        track_profiles,
        similarity_threshold=float(identity_similarity_threshold),
    )
    profiles = [
        _profile_from_cluster(cluster, person_number=index)
        for index, cluster in enumerate(clusters, start=1)
    ]
    profiles.sort(key=lambda item: (item.first_seen_ms, item.person_id))

    # Keep labels stable and chronological after clustering.
    for index, profile in enumerate(profiles, start=1):
        profile.person_id = f"person_{index:02d}"
        track_word = "segment" if len(profile.track_ids) == 1 else "segments"
        profile.label = (
            f"Person {index:02d} | {len(profile.track_ids)} track {track_word} | "
            f"{profile.first_seen_ms / 1000.0:.1f}s–"
            f"{profile.last_seen_ms / 1000.0:.1f}s"
        )

    metrics: dict[str, object] = {
        "sampled_frames": scan.sampled_frames,
        "face_detections": scan.detections,
        "face_track_segments": scan.tracks,
        "rejected_face_tracks": scan.rejected_tracks,
        "gallery_identities": len(profiles),
        "duplicate_track_segments_merged": max(0, len(scan.findings) - len(profiles)),
        "tracks_without_decoded_profile_frame": tracks_without_frames,
        "tracks_with_sface_embedding": sum(
            item.embedding is not None for item in track_profiles
        ),
        "embedding_failures": embedding_failures,
        "identity_similarity_threshold": float(identity_similarity_threshold),
        "elapsed_seconds": round(time.perf_counter() - started, 4),
    }
    return FaceGallerySession(
        session_id=f"gallery_{int(time.time() * 1000)}",
        video_path=str(video_path),
        profiles=profiles,
        findings=scan.findings,
        metrics=metrics,
    )


def _reference_paths(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (str, Path)):
        values: Iterable[object] = [value]
    elif isinstance(value, Iterable):
        values = value
    else:
        values = [value]

    paths: list[str] = []
    for item in values:
        if item is None:
            continue
        if isinstance(item, (str, Path)):
            paths.append(str(item))
            continue
        candidate = getattr(item, "name", None) or getattr(item, "path", None)
        if candidate:
            paths.append(str(candidate))
    return paths


def match_uploaded_reference_photos(
    session: FaceGallerySession,
    uploaded_photos: object,
    *,
    face_model_path: str | Path,
    face_recognition_model_path: str | Path,
    threshold: float,
) -> list[dict[str, object]]:
    """Find the best gallery identity for every uploaded reference photo."""

    paths = _reference_paths(uploaded_photos)
    if not paths:
        return []

    encoder = FaceIdentityEncoder(
        face_model_path=face_model_path,
        recognition_model_path=face_recognition_model_path,
    )
    matches: list[dict[str, object]] = []

    for path in paths:
        image = cv2.imread(str(path))
        if image is None:
            matches.append(
                {
                    "uploaded_file": Path(path).name,
                    "matched_person_id": None,
                    "matched_label": None,
                    "similarity": None,
                    "accepted": False,
                    "reason": "image_could_not_be_read",
                }
            )
            continue

        try:
            reference_embedding = encoder.encode(image)
        except (ValueError, cv2.error):
            matches.append(
                {
                    "uploaded_file": Path(path).name,
                    "matched_person_id": None,
                    "matched_label": None,
                    "similarity": None,
                    "accepted": False,
                    "reason": "face_embedding_failed",
                }
            )
            continue

        best_profile: FaceProfile | None = None
        best_similarity = -1.0
        for profile in session.profiles:
            if profile.embedding is None:
                continue
            similarity = cosine_similarity(reference_embedding, profile.embedding)
            if similarity > best_similarity:
                best_profile = profile
                best_similarity = similarity

        accepted = best_profile is not None and best_similarity >= float(threshold)
        matches.append(
            {
                "uploaded_file": Path(path).name,
                "matched_person_id": best_profile.person_id if accepted else None,
                "matched_label": best_profile.label if accepted else None,
                "similarity": round(best_similarity, 6) if best_profile else None,
                "accepted": accepted,
                "reason": "matched" if accepted else "below_similarity_threshold",
            }
        )

    return matches


def resolve_blur_person_ids(
    session: FaceGallerySession,
    selected_labels: Sequence[str] | None,
    *,
    gallery_action: GallerySelectionAction,
    uploaded_matches: Sequence[dict[str, object]] = (),
    uploaded_photo_action: UploadedPhotoAction = "blur",
) -> set[str]:
    selected_labels = selected_labels or []
    label_to_id = session.label_to_person_id
    selected_ids = {
        label_to_id[label]
        for label in selected_labels
        if label in label_to_id
    }

    if gallery_action == "blur_selected":
        blur_ids = set(selected_ids)
    elif gallery_action == "keep_selected_visible":
        blur_ids = session.all_person_ids - selected_ids
    else:
        raise ValueError(f"Unknown gallery action: {gallery_action!r}")

    uploaded_ids = {
        str(item["matched_person_id"])
        for item in uploaded_matches
        if item.get("accepted") and item.get("matched_person_id")
    }
    if uploaded_photo_action == "blur":
        blur_ids.update(uploaded_ids)
    elif uploaded_photo_action == "keep_visible":
        blur_ids.difference_update(uploaded_ids)
    else:
        raise ValueError(f"Unknown uploaded-photo action: {uploaded_photo_action!r}")

    return blur_ids


def _selected_face_findings(
    session: FaceGallerySession,
    blur_person_ids: set[str],
) -> list[Finding]:
    person_by_track: dict[str, str] = {}
    for profile in session.profiles:
        for track_id in profile.track_ids:
            person_by_track[track_id] = profile.person_id

    selected: list[Finding] = []
    for finding in session.findings:
        person_id = person_by_track.get(finding.value)
        if person_id not in blur_person_ids:
            continue
        selected.append(
            replace(
                finding,
                type="user_selected_face",
                value=str(person_id),
                reason=(
                    "Face selected through the user-reviewed gallery or an "
                    "uploaded reference-photo match."
                ),
                sources=sorted({*finding.sources, "user_face_gallery", "sface"}),
            )
        )
    return selected


def _face_report(finding: Finding) -> dict[str, object]:
    return {
        "id": finding.id,
        "type": finding.type,
        "value": finding.value,
        "modality": finding.modality,
        "start_ms": finding.start_ms,
        "end_ms": finding.end_ms,
        "confidence": finding.confidence,
        "reason": finding.reason,
        "visual_location": finding.visual_location,
        "action": finding.action,
        "sources": finding.sources,
        "observations": [
            {
                "time_ms": item.time_ms,
                "x": item.x,
                "y": item.y,
                "width": item.width,
                "height": item.height,
                "confidence": item.confidence,
            }
            for item in finding.observations
        ],
    }


def _update_report(
    report_path: Path,
    *,
    session: FaceGallerySession,
    selected_face_findings: list[Finding],
    blur_person_ids: set[str],
    gallery_action: GallerySelectionAction,
    uploaded_photo_action: UploadedPhotoAction,
    uploaded_matches: Sequence[dict[str, object]],
    metrics: dict[str, object],
) -> None:
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        payload = {}

    configuration = payload.setdefault("configuration", {})
    if isinstance(configuration, dict):
        configuration.update(
            {
                "redact_faces": True,
                "face_redaction_mode": "manual_gallery",
                "gallery_selection_action": gallery_action,
                "uploaded_photo_action": uploaded_photo_action,
                "gallery_session_id": session.session_id,
            }
        )

    metrics_payload = payload.setdefault("metrics", {})
    if isinstance(metrics_payload, dict):
        metrics_payload.update(metrics)

    match_by_person: dict[str, list[dict[str, object]]] = {}
    for match in uploaded_matches:
        person_id = match.get("matched_person_id")
        if person_id:
            match_by_person.setdefault(str(person_id), []).append(
                {
                    "uploaded_file": match.get("uploaded_file"),
                    "similarity": match.get("similarity"),
                }
            )

    payload["face_gallery"] = {
        "purpose": "user-reviewed face selection",
        "privacy": (
            "Profile crops and SFace embeddings remain in memory for the current "
            "web session and are not embedded in this report."
        ),
        "selection_action": gallery_action,
        "uploaded_photo_action": uploaded_photo_action,
        "profiles": [
            {
                **profile.public_summary(),
                "blurred": profile.person_id in blur_person_ids,
                "uploaded_photo_matches": match_by_person.get(profile.person_id, []),
            }
            for profile in session.profiles
        ],
        "uploaded_photo_results": list(uploaded_matches),
    }

    findings_payload = payload.setdefault("findings", [])
    if isinstance(findings_payload, list):
        findings_payload.extend(_face_report(item) for item in selected_face_findings)

    privacy = payload.setdefault("privacy", {})
    if isinstance(privacy, dict):
        privacy.update(
            {
                "face_gallery_profiles_persisted": False,
                "face_gallery_embeddings_persisted": False,
            }
        )

    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def analyze_video_with_face_gallery(
    input_path: str | Path,
    *,
    session: FaceGallerySession,
    selected_labels: Sequence[str] | None,
    gallery_action: GallerySelectionAction,
    uploaded_photos: object = None,
    uploaded_photo_action: UploadedPhotoAction = "blur",
    face_model_path: str | Path = DEFAULT_FACE_MODEL,
    face_recognition_model_path: str | Path,
    reference_match_threshold: float = 0.363,
    **pipeline_kwargs: Any,
) -> PipelineResult:
    """Run normal FrameGuard analysis and apply a reviewed face selection."""

    input_path = Path(input_path).resolve()
    if str(input_path) != session.video_path:
        raise ValueError(
            "The face gallery belongs to another video. Extract profiles again "
            "after changing the uploaded video."
        )

    base_kwargs = dict(pipeline_kwargs)
    base_kwargs["redact_faces"] = False
    for key in (
        "face_redaction_mode",
        "reference_face_path",
        "face_recognition_model_path",
        "reference_match_threshold",
        "face_model_path",
        "face_sample_interval_ms",
        "face_score_threshold",
        "face_max_track_gap_ms",
        "face_min_track_observations",
    ):
        base_kwargs.pop(key, None)

    result = analyze_video(input_path, **base_kwargs)
    recorder = RunEventRecorder(
        result.log_path,
        run_id=result.run_id,
        level=str(base_kwargs.get("run_log_level", "INFO")),
    )

    uploaded_matches = match_uploaded_reference_photos(
        session,
        uploaded_photos,
        face_model_path=face_model_path,
        face_recognition_model_path=face_recognition_model_path,
        threshold=float(reference_match_threshold),
    )
    blur_person_ids = resolve_blur_person_ids(
        session,
        selected_labels,
        gallery_action=gallery_action,
        uploaded_matches=uploaded_matches,
        uploaded_photo_action=uploaded_photo_action,
    )
    selected_face_findings = _selected_face_findings(session, blur_person_ids)
    combined_findings = merge_findings([*result.findings, *selected_face_findings])

    recorder.info(
        "face_gallery.render.started",
        identities=len(session.profiles),
        selected_gallery_labels=len(selected_labels or []),
        uploaded_photos=len(_reference_paths(uploaded_photos)),
        uploaded_matches=sum(bool(item.get("accepted")) for item in uploaded_matches),
        identities_blurred=len(blur_person_ids),
        track_segments_blurred=len(selected_face_findings),
    )
    render_redacted_video(
        input_path,
        result.output_video,
        combined_findings,
        recorder=recorder,
    )

    metrics = dict(result.metrics)
    metrics.update(
        {
            "face_redaction_mode": "manual_gallery",
            "gallery_identities": len(session.profiles),
            "gallery_face_track_segments": len(session.findings),
            "gallery_identities_blurred": len(blur_person_ids),
            "gallery_face_track_segments_blurred": len(selected_face_findings),
            "gallery_uploaded_photos": len(_reference_paths(uploaded_photos)),
            "gallery_uploaded_photo_matches": sum(
                bool(item.get("accepted")) for item in uploaded_matches
            ),
            "output_bytes": result.output_video.stat().st_size,
        }
    )
    _update_report(
        result.report_path,
        session=session,
        selected_face_findings=selected_face_findings,
        blur_person_ids=blur_person_ids,
        gallery_action=gallery_action,
        uploaded_photo_action=uploaded_photo_action,
        uploaded_matches=uploaded_matches,
        metrics=metrics,
    )
    recorder.info(
        "face_gallery.render.completed",
        output=result.output_video.name,
        identities_blurred=len(blur_person_ids),
        track_segments_blurred=len(selected_face_findings),
    )

    return PipelineResult(
        run_id=result.run_id,
        output_video=result.output_video,
        report_path=result.report_path,
        log_path=result.log_path,
        findings=combined_findings,
        metrics=metrics,
    )

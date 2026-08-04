from __future__ import annotations

import csv
import json
import math
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = PROJECT_ROOT / "samples" / "face_reference_dataset"

FRAME_WIDTH = 480
FRAME_HEIGHT = 270
FPS = 10
FRAME_COUNT = 24

REGIONS = {
    "top_left": (0.18, 0.23),
    "top_center": (0.50, 0.23),
    "top_right": (0.82, 0.23),
    "center_left": (0.18, 0.50),
    "center": (0.50, 0.50),
    "center_right": (0.82, 0.50),
    "bottom_left": (0.18, 0.77),
    "bottom_right": (0.82, 0.77),
}

FACE_HEIGHTS = {
    "small": 32,
    "medium": 64,
    "large": 112,
}

MOTIONS = (
    "static",
    "horizontal",
    "vertical",
    "diagonal",
    "orbit",
    "zoom",
)


@dataclass(frozen=True)
class Subject:
    name: str
    image_path: Path
    face_box: tuple[int, int, int, int]


@dataclass(frozen=True)
class Placement:
    subject: Subject
    region: str
    size: str
    motion: str
    phase: float = 0.0


def _background(frame_index: int, variant: int) -> np.ndarray:
    y = np.linspace(0, 1, FRAME_HEIGHT, dtype=np.float32)[:, None]
    x = np.linspace(0, 1, FRAME_WIDTH, dtype=np.float32)[None, :]
    phase = frame_index / max(1, FRAME_COUNT - 1)
    base = np.zeros((FRAME_HEIGHT, FRAME_WIDTH, 3), dtype=np.float32)
    base[..., 0] = 35 + 55 * x + 10 * math.sin(phase * math.tau)
    base[..., 1] = 45 + 45 * y + 8 * variant
    base[..., 2] = 65 + 35 * (1 - x) + 12 * math.cos(phase * math.tau)
    grid_x = ((np.arange(FRAME_WIDTH) // 40) % 2)[None, :]
    grid_y = ((np.arange(FRAME_HEIGHT) // 40) % 2)[:, None]
    checker = ((grid_x + grid_y) % 2) * 5
    base += checker[..., None]
    return np.clip(base, 0, 255).astype(np.uint8)


def _motion_offset(motion: str, t: float, amplitude: float, phase: float) -> tuple[float, float, float]:
    angle = math.tau * t + phase
    if motion == "static":
        return 0.0, 0.0, 1.0
    if motion == "horizontal":
        return amplitude * math.sin(angle), 0.0, 1.0
    if motion == "vertical":
        return 0.0, amplitude * math.sin(angle), 1.0
    if motion == "diagonal":
        shift = amplitude * math.sin(angle)
        return shift, shift * 0.6, 1.0
    if motion == "orbit":
        return amplitude * math.cos(angle), amplitude * 0.6 * math.sin(angle), 1.0
    if motion == "zoom":
        return 0.0, 0.0, 1.0 + 0.10 * math.sin(angle)
    raise ValueError(f"Unknown motion: {motion}")


def _clip_box(x: int, y: int, width: int, height: int) -> list[int]:
    x1 = max(0, min(FRAME_WIDTH, x))
    y1 = max(0, min(FRAME_HEIGHT, y))
    x2 = max(0, min(FRAME_WIDTH, x + width))
    y2 = max(0, min(FRAME_HEIGHT, y + height))
    return [x1, y1, max(0, x2 - x1), max(0, y2 - y1)]


def _paste_subject(
    frame: np.ndarray,
    placement: Placement,
    frame_index: int,
) -> list[int]:
    source = cv2.imread(str(placement.subject.image_path), cv2.IMREAD_COLOR)
    if source is None:
        raise ValueError(f"Could not read subject image: {placement.subject.image_path}")

    fx, fy, fw, fh = placement.subject.face_box
    target_face_height = FACE_HEIGHTS[placement.size]
    t = frame_index / max(1, FRAME_COUNT - 1)
    dx, dy, zoom = _motion_offset(
        placement.motion,
        t,
        amplitude=10.0 if placement.size != "large" else 6.0,
        phase=placement.phase,
    )
    scale = target_face_height / max(1, fh) * zoom
    target_width = max(1, int(round(source.shape[1] * scale)))
    target_height = max(1, int(round(source.shape[0] * scale)))
    resized = cv2.resize(source, (target_width, target_height), interpolation=cv2.INTER_AREA)

    face_x = fx * scale
    face_y = fy * scale
    face_w = fw * scale
    face_h = fh * scale
    region_x, region_y = REGIONS[placement.region]
    face_center_x = region_x * FRAME_WIDTH + dx
    face_center_y = region_y * FRAME_HEIGHT + dy

    portrait_x = int(round(face_center_x - face_x - face_w / 2))
    portrait_y = int(round(face_center_y - face_y - face_h / 2))

    dst_x1 = max(0, portrait_x)
    dst_y1 = max(0, portrait_y)
    dst_x2 = min(FRAME_WIDTH, portrait_x + target_width)
    dst_y2 = min(FRAME_HEIGHT, portrait_y + target_height)
    if dst_x1 < dst_x2 and dst_y1 < dst_y2:
        src_x1 = dst_x1 - portrait_x
        src_y1 = dst_y1 - portrait_y
        src_x2 = src_x1 + (dst_x2 - dst_x1)
        src_y2 = src_y1 + (dst_y2 - dst_y1)
        frame[dst_y1:dst_y2, dst_x1:dst_x2] = resized[src_y1:src_y2, src_x1:src_x2]

    return _clip_box(
        int(round(portrait_x + face_x)),
        int(round(portrait_y + face_y)),
        int(round(face_w)),
        int(round(face_h)),
    )


def _write_video(
    output_path: Path,
    placements: list[Placement],
    label: str,
    variant: int,
) -> dict[str, object]:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    annotations: list[dict[str, object]] = []

    with tempfile.TemporaryDirectory(prefix="frameguard-face-video-") as temp_dir:
        intermediate = Path(temp_dir) / "intermediate.mp4"
        writer = cv2.VideoWriter(
            str(intermediate),
            cv2.VideoWriter_fourcc(*"mp4v"),
            FPS,
            (FRAME_WIDTH, FRAME_HEIGHT),
        )
        if not writer.isOpened():
            raise RuntimeError("OpenCV could not create the sample video")

        try:
            for frame_index in range(FRAME_COUNT):
                frame = _background(frame_index, variant)
                frame_boxes: list[dict[str, object]] = []
                for placement in placements:
                    box = _paste_subject(frame, placement, frame_index)
                    frame_boxes.append({"subject": placement.subject.name, "box": box})

                cv2.rectangle(frame, (0, FRAME_HEIGHT - 28), (FRAME_WIDTH, FRAME_HEIGHT), (15, 15, 15), -1)
                cv2.putText(
                    frame,
                    label,
                    (10, FRAME_HEIGHT - 9),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.48,
                    (240, 240, 240),
                    1,
                    cv2.LINE_AA,
                )
                writer.write(frame)
                annotations.append({"frame": frame_index, "faces": frame_boxes})
        finally:
            writer.release()

        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg:
            subprocess.run(
                [
                    ffmpeg,
                    "-y",
                    "-loglevel",
                    "error",
                    "-i",
                    str(intermediate),
                    "-an",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "veryfast",
                    "-crf",
                    "30",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(output_path),
                ],
                check=True,
            )
        else:
            shutil.copy2(intermediate, output_path)

    return {
        "filename": output_path.name,
        "width": FRAME_WIDTH,
        "height": FRAME_HEIGHT,
        "fps": FPS,
        "frame_count": FRAME_COUNT,
        "duration_seconds": FRAME_COUNT / FPS,
        "placements": [
            {
                "subject": item.subject.name,
                "region": item.region,
                "size": item.size,
                "motion": item.motion,
            }
            for item in placements
        ],
        "frames": annotations,
    }


def generate(output_dir: Path) -> None:
    reference_dir = output_dir / "reference"
    video_dir = output_dir / "videos"
    reference_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    subjects = {
        "subject_a": Subject(
            "subject_a",
            reference_dir / "subject_a.png",
            (62, 46, 95, 95),
        ),
        "subject_b": Subject(
            "subject_b",
            reference_dir / "subject_b.png",
            (75, 85, 222, 222),
        ),
    }
    for subject in subjects.values():
        if not subject.image_path.is_file():
            raise FileNotFoundError(
                f"Missing reference image {subject.image_path}. Use the provided dataset assets."
            )

    manifest_rows: list[dict[str, object]] = []
    annotation_payload: dict[str, object] = {
        "schema_version": "1.0",
        "description": "Synthetic placement ground truth for FrameGuard face tests",
        "videos": [],
    }

    index = 0
    for subject in subjects.values():
        for region in REGIONS:
            for size in FACE_HEIGHTS:
                motion = MOTIONS[index % len(MOTIONS)]
                filename = f"{subject.name}_{region}_{size}.mp4"
                placements = [Placement(subject, region, size, motion, phase=index * 0.31)]
                record = _write_video(
                    video_dir / filename,
                    placements,
                    f"{subject.name} | {region} | {size} | {motion}",
                    index % 5,
                )
                annotation_payload["videos"].append(record)  # type: ignore[index]
                manifest_rows.append(
                    {
                        "filename": filename,
                        "scenario": "single_face",
                        "subjects": subject.name,
                        "region": region,
                        "size": size,
                        "motion": motion,
                        "expected_faces": 1,
                    }
                )
                index += 1

    dual_regions = [
        ("top_left", "bottom_right"),
        ("top_right", "bottom_left"),
        ("center_left", "center_right"),
        ("top_center", "bottom_left"),
        ("bottom_right", "top_left"),
        ("center", "top_right"),
        ("bottom_left", "center_right"),
        ("top_left", "center"),
    ]
    dual_sizes = [
        ("small", "large"),
        ("large", "small"),
        ("medium", "medium"),
        ("small", "medium"),
        ("medium", "large"),
        ("large", "medium"),
        ("small", "small"),
        ("large", "large"),
    ]
    for dual_index, ((region_a, region_b), (size_a, size_b)) in enumerate(
        zip(dual_regions, dual_sizes, strict=True),
        start=1,
    ):
        motion_a = MOTIONS[(index + dual_index) % len(MOTIONS)]
        motion_b = MOTIONS[(index + dual_index + 2) % len(MOTIONS)]
        filename = f"dual_subjects_{dual_index:02d}.mp4"
        placements = [
            Placement(subjects["subject_a"], region_a, size_a, motion_a, phase=0.0),
            Placement(subjects["subject_b"], region_b, size_b, motion_b, phase=math.pi),
        ]
        record = _write_video(
            video_dir / filename,
            placements,
            f"dual | A:{region_a}/{size_a} | B:{region_b}/{size_b}",
            dual_index % 5,
        )
        annotation_payload["videos"].append(record)  # type: ignore[index]
        manifest_rows.append(
            {
                "filename": filename,
                "scenario": "two_faces_reference_matching",
                "subjects": "subject_a+subject_b",
                "region": f"{region_a}+{region_b}",
                "size": f"{size_a}+{size_b}",
                "motion": f"{motion_a}+{motion_b}",
                "expected_faces": 2,
            }
        )

    with (output_dir / "manifest.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest_rows[0]))
        writer.writeheader()
        writer.writerows(manifest_rows)

    (output_dir / "annotations.json").write_text(
        json.dumps(annotation_payload, indent=2),
        encoding="utf-8",
    )
    print(f"Generated {len(manifest_rows)} videos in {video_dir}")


if __name__ == "__main__":
    generate(DEFAULT_OUTPUT)

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2

from .schemas import Finding
from .video import probe_video


def _expanded_box(
    x: int,
    y: int,
    width: int,
    height: int,
    frame_width: int,
    frame_height: int,
    padding: int = 10,
) -> tuple[int, int, int, int]:
    return (
        max(0, x - padding),
        max(0, y - padding),
        min(frame_width, x + width + padding),
        min(frame_height, y + height + padding),
    )


def _blur_region(frame, box: tuple[int, int, int, int]) -> None:
    x1, y1, x2, y2 = box
    region = frame[y1:y2, x1:x2]
    if region.size == 0:
        return
    smallest = max(1, min(region.shape[:2]))
    kernel = min(61, smallest if smallest % 2 else smallest - 1)
    if kernel < 3:
        frame[y1:y2, x1:x2] = 0
        return
    frame[y1:y2, x1:x2] = cv2.GaussianBlur(region, (kernel, kernel), 0)


def _audio_filter(findings: list[Finding], padding_ms: int = 250) -> str | None:
    filters: list[str] = []
    for finding in findings:
        if finding.modality not in {"audio", "both"}:
            continue
        start = max(0.0, (finding.start_ms - padding_ms) / 1000.0)
        end = max(start, (finding.end_ms + padding_ms) / 1000.0)
        filters.append(f"volume=0:enable='between(t,{start:.3f},{end:.3f})'")
    return ",".join(filters) or None


def render_redacted_video(
    input_path: str | Path,
    output_path: str | Path,
    findings: list[Finding],
) -> Path:
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    info = probe_video(input_path)

    with tempfile.TemporaryDirectory(prefix="frameguard-render-") as temp_dir:
        temp_video = Path(temp_dir) / "video_only.mp4"
        capture = cv2.VideoCapture(str(input_path))
        writer = cv2.VideoWriter(
            str(temp_video),
            cv2.VideoWriter_fourcc(*"mp4v"),
            info.fps,
            (info.width, info.height),
        )
        if not capture.isOpened() or not writer.isOpened():
            capture.release()
            writer.release()
            raise RuntimeError("Could not initialize video reader/writer")

        frame_index = 0
        try:
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                time_ms = int(frame_index / info.fps * 1000)
                for finding in findings:
                    if finding.modality not in {"visual", "both"}:
                        continue
                    if not finding.observations:
                        continue
                    if not (finding.start_ms <= time_ms <= finding.end_ms):
                        continue
                    observation = finding.nearest_observation(time_ms)
                    if observation is None:
                        continue
                    _blur_region(
                        frame,
                        _expanded_box(
                            observation.x,
                            observation.y,
                            observation.width,
                            observation.height,
                            info.width,
                            info.height,
                        ),
                    )
                writer.write(frame)
                frame_index += 1
        finally:
            capture.release()
            writer.release()

        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            shutil.copy2(temp_video, output_path)
            return output_path

        command = [
            ffmpeg,
            "-y",
            "-i",
            str(temp_video),
            "-i",
            str(input_path),
            "-map",
            "0:v:0",
            "-map",
            "1:a?",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            "21",
            "-pix_fmt",
            "yuv420p",
        ]
        audio_filter = _audio_filter(findings)
        if audio_filter:
            command.extend(["-af", audio_filter])
        command.extend(["-c:a", "aac", "-shortest", str(output_path)])
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg rendering failed: {result.stderr[-1600:]}")
    return output_path

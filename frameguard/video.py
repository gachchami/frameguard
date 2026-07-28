from __future__ import annotations

import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import cv2


@dataclass(frozen=True, slots=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    frame_count: int
    duration_ms: int


@dataclass(frozen=True, slots=True)
class VideoChunk:
    path: Path
    start_seconds: float
    duration_seconds: float


def probe_video(path: str | Path) -> VideoInfo:
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {path}")
    try:
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    finally:
        capture.release()
    if fps <= 0 or width <= 0 or height <= 0:
        raise ValueError(f"Invalid video metadata: {path}")
    return VideoInfo(
        width=width,
        height=height,
        fps=fps,
        frame_count=frame_count,
        duration_ms=int(frame_count / fps * 1000),
    )


def iter_frames_between(
    path: str | Path,
    start_ms: int,
    end_ms: int,
    interval_ms: int = 400,
) -> Iterator[tuple[int, object]]:
    info = probe_video(path)
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        raise ValueError(f"Could not open video: {path}")
    try:
        time_ms = max(0, start_ms)
        final_ms = min(info.duration_ms, max(time_ms, end_ms))
        while time_ms <= final_ms:
            capture.set(cv2.CAP_PROP_POS_MSEC, time_ms)
            ok, frame = capture.read()
            if not ok:
                break
            yield time_ms, frame
            time_ms += max(1, interval_ms)
    finally:
        capture.release()


def split_video(
    input_path: str | Path,
    output_dir: str | Path,
    chunk_seconds: float,
) -> list[VideoChunk]:
    """Create short MP4 chunks with their original audio retained."""

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        raise RuntimeError("ffmpeg is required to split videos")
    info = probe_video(input_path)
    total_seconds = info.duration_ms / 1000.0
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    chunks: list[VideoChunk] = []
    start = 0.0
    index = 0
    while start < total_seconds:
        duration = min(chunk_seconds, total_seconds - start)
        path = output_dir / f"chunk_{index:03d}.mp4"
        command = [
            ffmpeg,
            "-y",
            "-ss",
            f"{start:.3f}",
            "-i",
            str(input_path),
            "-t",
            f"{duration:.3f}",
            "-c:v",
            "libx264",
            "-preset",
            "ultrafast",
            "-crf",
            "20",
            "-pix_fmt",
            "yuv420p",
            "-c:a",
            "aac",
            "-ar",
            "16000",
            "-ac",
            "1",
            str(path),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"FFmpeg chunking failed: {result.stderr[-1600:]}")
        chunks.append(VideoChunk(path=path, start_seconds=start, duration_seconds=duration))
        start += duration
        index += 1
    return chunks

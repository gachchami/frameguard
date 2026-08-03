from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2

from .observability import RunEventRecorder
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


def _strong_redact_region(frame, box: tuple[int, int, int, int]) -> None:
    x1, y1, x2, y2 = box
    region = frame[y1:y2, x1:x2]
    if region.size == 0:
        return

    width = max(1, x2 - x1)
    height = max(1, y2 - y1)
    small_width = max(2, width // 18)
    small_height = max(2, height // 18)
    pixelated = cv2.resize(region, (small_width, small_height), interpolation=cv2.INTER_AREA)
    region = cv2.resize(pixelated, (width, height), interpolation=cv2.INTER_NEAREST)

    smallest = max(1, min(region.shape[:2]))
    kernel = min(81, smallest if smallest % 2 else smallest - 1)
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
    *,
    recorder: RunEventRecorder | None = None,
) -> Path:
    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    info = probe_video(input_path)

    visual_findings = [
        finding
        for finding in findings
        if finding.modality in {"visual", "both"} and finding.observations
    ]
    audio_findings = [
        finding for finding in findings if finding.modality in {"audio", "both"}
    ]

    if recorder:
        recorder.info(
            "render.started",
            frame_count=info.frame_count,
            fps=round(info.fps, 3),
            visual_findings=len(visual_findings),
            audio_findings=len(audio_findings),
        )

    redaction_applications = 0
    written_frames = 0

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
                for finding in visual_findings:
                    if not (finding.start_ms <= time_ms <= finding.end_ms):
                        continue
                    observation = finding.nearest_observation(time_ms)
                    if observation is None:
                        continue
                    _strong_redact_region(
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
                    redaction_applications += 1
                writer.write(frame)
                written_frames += 1
                frame_index += 1
        finally:
            capture.release()
            writer.release()

        ffmpeg = shutil.which("ffmpeg")
        if not ffmpeg:
            shutil.copy2(temp_video, output_path)
            if recorder:
                recorder.warning(
                    "render.ffmpeg_missing",
                    audio_redaction_applied=False,
                    output=output_path.name,
                )
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

        result = subprocess.run(command, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            if recorder:
                recorder.error(
                    "render.ffmpeg_failed",
                    return_code=result.returncode,
                    stderr_characters=len(result.stderr),
                )
            raise RuntimeError(
                f"FFmpeg rendering failed with exit code {result.returncode}. "
                "See the FrameGuard run log."
            )

    if recorder:
        recorder.info(
            "render.completed",
            output=output_path.name,
            output_bytes=output_path.stat().st_size if output_path.exists() else 0,
            written_frames=written_frames,
            redaction_applications=redaction_applications,
            audio_intervals=len(audio_findings),
        )
    return output_path

from __future__ import annotations

import shutil
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "samples" / "frameguard_demo.mp4"
WIDTH, HEIGHT, FPS, DURATION = 1280, 720, 24, 15


def _font(size: int):
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/System/Library/Fonts/Menlo.ttc",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def _frame(second: float) -> np.ndarray:
    image = Image.new("RGB", (WIDTH, HEIGHT), (20, 24, 32))
    draw = ImageDraw.Draw(image)
    title = _font(38)
    body = _font(32)
    small = _font(24)

    draw.rectangle((45, 45, WIDTH - 45, HEIGHT - 45), outline=(100, 115, 145), width=3)
    draw.text((75, 70), "FrameGuard demo recording", font=title, fill=(225, 230, 240))

    if second < 5:
        draw.text((75, 160), "Terminal", font=body, fill=(150, 210, 255))
        draw.text((100, 245), "$ export API_KEY=sk-demo-83hhd8282hd91jd82", font=body, fill=(235, 235, 235))
        draw.text((100, 310), "$ uv run python app.py", font=body, fill=(235, 235, 235))
    elif second < 10:
        draw.text((75, 160), "Customer profile", font=body, fill=(150, 210, 255))
        draw.text((100, 245), "Email: alice@example.com", font=body, fill=(235, 235, 235))
        draw.text((100, 310), "Internal server: 192.168.1.24", font=body, fill=(235, 235, 235))
    else:
        draw.text((75, 160), "Support dashboard", font=body, fill=(150, 210, 255))
        draw.text((100, 245), "Account: CUST-493821", font=body, fill=(235, 235, 235))
        draw.text((100, 310), "Status: verification pending", font=body, fill=(235, 235, 235))

    draw.text((75, HEIGHT - 105), f"Time: {second:04.1f}s", font=small, fill=(155, 165, 185))
    return cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2BGR)


def _run(command: list[str]) -> None:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr[-2000:])


def main() -> None:
    ffmpeg = shutil.which("ffmpeg")
    espeak = shutil.which("espeak-ng") or shutil.which("espeak")
    if not ffmpeg or not espeak:
        raise SystemExit("This script requires ffmpeg and espeak/espeak-ng.")

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="frameguard-demo-") as temp:
        temp_dir = Path(temp)
        video_only = temp_dir / "video_only.mp4"
        writer = cv2.VideoWriter(
            str(video_only),
            cv2.VideoWriter_fourcc(*"mp4v"),
            FPS,
            (WIDTH, HEIGHT),
        )
        for frame_index in range(FPS * DURATION):
            writer.write(_frame(frame_index / FPS))
        writer.release()

        lines = [
            ("The screen briefly exposes an API key.", 600),
            ("The customer email is alice at example dot com.", 5400),
            ("Call the customer at plus nine one, nine eight seven six five, four three two one zero.", 10100),
        ]
        delayed_inputs: list[str] = []
        filter_parts: list[str] = []
        for index, (text, delay_ms) in enumerate(lines):
            wav = temp_dir / f"speech_{index}.wav"
            _run([espeak, "-s", "145", "-w", str(wav), text])
            delayed_inputs.extend(["-i", str(wav)])
            filter_parts.append(f"[{index}:a]adelay={delay_ms}|{delay_ms}[a{index}]")
        labels = "".join(f"[a{index}]" for index in range(len(lines)))
        mixed_audio = temp_dir / "mixed.wav"
        _run(
            [
                ffmpeg,
                "-y",
                *delayed_inputs,
                "-filter_complex",
                ";".join(filter_parts) + f";{labels}amix=inputs={len(lines)},apad,atrim=0:{DURATION}[out]",
                "-map",
                "[out]",
                "-ar",
                "16000",
                "-ac",
                "1",
                str(mixed_audio),
            ]
        )
        _run(
            [
                ffmpeg,
                "-y",
                "-i",
                str(video_only),
                "-i",
                str(mixed_audio),
                "-c:v",
                "libx264",
                "-preset",
                "veryfast",
                "-crf",
                "20",
                "-pix_fmt",
                "yuv420p",
                "-c:a",
                "aac",
                "-shortest",
                str(OUTPUT),
            ]
        )
    print(f"Created {OUTPUT}")


if __name__ == "__main__":
    main()

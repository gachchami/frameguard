from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

MODEL_NAME = "face_detection_yunet_2023mar.onnx"
EXPECTED_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"
DOWNLOAD_URLS = (
    "https://github.com/opencv/opencv_zoo/raw/refs/heads/main/"
    "models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    "https://huggingface.co/opencv/face_detection_yunet/resolve/main/"
    "face_detection_yunet_2023mar.onnx",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def download(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    errors: list[str] = []

    with tempfile.TemporaryDirectory(prefix="frameguard-yunet-") as temp_dir:
        temporary = Path(temp_dir) / MODEL_NAME
        for url in DOWNLOAD_URLS:
            try:
                print(f"Downloading {MODEL_NAME} from {url}")
                request = urllib.request.Request(
                    url,
                    headers={"User-Agent": "FrameGuard-YuNet-Downloader/1.0"},
                )
                with urllib.request.urlopen(request, timeout=60) as response:
                    with temporary.open("wb") as output:
                        shutil.copyfileobj(response, output)
                actual = sha256(temporary)
                if actual != EXPECTED_SHA256:
                    raise RuntimeError(
                        f"SHA256 mismatch: expected {EXPECTED_SHA256}, got {actual}"
                    )
                shutil.copy2(temporary, destination)
                print(f"Saved verified model to {destination}")
                return
            except (OSError, RuntimeError, urllib.error.URLError) as exc:
                errors.append(f"{url}: {exc}")
                temporary.unlink(missing_ok=True)

    joined = "\n".join(errors)
    raise RuntimeError(f"Could not download YuNet model:\n{joined}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download and verify the OpenCV Zoo YuNet ONNX model."
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models") / MODEL_NAME,
    )
    args = parser.parse_args()

    if args.output.exists() and sha256(args.output) == EXPECTED_SHA256:
        print(f"Model already present and verified: {args.output}")
        return 0

    try:
        download(args.output)
    except RuntimeError as exc:
        print(exc, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

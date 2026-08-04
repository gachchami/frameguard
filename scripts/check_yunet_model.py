from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import cv2

from frameguard.face_tracking import DEFAULT_FACE_MODEL, YuNetFaceDetector

EXPECTED_SHA256 = "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate FrameGuard's YuNet model.")
    parser.add_argument("--model", type=Path, default=DEFAULT_FACE_MODEL)
    parser.add_argument("--image", type=Path)
    args = parser.parse_args()

    if not args.model.is_file():
        raise SystemExit(f"Missing model: {args.model}")

    actual = sha256(args.model)
    if actual != EXPECTED_SHA256:
        raise SystemExit(
            f"Invalid model checksum: expected {EXPECTED_SHA256}, got {actual}"
        )

    detector = YuNetFaceDetector(args.model)
    print(f"OpenCV: {cv2.__version__}")
    print(f"YuNet model verified: {args.model}")

    if args.image:
        image = cv2.imread(str(args.image))
        if image is None:
            raise SystemExit(f"Could not read image: {args.image}")
        detections = detector.detect(image)
        print(f"Faces detected: {len(detections)}")
        for index, detection in enumerate(detections, start=1):
            print(
                f"face_{index:03d}: box=({detection.x}, {detection.y}, "
                f"{detection.width}, {detection.height}) "
                f"confidence={detection.confidence:.3f}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

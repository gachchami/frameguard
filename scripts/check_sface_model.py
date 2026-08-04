from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2

from frameguard.face_reference import DEFAULT_SFACE_MODEL


def main() -> int:
    parser = argparse.ArgumentParser(description="Load the SFace ONNX model with OpenCV.")
    parser.add_argument("--model", type=Path, default=DEFAULT_SFACE_MODEL)
    args = parser.parse_args()

    if not args.model.is_file():
        print(f"Missing SFace model: {args.model}", file=sys.stderr)
        return 1
    if not hasattr(cv2, "FaceRecognizerSF"):
        print("OpenCV does not expose cv2.FaceRecognizerSF", file=sys.stderr)
        return 1

    recognizer = cv2.FaceRecognizerSF.create(str(args.model), "")
    print("SFace model loaded successfully")
    print("Model:", args.model)
    print("OpenCV:", cv2.__version__)
    print("Recognizer:", type(recognizer).__name__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

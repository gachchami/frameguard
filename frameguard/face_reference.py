from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import cv2
import numpy as np

if TYPE_CHECKING:
    from .face_tracking import FaceDetection, YuNetFaceDetector

DEFAULT_SFACE_MODEL = (
    Path(__file__).resolve().parent.parent
    / "models"
    / "face_recognition_sface_2021dec.onnx"
)

DEFAULT_COSINE_THRESHOLD = 0.363


class ReferenceFaceMatcher:
    """Match detected faces against one user-supplied reference image.

    The reference embedding is kept in process memory only. FrameGuard does not
    persist the uploaded reference image, aligned face crop, or embedding to the
    audit report or operational log.
    """

    def __init__(
        self,
        *,
        reference_image_path: str | Path,
        detector: "YuNetFaceDetector",
        model_path: str | Path = DEFAULT_SFACE_MODEL,
        cosine_threshold: float = DEFAULT_COSINE_THRESHOLD,
    ) -> None:
        self.reference_image_path = Path(reference_image_path)
        self.model_path = Path(model_path)
        self.cosine_threshold = float(cosine_threshold)

        if not self.reference_image_path.is_file():
            raise FileNotFoundError(
                f"Reference face image does not exist: {self.reference_image_path}"
            )
        if not self.model_path.is_file():
            raise FileNotFoundError(
                "SFace recognition model is missing: "
                f"{self.model_path}. Run scripts/download_sface_model.py on an "
                "internet-connected machine, commit the model, and pull it into "
                "the target container."
            )
        if not hasattr(cv2, "FaceRecognizerSF"):
            raise RuntimeError(
                "This OpenCV build does not provide cv2.FaceRecognizerSF. "
                "Install opencv-python-headless>=4.10."
            )

        reference_image = cv2.imread(str(self.reference_image_path))
        if reference_image is None:
            raise ValueError(
                f"Could not decode reference face image: {self.reference_image_path}"
            )

        reference_faces = detector.detect(reference_image)
        if not reference_faces:
            raise ValueError(
                "No face was detected in the uploaded reference image. Upload a "
                "clear, front-facing image with one visible face."
            )
        if len(reference_faces) > 1:
            raise ValueError(
                "Multiple faces were detected in the uploaded reference image. "
                "Crop the image so it contains exactly one face."
            )

        self._recognizer = cv2.FaceRecognizerSF.create(
            model=str(self.model_path),
            config="",
        )
        self._reference_feature = self._extract_feature(
            reference_image,
            reference_faces[0],
        )

    def _extract_feature(
        self,
        image: np.ndarray,
        detection: "FaceDetection",
    ) -> np.ndarray:
        face_row = detection.to_yunet_row()
        aligned = self._recognizer.alignCrop(image, face_row)
        feature = self._recognizer.feature(aligned)
        if feature is None or feature.size == 0:
            raise RuntimeError("SFace returned an empty face embedding")
        return feature

    def score(
        self,
        frame: np.ndarray,
        detection: "FaceDetection",
    ) -> float:
        candidate_feature = self._extract_feature(frame, detection)
        return float(
            self._recognizer.match(
                self._reference_feature,
                candidate_feature,
                cv2.FaceRecognizerSF_FR_COSINE,
            )
        )

    def matches(
        self,
        frame: np.ndarray,
        detection: "FaceDetection",
    ) -> tuple[bool, float]:
        similarity = self.score(frame, detection)
        return similarity >= self.cosine_threshold, similarity

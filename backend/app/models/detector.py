"""Face detection abstraction used by enrollment and recognition services.

InsightFace is preferred when installed. OpenCV's bundled Haar cascade is kept
as a lightweight CPU fallback so the application can be developed and tested
without downloading model weights.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class DetectedFace:
    """A detected face represented by its pixel bounding box."""

    bbox: tuple[int, int, int, int]
    confidence: float
    landmarks: Any | None = None


class FaceDetector:
    """Detect faces with InsightFace when available, otherwise OpenCV Haar."""

    def __init__(self, confidence_threshold: float = 0.5, use_insightface: bool = True) -> None:
        self.confidence_threshold = confidence_threshold
        self._insightface = None
        self.backend = "opencv-haar"
        if use_insightface:
            self._try_load_insightface()
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        self._cascade = cv2.CascadeClassifier(str(cascade_path))
        if self._cascade.empty():
            raise RuntimeError(f"Unable to load face detector cascade: {cascade_path}")

    def _try_load_insightface(self) -> None:
        try:
            from insightface.app import FaceAnalysis  # type: ignore

            model = FaceAnalysis(name="buffalo_l", providers=["CPUExecutionProvider"])
            model.prepare(ctx_id=0, det_size=(640, 640))
            self._insightface = model
            self.backend = "insightface-scrfd"
        except (ImportError, RuntimeError, OSError):
            self._insightface = None

    def detect(self, image: np.ndarray) -> list[DetectedFace]:
        """Return all faces in a BGR image, ordered by left-to-right position."""
        if image is None or image.size == 0:
            raise ValueError("Image is empty")
        if self._insightface is not None:
            faces = self._insightface.get(image)
            result = []
            for face in faces:
                score = float(getattr(face, "det_score", 0.0))
                if score >= self.confidence_threshold:
                    x1, y1, x2, y2 = [int(v) for v in face.bbox]
                    result.append(DetectedFace((x1, y1, x2 - x1, y2 - y1), score, getattr(face, "kps", None)))
            return sorted(result, key=lambda item: item.bbox[0])

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        boxes = self._cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return [DetectedFace(tuple(int(v) for v in box), 1.0) for box in sorted(boxes, key=lambda item: item[0])]


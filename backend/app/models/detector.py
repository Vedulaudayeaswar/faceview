"""OpenCV YuNet face detection."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class DetectedFace:
    """A YuNet face detection in x, y, width, height pixel coordinates."""

    bbox: tuple[int, int, int, int]
    confidence: float
    landmarks: np.ndarray | None = None
    embedding: np.ndarray | None = None


class FaceDetector:
    """Detect faces with OpenCV Zoo's YuNet ONNX model."""

    def __init__(self, confidence_threshold: float = 0.5, model_path: str | Path | None = None) -> None:
        path = Path(model_path) if model_path else Path(__file__).resolve().parents[2] / "models" / "face_detection_yunet_2023mar.onnx"
        if not path.exists():
            raise RuntimeError(f"YuNet model not found: {path}. Run python backend/scripts/download_models.py")
        self.confidence_threshold = confidence_threshold
        self.model_path = str(path)
        self.backend = "opencv-yunet"
        self._detector = cv2.FaceDetectorYN_create(self.model_path, "", (320, 320), confidence_threshold, 0.3, 5000)

    def detect(self, image: np.ndarray) -> list[DetectedFace]:
        if image is None or image.size == 0:
            raise ValueError("Image is empty")
        height, width = image.shape[:2]
        self._detector.setInputSize((width, height))
        _, detections = self._detector.detect(image)
        if detections is None:
            return []
        result = []
        for row in detections:
            x, y, w, h, score = row[:5]
            if float(score) < self.confidence_threshold:
                continue
            landmarks = np.asarray(row[5:15], dtype=np.float32)
            result.append(DetectedFace((int(x), int(y), int(w), int(h)), float(score), landmarks))
        return sorted(result, key=lambda item: item.bbox[0])

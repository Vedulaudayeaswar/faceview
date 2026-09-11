"""Face-specific YOLOv8 face detection using OpenCV DNN."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class DetectedFace:
    """A detected face represented by an x, y, width, height box."""

    bbox: tuple[int, int, int, int]
    confidence: float
    landmarks: np.ndarray | None = None
    embedding: np.ndarray | None = None


class FaceDetector:
    """Run the WIDERFace-trained YOLOv8n-Face ONNX checkpoint.

    This is a face-specific YOLO checkpoint, not a COCO person detector.
    """

    def __init__(
        self,
        confidence_threshold: float = 0.45,
        nms_threshold: float = 0.45,
        model_path: str | Path | None = None,
        input_size: int = 640,
    ) -> None:
        path = Path(model_path) if model_path else Path(__file__).resolve().parents[2] / "models" / "yolov8n-face-lindevs.onnx"
        if not path.exists():
            raise RuntimeError(f"YOLO face model not found: {path}. Run python backend/scripts/download_models.py")
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.input_size = input_size
        self.model_path = str(path)
        self.backend = "yolov8n-face-widerface"
        self._model = cv2.dnn.readNetFromONNX(self.model_path)

    def detect(self, image: np.ndarray) -> list[DetectedFace]:
        """Return all face boxes after confidence filtering and NMS."""
        if image is None or image.size == 0:
            raise ValueError("Image is empty")
        height, width = image.shape[:2]
        length = max(height, width)
        scale = length / self.input_size
        padded = cv2.copyMakeBorder(image, 0, length - height, 0, length - width, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        blob = cv2.dnn.blobFromImage(padded, 1 / 255.0, (self.input_size, self.input_size), swapRB=True)
        self._model.setInput(blob)
        output = self._model.forward()
        predictions = cv2.transpose(output[0])

        boxes: list[list[float]] = []
        scores: list[float] = []
        for prediction in predictions:
            score = float(np.max(prediction[4:]))
            if score < self.confidence_threshold:
                continue
            center_x, center_y, box_width, box_height = prediction[:4]
            x = float((center_x - box_width / 2) * scale)
            y = float((center_y - box_height / 2) * scale)
            boxes.append([x, y, float(box_width * scale), float(box_height * scale)])
            scores.append(score)

        indices = cv2.dnn.NMSBoxes(boxes, scores, self.confidence_threshold, self.nms_threshold)
        faces = []
        for index in np.asarray(indices).reshape(-1):
            x, y, box_width, box_height = boxes[int(index)]
            left = max(0, int(round(x)))
            top = max(0, int(round(y)))
            right = min(width, int(round(x + box_width)))
            bottom = min(height, int(round(y + box_height)))
            if right > left and bottom > top:
                faces.append(DetectedFace((left, top, right - left, bottom - top), scores[int(index)]))
        return sorted(faces, key=lambda item: item.bbox[0])

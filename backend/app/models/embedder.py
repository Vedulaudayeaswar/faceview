"""OpenCV SFace feature extraction and embedding normalization."""

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


class EmbeddingModelUnavailable(RuntimeError):
    """Raised when the OpenCV SFace model cannot be loaded."""


def normalize_embedding(vector: np.ndarray) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(values))
    if norm == 0.0:
        raise ValueError("Cannot normalize a zero-length embedding")
    return values / norm


@dataclass(frozen=True)
class FaceEmbedding:
    vector: np.ndarray
    model_name: str
    dimension: int


class SFaceEmbedder:
    """Generate normalized 128-D embeddings with OpenCV FaceRecognizerSF."""

    def __init__(self, model_path: str | Path | None = None) -> None:
        path = Path(model_path) if model_path else Path(__file__).resolve().parents[2] / "models" / "face_recognition_sface_2021dec.onnx"
        if not path.exists():
            raise EmbeddingModelUnavailable(f"SFace model not found: {path}. Run python backend/scripts/download_models.py")
        try:
            self._recognizer = cv2.FaceRecognizerSF_create(str(path), "")
        except Exception as exc:
            raise EmbeddingModelUnavailable(f"Unable to load OpenCV SFace model: {exc}") from exc
        self.model_name = "opencv-sface"

    def embed(self, image: np.ndarray, bbox: tuple[int, int, int, int] | None = None, landmarks: np.ndarray | None = None) -> FaceEmbedding:
        if bbox is None:
            raise ValueError("SFace embedding requires a detected face bounding box")
        x, y, w, h = bbox
        if landmarks is None or len(landmarks) != 10:
            landmarks = self._fallback_landmarks(x, y, w, h)
        face_info = np.asarray([x, y, w, h, 1.0, *landmarks.tolist()], dtype=np.float32)
        aligned = self._recognizer.alignCrop(image, face_info)
        feature = self._recognizer.feature(aligned)
        vector = normalize_embedding(feature)
        return FaceEmbedding(vector, self.model_name, int(vector.shape[0]))

    @staticmethod
    def _fallback_landmarks(x: int, y: int, w: int, h: int) -> np.ndarray:
        return np.asarray([x + .30*w, y + .38*h, x + .70*w, y + .38*h, x + .50*w, y + .55*h, x + .35*w, y + .76*h, x + .65*w, y + .76*h], dtype=np.float32)


def embed_detected(embedder, image: np.ndarray, face) -> FaceEmbedding:
    """Call modern landmark-aware embedders while keeping test doubles compatible."""
    try:
        return embedder.embed(image, face.bbox, face.landmarks)
    except TypeError:
        return embedder.embed(image, face.bbox)

"""Fixed pretrained FaceNet embedding extraction."""

import os
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


class EmbeddingModelUnavailable(RuntimeError):
    """Raised when the pretrained FaceNet model cannot be loaded."""


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


class FaceNetEmbedder:
    """Produce 512-D FaceNet embeddings with InceptionResnetV1/VGGFace2.

    YOLO supplies the face box. The face is clipped, padded to a square,
    resized to FaceNet's required 160x160 RGB input, and prewhitened.
    """

    def __init__(self, cache_dir: str | Path | None = None) -> None:
        cache = Path(cache_dir) if cache_dir else Path(__file__).resolve().parents[2] / "models" / "facenet-cache"
        cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("TORCH_HOME", str(cache))
        try:
            import torch
            from facenet_pytorch import InceptionResnetV1
        except ImportError as exc:
            raise EmbeddingModelUnavailable("FaceNet dependencies are missing. Install backend/requirements.txt and run python backend/scripts/download_models.py") from exc
        self._torch = torch
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        try:
            self._model = InceptionResnetV1(pretrained="vggface2").eval().to(self.device)
        except Exception as exc:
            raise EmbeddingModelUnavailable(f"Unable to load pretrained FaceNet VGGFace2 weights: {exc}") from exc
        self.model_name = "facenet-inceptionresnetv1-vggface2"

    def embed(self, image: np.ndarray, bbox: tuple[int, int, int, int] | None = None, landmarks=None) -> FaceEmbedding:
        del landmarks
        if bbox is None:
            raise ValueError("FaceNet embedding requires a detected face bounding box")
        face = self._preprocess(image, bbox)
        tensor = self._torch.from_numpy(face).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with self._torch.no_grad():
            vector = self._model(tensor).cpu().numpy()[0]
        values = normalize_embedding(vector)
        return FaceEmbedding(values, self.model_name, int(values.shape[0]))

    @staticmethod
    def _preprocess(image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
        x, y, width, height = bbox
        image_height, image_width = image.shape[:2]
        padding = int(max(width, height) * 0.25)
        left, top = max(0, x - padding), max(0, y - padding)
        right, bottom = min(image_width, x + width + padding), min(image_height, y + height + padding)
        crop = image[top:bottom, left:right]
        if crop.size == 0 or min(crop.shape[:2]) < 12:
            raise ValueError("Detected face is too small for FaceNet embedding")
        rgb = cv2.cvtColor(cv2.resize(crop, (160, 160), interpolation=cv2.INTER_CUBIC), cv2.COLOR_BGR2RGB)
        return (rgb.astype(np.float32) - 127.5) / 128.0


def embed_detected(embedder, image: np.ndarray, face) -> FaceEmbedding:
    """Use a detected YOLO face box with FaceNet; preserve simple test doubles."""
    try:
        return embedder.embed(image, face.bbox, face.landmarks)
    except TypeError:
        return embedder.embed(image, face.bbox)

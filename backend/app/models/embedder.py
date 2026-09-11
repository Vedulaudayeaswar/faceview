"""Pre-trained face embedding generation.

The production implementation uses InsightFace's ArcFace embedding exposed by
``FaceAnalysis``. No identity-specific training happens here.
"""

from dataclasses import dataclass

import numpy as np
import cv2


class EmbeddingModelUnavailable(RuntimeError):
    """Raised when the pre-trained embedding model cannot be loaded."""


def normalize_embedding(vector: np.ndarray) -> np.ndarray:
    """Return a float32 unit vector suitable for inner-product search."""
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


class ArcFaceEmbedder:
    """Generate normalized ArcFace vectors using a fixed InsightFace model."""

    def __init__(self, model_name: str = "buffalo_l") -> None:
        try:
            from insightface.app import FaceAnalysis  # type: ignore
        except ImportError as exc:
            raise EmbeddingModelUnavailable(
                "InsightFace is not installed. Install backend requirements and model weights."
            ) from exc
        try:
            self._app = FaceAnalysis(name=model_name, providers=["CPUExecutionProvider"])
            self._app.prepare(ctx_id=0, det_size=(640, 640))
        except Exception as exc:  # model download/provider errors vary by platform
            raise EmbeddingModelUnavailable(f"Unable to load embedding model '{model_name}': {exc}") from exc
        self.model_name = model_name

    def embed(self, image: np.ndarray, bbox: tuple[int, int, int, int] | None = None) -> FaceEmbedding:
        """Extract one normalized embedding, optionally selecting a known box."""
        source = image
        target_bbox = bbox
        if bbox is not None:
            # Video faces can be much smaller than enrollment faces. Crop with
            # context and upscale before detection/alignment so ArcFace gets a
            # useful face patch instead of a 20-30px face in a large frame.
            x, y, w, h = bbox
            height, width = image.shape[:2]
            padding = int(max(w, h) * 0.35)
            left = max(0, x - padding)
            top = max(0, y - padding)
            right = min(width, x + w + padding)
            bottom = min(height, y + h + padding)
            source = image[top:bottom, left:right]
            if source.size == 0:
                raise ValueError("Detected face crop is empty")
            scale = max(1.0, 160.0 / max(1, min(source.shape[:2])))
            if scale > 1.0:
                source = cv2.resize(source, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            target_bbox = None

        faces = self._app.get(source)
        if not faces:
            raise ValueError("No face detected")
        face = faces[0]
        if target_bbox is not None:
            x, y, w, h = target_bbox
            target = np.array([x, y, x + w, y + h], dtype=np.float32)
            face = min(faces, key=lambda item: float(np.linalg.norm(np.asarray(item.bbox) - target)))
        raw = getattr(face, "embedding", None)
        if raw is None:
            raise RuntimeError("InsightFace did not return an embedding")
        vector = normalize_embedding(raw)
        return FaceEmbedding(vector, self.model_name, int(vector.shape[0]))

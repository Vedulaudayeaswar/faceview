"""Single-image enrollment validation and embedding preparation."""

from dataclasses import dataclass
from typing import Protocol

import cv2
import numpy as np

from app.models.embedder import embed_detected


class EnrollmentError(ValueError):
    """A user-correctable enrollment validation error."""


class DetectorProtocol(Protocol):
    def detect(self, image: np.ndarray) -> list: ...


class EmbedderProtocol(Protocol):
    def embed(self, image: np.ndarray, bbox: tuple[int, int, int, int] | None = None): ...


@dataclass(frozen=True)
class PreparedEnrollment:
    image: np.ndarray
    embedding: np.ndarray
    model_name: str


@dataclass(frozen=True)
class BatchFailure:
    filename: str
    reason: str


@dataclass(frozen=True)
class BatchEnrollmentReport:
    total_files: int
    successful: int
    failed: int
    failures: list[BatchFailure]


class EnrollmentService:
    """Validate one uploaded photo and produce data ready for persistence."""

    def __init__(self, detector: DetectorProtocol, embedder: EmbedderProtocol) -> None:
        self.detector = detector
        self.embedder = embedder

    def prepare_single(self, image_bytes: bytes) -> PreparedEnrollment:
        """Decode an upload, require exactly one face, and create one embedding."""
        image = cv2.imdecode(np.frombuffer(image_bytes, dtype=np.uint8), cv2.IMREAD_COLOR)
        if image is None or image.size == 0:
            raise EnrollmentError("Invalid image. Please upload a valid image file.")
        faces = self.detector.detect(image)
        if len(faces) == 0:
            raise EnrollmentError("No face detected. Please upload another image.")
        if len(faces) > 1:
            raise EnrollmentError("Multiple faces detected. Please upload an image containing only one person.")
        face = faces[0]
        result = embed_detected(self.embedder, image, face)
        return PreparedEnrollment(image=image, embedding=result.vector, model_name=result.model_name)

    def validate_batch(self, files: list[tuple[str, bytes]]) -> BatchEnrollmentReport:
        failures: list[BatchFailure] = []
        successful = 0
        for filename, contents in files:
            try:
                self.prepare_single(contents)
                successful += 1
            except (EnrollmentError, ValueError, RuntimeError) as exc:
                failures.append(BatchFailure(filename, str(exc)))
        return BatchEnrollmentReport(len(files), successful, len(failures), failures)

"""Camera capture and frame-level recognition utilities."""

import time
from dataclasses import dataclass
from urllib.parse import urlparse

import cv2
import numpy as np

from app.models.embedder import normalize_embedding
from app.services.recognition_service import RecognitionService, RecognitionResult


class CameraUnavailable(RuntimeError):
    pass


def validate_rtsp_url(url: str) -> str:
    """Validate an RTSP URL without exposing or logging credentials."""
    parsed = urlparse(url)
    if parsed.scheme.lower() != "rtsp" or not parsed.hostname:
        raise ValueError("Invalid RTSP URL. Use rtsp://host/path.")
    return url


class CameraCapture:
    def __init__(self, source: int | str) -> None:
        self.source = source
        self.capture = cv2.VideoCapture(source)
        if not self.capture.isOpened():
            self.capture.release()
            raise CameraUnavailable(f"Unable to open camera source: {source}")

    def read(self) -> np.ndarray:
        ok, frame = self.capture.read()
        if not ok or frame is None:
            raise CameraUnavailable("Camera disconnected or returned an invalid frame")
        return frame

    def release(self) -> None:
        self.capture.release()


@dataclass(frozen=True)
class FrameFaceResult:
    bbox: tuple[int, int, int, int]
    identity_id: int | None
    label: str
    similarity: float


@dataclass(frozen=True)
class ProcessedFrame:
    frame: np.ndarray
    faces: list[FrameFaceResult]
    latency_ms: float


class FrameRecognitionProcessor:
    def __init__(self, detector, embedder, recognizer: RecognitionService, threshold: float) -> None:
        self.detector = detector
        self.embedder = embedder
        self.recognizer = recognizer
        self.threshold = threshold

    def process(self, frame: np.ndarray) -> ProcessedFrame:
        started = time.perf_counter()
        faces = self.detector.detect(frame)
        results = []
        annotated = frame.copy()
        for face in faces:
            embedding = normalize_embedding(face.embedding) if face.embedding is not None else self.embedder.embed(frame, face.bbox).vector
            outcome: RecognitionResult = self.recognizer.classify(
                embedding, self.threshold
            )
            label = f"{outcome.result} {outcome.similarity:.1%}"
            x, y, w, h = face.bbox
            color = (0, 180, 0) if outcome.result == "KNOWN" else (0, 0, 220)
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            cv2.putText(annotated, label, (x, max(20, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            results.append(FrameFaceResult(face.bbox, outcome.identity_id, label, outcome.similarity))
        elapsed = (time.perf_counter() - started) * 1000
        return ProcessedFrame(annotated, results, elapsed)

import cv2
import numpy as np

from app.models.detector import DetectedFace
from app.services.enrollment_service import EnrollmentService


def encoded_image() -> bytes:
    ok, encoded = cv2.imencode(".png", np.full((80, 80, 3), 127, dtype=np.uint8))
    assert ok
    return encoded.tobytes()


class FakeDetector:
    def __init__(self, count: int):
        self.count = count

    def detect(self, image):
        return [DetectedFace((5, 5, 20, 20), 0.9) for _ in range(self.count)]


class FakeEmbedding:
    vector = np.array([1.0, 0.0], dtype=np.float32)
    model_name = "test-arcface"


class FakeEmbedder:
    def embed(self, image, bbox=None):
        return FakeEmbedding()


def test_batch_report_contains_per_file_failures() -> None:
    files = [("valid.png", encoded_image()), ("no-face.png", encoded_image())]
    service = EnrollmentService(FakeDetector(1), FakeEmbedder())
    # Make the second validation fail without changing the public API.
    calls = {"count": 0}

    def detect(_image):
        calls["count"] += 1
        return FakeDetector(1 if calls["count"] == 1 else 0).detect(_image)

    service.detector.detect = detect
    report = service.validate_batch(files)
    assert (report.total_files, report.successful, report.failed) == (2, 1, 1)
    assert report.failures[0].filename == "no-face.png"

import cv2
import numpy as np
import pytest

from app.models.detector import DetectedFace
from app.services.enrollment_service import EnrollmentError, EnrollmentService


def encoded_image() -> bytes:
    ok, encoded = cv2.imencode(".png", np.full((80, 80, 3), 127, dtype=np.uint8))
    assert ok
    return encoded.tobytes()


class FakeDetector:
    def __init__(self, count: int):
        self.count = count

    def detect(self, image):
        return [DetectedFace((5 + i * 20, 5, 20, 20), 0.9) for i in range(self.count)]


class FakeEmbedding:
    vector = np.array([1.0, 0.0], dtype=np.float32)
    model_name = "test-sface"


class FakeEmbedder:
    def embed(self, image, bbox=None):
        return FakeEmbedding()


def test_valid_single_image_is_prepared() -> None:
    result = EnrollmentService(FakeDetector(1), FakeEmbedder()).prepare_single(encoded_image())
    assert result.model_name == "test-sface"
    assert result.embedding.tolist() == [1.0, 0.0]


@pytest.mark.parametrize(
    ("count", "message"),
    [
        (0, "No face detected"),
        (2, "Multiple faces detected"),
    ],
)
def test_enrollment_requires_exactly_one_face(count: int, message: str) -> None:
    with pytest.raises(EnrollmentError, match=message):
        EnrollmentService(FakeDetector(count), FakeEmbedder()).prepare_single(encoded_image())

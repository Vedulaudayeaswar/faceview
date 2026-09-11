import cv2
import numpy as np

from app.models.detector import FaceDetector


def test_detector_returns_no_faces_for_blank_image() -> None:
    detector = FaceDetector()
    image = np.zeros((240, 320, 3), dtype=np.uint8)
    assert detector.detect(image) == []


def test_detector_rejects_empty_image() -> None:
    detector = FaceDetector()
    try:
        detector.detect(np.array([], dtype=np.uint8))
    except ValueError as exc:
        assert str(exc) == "Image is empty"
    else:
        raise AssertionError("Expected ValueError for an empty image")

import numpy as np
import pytest

from app.services.camera_service import CameraCapture, CameraUnavailable, FrameRecognitionProcessor
from app.services.recognition_service import RecognitionService
from app.services.vector_store import VectorStore


def test_invalid_camera_source_is_reported() -> None:
    with pytest.raises(CameraUnavailable):
        CameraCapture("definitely-not-a-real-camera-source")


def test_frame_processor_supports_zero_faces(tmp_path) -> None:
    class Detector:
        def detect(self, frame):
            return []

    store = VectorStore(2, tmp_path / "vectors.npz")
    processor = FrameRecognitionProcessor(Detector(), object(), RecognitionService(store), 0.6)
    result = processor.process(np.zeros((120, 160, 3), dtype=np.uint8))
    assert result.faces == []
    assert result.latency_ms >= 0


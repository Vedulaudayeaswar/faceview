import numpy as np

from app.services.recognition_service import RecognitionService
from app.services.vector_store import VectorStore


def test_recognition_returns_nearest_identity(tmp_path) -> None:
    store = VectorStore(2, tmp_path / "vectors.npz")
    store.add(7, np.array([1, 0], dtype=np.float32))
    service = RecognitionService(store, {7: 42})
    result = service.recognize(np.array([0.98, 0.02], dtype=np.float32))
    assert result.vector_id == 7
    assert result.identity_id == 42
    assert result.similarity > 0.9


def test_recognition_handles_empty_index(tmp_path) -> None:
    result = RecognitionService(VectorStore(2, tmp_path / "vectors.npz")).recognize(np.array([1, 0]))
    assert result.identity_id is None
    assert result.vector_id is None


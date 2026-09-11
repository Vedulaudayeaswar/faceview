import numpy as np
import pytest

from app.services.recognition_service import RecognitionService
from app.services.vector_store import VectorStore


def test_low_similarity_is_unknown(tmp_path) -> None:
    store = VectorStore(2, tmp_path / "vectors.npz")
    store.add(1, np.array([1, 0], dtype=np.float32))
    service = RecognitionService(store, {1: 99})
    result = service.classify(np.array([0.6, 0.8], dtype=np.float32), threshold=0.9)
    assert result.result == "UNKNOWN"
    assert result.identity_id is None
    assert result.similarity == pytest.approx(0.6)


def test_similarity_at_threshold_is_known(tmp_path) -> None:
    store = VectorStore(2, tmp_path / "vectors.npz")
    store.add(1, np.array([1, 0], dtype=np.float32))
    service = RecognitionService(store, {1: 99})
    assert service.classify(np.array([1, 0]), threshold=0.75).result == "KNOWN"


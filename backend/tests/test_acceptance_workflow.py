import numpy as np

from app.services.recognition_service import RecognitionService
from app.services.vector_store import VectorStore


def test_add_recognize_add_delete_without_retraining(tmp_path) -> None:
    store = VectorStore(3, tmp_path / "acceptance.npz")
    recognizer = RecognitionService(store)

    # Add person A: only database/vector operations are performed.
    store.add(1, np.array([1, 0, 0], dtype=np.float32))
    recognizer.register_vector_identity(1, 101)
    assert recognizer.classify(np.array([1, 0, 0]), 0.8).identity_id == 101

    # Add person B while the service is running: no model object is recreated.
    store.add(2, np.array([0, 1, 0], dtype=np.float32))
    recognizer.register_vector_identity(2, 202)
    assert recognizer.classify(np.array([0, 1, 0]), 0.8).identity_id == 202

    # Delete B: the same running recognizer now returns UNKNOWN.
    store.remove(2)
    recognizer.remove_vector_identity(2)
    result = recognizer.classify(np.array([0, 1, 0]), 0.8)
    assert result.result == "UNKNOWN"
    assert result.identity_id is None


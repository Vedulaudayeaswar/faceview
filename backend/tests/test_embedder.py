import numpy as np
import pytest

from app.models.embedder import normalize_embedding


def test_normalize_embedding_returns_unit_float32_vector() -> None:
    vector = normalize_embedding(np.array([3, 4], dtype=np.float64))
    assert vector.dtype == np.float32
    assert np.allclose(vector, [0.6, 0.8])
    assert np.isclose(np.linalg.norm(vector), 1.0)


def test_normalize_embedding_rejects_zero_vector() -> None:
    with pytest.raises(ValueError, match="zero-length"):
        normalize_embedding(np.zeros(3))


import numpy as np

from app.services.vector_store import VectorStore


def test_vector_store_add_search_remove_rebuild_and_restart(tmp_path) -> None:
    path = tmp_path / "vectors.npz"
    store = VectorStore(3, path)
    store.add(10, np.array([1, 0, 0], dtype=np.float32))
    store.add(20, np.array([0, 1, 0], dtype=np.float32))
    assert store.search(np.array([0.99, 0.01, 0]))[0][0] == 10
    store.remove(10)
    assert store.search(np.array([1, 0, 0]))[0][0] == 20
    store.rebuild([(30, np.array([0, 0, 1], dtype=np.float32))])
    store.save()
    restarted = VectorStore(3, path)
    restarted.load()
    assert restarted.search(np.array([0, 0, 1]))[0][0] == 30


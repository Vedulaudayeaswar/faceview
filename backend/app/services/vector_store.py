"""Persistent normalized-vector search store.

FAISS is used in production when installed. The NumPy backend has identical
inner-product semantics and keeps local development/tests usable when FAISS
cannot be installed on a platform.
"""

from pathlib import Path

import numpy as np

from app.models.embedder import normalize_embedding


class VectorStore:
    def __init__(self, dimension: int, path: str | Path) -> None:
        self.dimension = dimension
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.ids: list[int] = []
        self.vectors = np.empty((0, dimension), dtype=np.float32)
        try:
            import faiss  # type: ignore

            self._faiss = faiss
            self.backend = "faiss"
            self._index = faiss.IndexIDMap2(faiss.IndexFlatIP(dimension))
        except ImportError:
            self._faiss = None
            self.backend = "numpy-fallback"
            self._index = None

    def add(self, vector_id: int, vector: np.ndarray) -> None:
        values = normalize_embedding(vector)
        if values.size != self.dimension:
            raise ValueError(f"Expected embedding dimension {self.dimension}, got {values.size}")
        self.remove(vector_id)
        if self._faiss is not None:
            self._index.add_with_ids(values.reshape(1, -1), np.array([vector_id], dtype=np.int64))
        self.ids.append(vector_id)
        self.vectors = np.vstack([self.vectors, values])

    def search(self, vector: np.ndarray, k: int = 1) -> list[tuple[int, float]]:
        if not self.ids:
            return []
        query = normalize_embedding(vector)
        if self._faiss is not None:
            scores, ids = self._index.search(query.reshape(1, -1), min(k, len(self.ids)))
            return [(int(item_id), float(score)) for item_id, score in zip(ids[0], scores[0]) if item_id != -1]
        scores = self.vectors @ query
        order = np.argsort(-scores)[:k]
        return [(self.ids[int(index)], float(scores[int(index)])) for index in order]

    def remove(self, vector_id: int) -> None:
        if vector_id not in self.ids:
            return
        index = self.ids.index(vector_id)
        self.ids.pop(index)
        self.vectors = np.delete(self.vectors, index, axis=0)
        if self._faiss is not None:
            self._index.remove_ids(np.array([vector_id], dtype=np.int64))

    def rebuild(self, vectors: list[tuple[int, np.ndarray]]) -> None:
        self.ids = []
        self.vectors = np.empty((0, self.dimension), dtype=np.float32)
        if self._faiss is not None:
            self._index = self._faiss.IndexIDMap2(self._faiss.IndexFlatIP(self.dimension))
        for vector_id, vector in vectors:
            self.add(vector_id, vector)

    def save(self) -> None:
        np.savez_compressed(self.path, ids=np.asarray(self.ids, dtype=np.int64), vectors=self.vectors)

    def load(self) -> None:
        if not self.path.exists():
            return
        stored = np.load(self.path)
        ids = stored["ids"].tolist()
        vectors = stored["vectors"]
        self.rebuild(list(zip(ids, vectors)))


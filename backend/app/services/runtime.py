"""Lazy shared runtime for the fixed detector/embedder and persistent index."""

from pathlib import Path
from threading import RLock

import numpy as np
from sqlalchemy import select

from app.database.database import session_factory
from app.database.models import FaceImage, Identity
from app.models.detector import FaceDetector
from app.models.embedder import SFaceEmbedder
from app.services.recognition_service import RecognitionService
from app.services.vector_store import VectorStore


EMBEDDING_DIMENSION = 128
_lock = RLock()
_detector = None
_embedder = None
_vector_store = None
_recognizer = None


def get_runtime(session_factory_value):
    """Load the fixed model once and rebuild the active index from SQLite."""
    global _detector, _embedder, _vector_store, _recognizer
    with _lock:
        if _detector is None:
            _detector = FaceDetector()
            _embedder = SFaceEmbedder()
            _vector_store = VectorStore(EMBEDDING_DIMENSION, Path("data/indexes/faces.npz"))
            active = session_factory_value()
            try:
                rows = active.execute(
                    select(FaceImage, Identity.id)
                    .join(Identity, FaceImage.identity_id == Identity.id)
                    .where(FaceImage.status == "ACTIVE", Identity.status == "ACTIVE")
                ).all()
                vectors = [
                    (image.id, np.frombuffer(image.embedding, dtype=np.float32))
                    for image, _ in rows
                    if len(image.embedding) == EMBEDDING_DIMENSION * 4
                ]
                _vector_store.rebuild(vectors)
                _vector_store.save()
                _recognizer = RecognitionService(
                    _vector_store,
                    {image.id: identity_id for image, identity_id in rows if len(image.embedding) == EMBEDDING_DIMENSION * 4},
                )
            finally:
                active.close()
        return _detector, _embedder, _recognizer


def add_vector(vector_id: int, identity_id: int, embedding: np.ndarray) -> None:
    with _lock:
        if _vector_store is not None and _recognizer is not None:
            _vector_store.add(vector_id, embedding)
            _recognizer.register_vector_identity(vector_id, identity_id)
            _vector_store.save()


def remove_vectors(vector_ids: list[int]) -> None:
    with _lock:
        if _vector_store is not None and _recognizer is not None:
            for vector_id in vector_ids:
                _vector_store.remove(vector_id)
                _recognizer.remove_vector_identity(vector_id)
            _vector_store.save()

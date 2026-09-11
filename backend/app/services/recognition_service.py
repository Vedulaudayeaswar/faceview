"""Embedding-to-identity nearest-neighbor recognition."""

from dataclasses import dataclass

import numpy as np

from app.services.vector_store import VectorStore


@dataclass(frozen=True)
class RecognitionCandidate:
    vector_id: int | None
    identity_id: int | None
    similarity: float


@dataclass(frozen=True)
class RecognitionResult:
    result: str
    identity_id: int | None
    vector_id: int | None
    similarity: float


class RecognitionService:
    def __init__(self, vector_store: VectorStore, vector_to_identity: dict[int, int] | None = None) -> None:
        self.vector_store = vector_store
        self.vector_to_identity = vector_to_identity or {}

    def recognize(self, embedding: np.ndarray) -> RecognitionCandidate:
        matches = self.vector_store.search(embedding, k=1)
        if not matches:
            return RecognitionCandidate(None, None, 0.0)
        vector_id, similarity = matches[0]
        return RecognitionCandidate(vector_id, self.vector_to_identity.get(vector_id), similarity)

    def register_vector_identity(self, vector_id: int, identity_id: int) -> None:
        self.vector_to_identity[vector_id] = identity_id

    def remove_vector_identity(self, vector_id: int) -> None:
        self.vector_to_identity.pop(vector_id, None)

    def classify(self, embedding: np.ndarray, threshold: float) -> RecognitionResult:
        """Apply the configured threshold; low scores are always UNKNOWN."""
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Recognition threshold must be between 0 and 1")
        candidate = self.recognize(embedding)
        accepted = candidate.identity_id is not None and candidate.similarity >= threshold
        return RecognitionResult(
            result="KNOWN" if accepted else "UNKNOWN",
            identity_id=candidate.identity_id if accepted else None,
            vector_id=candidate.vector_id if accepted else None,
            similarity=candidate.similarity,
        )

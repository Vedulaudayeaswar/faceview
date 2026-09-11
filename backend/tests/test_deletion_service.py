import numpy as np
from sqlalchemy import create_engine

from app.database.database import Base, session_factory
from app.database.repositories import IdentityRepository
from app.services.deletion_service import DeletionService
from app.services.recognition_service import RecognitionService
from app.services.vector_store import VectorStore


def test_deleting_identity_removes_vector_and_mapping(tmp_path) -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session = session_factory(engine)()
    repository = IdentityRepository(session)
    identity = repository.create("A-1", "A")
    image = repository.add_face_image(identity.id, "faces/a.png", b"embedding")
    image.embedding_id = 5
    session.commit()
    store = VectorStore(2, tmp_path / "vectors.npz")
    store.add(5, np.array([1, 0]))
    recognizer = RecognitionService(store, {5: identity.id})
    assert DeletionService(repository, recognizer).delete_identity(identity.id)
    assert store.search(np.array([1, 0])) == []
    assert recognizer.vector_to_identity == {}


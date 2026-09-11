"""Identity/vector deletion orchestration."""

from app.database.repositories import IdentityRepository
from app.services.recognition_service import RecognitionService


class DeletionService:
    def __init__(self, repository: IdentityRepository, recognizer: RecognitionService) -> None:
        self.repository = repository
        self.recognizer = recognizer

    def delete_identity(self, identity_id: int) -> bool:
        identity = self.repository.get(identity_id)
        if identity is None:
            return False
        vector_ids = [image.embedding_id for image in identity.face_images if image.embedding_id is not None]
        deleted = self.repository.delete(identity_id)
        for vector_id in vector_ids:
            self.recognizer.vector_store.remove(vector_id)
            self.recognizer.remove_vector_identity(vector_id)
        return deleted

    def delete_identities(self, identity_ids: list[int]) -> int:
        return sum(1 for identity_id in identity_ids if self.delete_identity(identity_id))


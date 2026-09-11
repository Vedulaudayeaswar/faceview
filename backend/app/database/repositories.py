"""Small repository operations used by application services and tests."""

from datetime import datetime, timezone

from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.database.models import FaceImage, Identity


class DuplicateIdentityError(ValueError):
    pass


class IdentityRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create(self, identity_code: str, name: str, metadata_json: str | None = None) -> Identity:
        existing = self.session.scalar(select(Identity).where(Identity.identity_code == identity_code))
        if existing is not None:
            if existing.status == "DELETED":
                # Identity codes remain unique for audit/history, but a deleted
                # identity may be enrolled again without manual DB editing.
                existing.name = name
                existing.metadata_json = metadata_json
                existing.status = "ACTIVE"
                existing.updated_at = datetime.now(timezone.utc)
                self.session.commit()
                self.session.refresh(existing)
                return existing
            raise DuplicateIdentityError(f"Identity code already exists: {identity_code}")
        identity = Identity(identity_code=identity_code, name=name, metadata_json=metadata_json)
        self.session.add(identity)
        try:
            self.session.commit()
        except IntegrityError as exc:
            self.session.rollback()
            raise DuplicateIdentityError(f"Identity code already exists: {identity_code}") from exc
        self.session.refresh(identity)
        return identity

    def list_active(self) -> list[Identity]:
        return list(self.session.scalars(select(Identity).where(Identity.status == "ACTIVE").order_by(Identity.id)))

    def get(self, identity_id: int) -> Identity | None:
        return self.session.get(Identity, identity_id)

    def delete(self, identity_id: int) -> bool:
        identity = self.get(identity_id)
        if identity is None:
            return False
        identity.status = "DELETED"
        for image in identity.face_images:
            image.status = "DELETED"
            image.deleted_at = datetime.now(timezone.utc)
        self.session.commit()
        return True

    def add_face_image(self, identity_id: int, image_path: str, embedding: bytes) -> FaceImage:
        image = FaceImage(identity_id=identity_id, image_path=image_path, embedding=embedding)
        self.session.add(image)
        self.session.commit()
        self.session.refresh(image)
        return image

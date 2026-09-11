"""Persistence helpers for recognition and audit events."""

from sqlalchemy.orm import Session

from app.database.models import AuditLog, RecognitionEvent


class EventService:
    def __init__(self, session: Session) -> None:
        self.session = session

    def record_recognition(
        self, camera_id: int | None, identity_id: int | None, similarity: float, result: str
    ) -> RecognitionEvent:
        if result not in {"KNOWN", "UNKNOWN"}:
            raise ValueError("Recognition result must be KNOWN or UNKNOWN")
        event = RecognitionEvent(
            camera_id=camera_id, identity_id=identity_id, similarity_score=float(similarity), result=result
        )
        self.session.add(event)
        self.session.commit()
        self.session.refresh(event)
        return event

    def audit(self, operation: str, description: str, identity_id: int | None = None) -> AuditLog:
        log = AuditLog(identity_id=identity_id, operation=operation, description=description)
        self.session.add(log)
        self.session.commit()
        self.session.refresh(log)
        return log


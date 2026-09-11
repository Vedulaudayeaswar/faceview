import pytest
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.database.database import Base
from app.database.models import AuditLog, RecognitionEvent
from app.services.event_service import EventService


def test_recognition_and_audit_events_are_persisted() -> None:
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    with Session(engine) as session:
        service = EventService(session)
        event = service.record_recognition(None, None, 0.42, "UNKNOWN")
        log = service.audit("ENROLL", "Created reference embedding")
        assert event.id and log.id
        assert len(session.scalars(select(RecognitionEvent)).all()) == 1
        assert len(session.scalars(select(AuditLog)).all()) == 1
        with pytest.raises(ValueError):
            service.record_recognition(None, None, 0.1, "MAYBE")


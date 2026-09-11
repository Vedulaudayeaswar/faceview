from sqlalchemy import create_engine

from app.database.database import Base, session_factory
from app.database.repositories import DuplicateIdentityError, IdentityRepository


def test_identity_create_read_delete_and_duplicate() -> None:
    engine = create_engine("sqlite:///:memory:", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    session = session_factory(engine)()
    repository = IdentityRepository(session)

    created = repository.create("EMP-001", "Uday")
    assert repository.get(created.id).name == "Uday"
    try:
        repository.create("EMP-001", "Other")
    except DuplicateIdentityError:
        pass
    else:
        raise AssertionError("Duplicate identity should fail")
    assert repository.delete(created.id) is True
    assert repository.get(created.id).status == "DELETED"
    session.close()


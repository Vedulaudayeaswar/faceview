"""SQLAlchemy engine and session setup."""

from collections.abc import Generator
from pathlib import Path

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker


class Base(DeclarativeBase):
    pass


def create_database(url: str = "sqlite:///./data/face_recognition.db"):
    if url.startswith("sqlite:///"):
        database_path = Path(url.removeprefix("sqlite:///"))
        database_path.parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(url, connect_args={"check_same_thread": False} if url.startswith("sqlite") else {})
    Base.metadata.create_all(engine)
    # Small backwards-compatible SQLite migration for databases created before
    # embeddings carried their model identity.
    if url.startswith("sqlite"):
        columns = {column["name"] for column in inspect(engine).get_columns("face_images")}
        if "embedding_model" not in columns:
            with engine.begin() as connection:
                connection.execute(text("ALTER TABLE face_images ADD COLUMN embedding_model VARCHAR(100) DEFAULT 'unknown'"))
    return engine


def session_factory(engine) -> sessionmaker[Session]:
    return sessionmaker(bind=engine, autoflush=False, expire_on_commit=False)


def get_session(factory: sessionmaker[Session]) -> Generator[Session, None, None]:
    session = factory()
    try:
        yield session
    finally:
        session.close()

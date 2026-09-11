"""Relational persistence models."""

from datetime import datetime, timezone

from sqlalchemy import DateTime, Float, ForeignKey, Integer, LargeBinary, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database.database import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Identity(Base):
    __tablename__ = "identities"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    identity_code: Mapped[str] = mapped_column(String(100), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(200))
    metadata_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="ACTIVE")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow, onupdate=utcnow)
    face_images: Mapped[list["FaceImage"]] = relationship(back_populates="identity", cascade="all, delete-orphan")


class FaceImage(Base):
    __tablename__ = "face_images"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    identity_id: Mapped[int] = mapped_column(ForeignKey("identities.id"), index=True)
    image_path: Mapped[str] = mapped_column(String(500))
    embedding_id: Mapped[int | None] = mapped_column(Integer, nullable=True, index=True)
    embedding: Mapped[bytes] = mapped_column(LargeBinary)
    embedding_model: Mapped[str] = mapped_column(String(100), default="unknown")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="ACTIVE")
    identity: Mapped[Identity] = relationship(back_populates="face_images")


class Camera(Base):
    __tablename__ = "cameras"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    camera_name: Mapped[str] = mapped_column(String(200))
    camera_type: Mapped[str] = mapped_column(String(20))
    device_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    rtsp_url: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    status: Mapped[str] = mapped_column(String(20), default="DISABLED")
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class RecognitionEvent(Base):
    __tablename__ = "recognition_events"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    identity_id: Mapped[int | None] = mapped_column(ForeignKey("identities.id"), nullable=True)
    camera_id: Mapped[int | None] = mapped_column(ForeignKey("cameras.id"), nullable=True)
    similarity_score: Mapped[float] = mapped_column(Float)
    result: Mapped[str] = mapped_column(String(20))
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)


class AuditLog(Base):
    __tablename__ = "audit_logs"
    id: Mapped[int] = mapped_column(Integer, primary_key=True)
    identity_id: Mapped[int | None] = mapped_column(ForeignKey("identities.id"), nullable=True)
    operation: Mapped[str] = mapped_column(String(50))
    description: Mapped[str] = mapped_column(Text)
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

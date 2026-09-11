"""Initial REST API for identities, cameras, settings, statistics, and events."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import func, select

from app.database.database import create_database, session_factory
from app.database.models import Camera, Identity, RecognitionEvent
from app.database.repositories import DuplicateIdentityError, IdentityRepository

router = APIRouter(prefix="/api")
_engine = create_database("sqlite:///./data/face_recognition.db")
_sessions = session_factory(_engine)
_settings = {"recognition_threshold": 0.60, "frame_skip": 2, "detection_confidence": 0.50}


class IdentityCreate(BaseModel):
    identity_code: str = Field(min_length=1, max_length=100)
    name: str = Field(min_length=1, max_length=200)
    metadata: dict | None = None


class CameraCreate(BaseModel):
    camera_name: str = Field(min_length=1, max_length=200)
    camera_type: str = Field(pattern="^(USB|RTSP)$")
    device_index: int | None = None
    rtsp_url: str | None = None
    status: str = "DISABLED"


@router.get("/identities")
def list_identities() -> list[dict]:
    with _sessions() as session:
        return [
            {"id": item.id, "identity_code": item.identity_code, "name": item.name, "status": item.status}
            for item in IdentityRepository(session).list_active()
        ]


@router.post("/identities", status_code=201)
def create_identity(payload: IdentityCreate) -> dict:
    with _sessions() as session:
        try:
            item = IdentityRepository(session).create(payload.identity_code, payload.name, str(payload.metadata or {}))
        except DuplicateIdentityError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        return {"id": item.id, "identity_code": item.identity_code, "name": item.name, "status": item.status}


@router.delete("/identities/{identity_id}")
def delete_identity(identity_id: int) -> dict[str, bool]:
    with _sessions() as session:
        if not IdentityRepository(session).delete(identity_id):
            raise HTTPException(status_code=404, detail="Identity not found")
        return {"deleted": True}


@router.get("/cameras")
def list_cameras() -> list[dict]:
    with _sessions() as session:
        cameras = session.scalars(select(Camera).order_by(Camera.id)).all()
        return [{"id": item.id, "camera_name": item.camera_name, "camera_type": item.camera_type, "status": item.status} for item in cameras]


@router.post("/cameras", status_code=201)
def create_camera(payload: CameraCreate) -> dict:
    with _sessions() as session:
        camera = Camera(**payload.model_dump())
        session.add(camera)
        session.commit()
        session.refresh(camera)
        return {"id": camera.id, "camera_name": camera.camera_name, "camera_type": camera.camera_type, "status": camera.status}


@router.get("/settings")
def get_settings() -> dict:
    return _settings.copy()


@router.put("/settings")
def update_settings(payload: dict[str, float | int]) -> dict:
    for key, value in payload.items():
        if key not in _settings:
            raise HTTPException(status_code=400, detail=f"Unsupported setting: {key}")
        _settings[key] = value
    return _settings.copy()


@router.get("/statistics")
def statistics() -> dict[str, int]:
    with _sessions() as session:
        return {
            "registered_identities": session.scalar(select(func.count()).select_from(Identity).where(Identity.status == "ACTIVE")) or 0,
            "active_cameras": session.scalar(select(func.count()).select_from(Camera).where(Camera.status == "ACTIVE")) or 0,
            "recognition_events": session.scalar(select(func.count()).select_from(RecognitionEvent)) or 0,
        }


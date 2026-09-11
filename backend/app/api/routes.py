"""REST API for identity management, uploads, cameras, and recognition."""

import json
import tempfile
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from pydantic import BaseModel, Field
from sqlalchemy import func, select

from app.database.database import create_database, session_factory
from app.database.models import Camera, FaceImage, Identity, RecognitionEvent
from app.database.repositories import DuplicateIdentityError, IdentityRepository
from app.services.enrollment_service import EnrollmentError, EnrollmentService
from app.services.event_service import EventService
from app.models.embedder import normalize_embedding
from app.services.runtime import add_vector, get_runtime, remove_vectors

router = APIRouter(prefix="/api")
_engine = create_database("sqlite:///./data/face_recognition.db")
_sessions = session_factory(_engine)
_settings = {"recognition_threshold": 0.60, "frame_skip": 2, "detection_confidence": 0.50}
_MAX_IMAGE_BYTES = 10 * 1024 * 1024
_MAX_VIDEO_BYTES = 250 * 1024 * 1024


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
        identity = IdentityRepository(session).get(identity_id)
        if identity is None:
            raise HTTPException(status_code=404, detail="Identity not found")
        vector_ids = [image.id for image in identity.face_images if image.status == "ACTIVE"]
        if not IdentityRepository(session).delete(identity_id):
            raise HTTPException(status_code=404, detail="Identity not found")
        remove_vectors(vector_ids)
        return {"deleted": True}


async def _read_upload(file: UploadFile, maximum: int) -> bytes:
    contents = await file.read()
    if not contents:
        raise HTTPException(status_code=400, detail="Uploaded file is empty")
    if len(contents) > maximum:
        raise HTTPException(status_code=413, detail="Uploaded file is too large")
    return contents


def _image_from_bytes(contents: bytes) -> np.ndarray:
    image = cv2.imdecode(np.frombuffer(contents, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.size == 0:
        raise HTTPException(status_code=400, detail="Invalid image file")
    return image


def _identity_names() -> dict[int, str]:
    with _sessions() as session:
        return {item.id: item.name for item in session.scalars(select(Identity).where(Identity.status == "ACTIVE"))}


def _recognize_frame(frame: np.ndarray, threshold: float, camera_id: int | None = None) -> list[dict]:
    detector, embedder, recognizer = get_runtime(_sessions)
    names = _identity_names()
    results = []
    for face in detector.detect(frame):
        try:
            embedding = normalize_embedding(face.embedding) if face.embedding is not None else embedder.embed(frame, face.bbox).vector
        except ValueError:
            continue
        outcome = recognizer.classify(embedding, threshold)
        result = {
            "bbox": list(face.bbox),
            "result": outcome.result,
            "identity_id": outcome.identity_id,
            "name": names.get(outcome.identity_id) if outcome.identity_id else None,
            "similarity": round(outcome.similarity, 6),
        }
        results.append(result)
        with _sessions() as session:
            EventService(session).record_recognition(camera_id, outcome.identity_id, outcome.similarity, outcome.result)
    return results


@router.post("/enroll/image", status_code=201)
async def enroll_image(
    identity_code: str = Query(min_length=1, max_length=100),
    name: str = Query(min_length=1, max_length=200),
    metadata: str | None = None,
    file: UploadFile = File(...),
) -> dict:
    contents = await _read_upload(file, _MAX_IMAGE_BYTES)
    detector, embedder, _ = get_runtime(_sessions)
    try:
        prepared = EnrollmentService(detector, embedder).prepare_single(contents)
    except EnrollmentError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    image_dir = Path("data/faces")
    image_dir.mkdir(parents=True, exist_ok=True)
    stored_path = image_dir / f"{uuid4().hex}.jpg"
    stored_path.write_bytes(contents)
    with _sessions() as session:
        try:
            identity = IdentityRepository(session).create(identity_code, name, metadata or "{}")
        except DuplicateIdentityError as exc:
            stored_path.unlink(missing_ok=True)
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        image_record = IdentityRepository(session).add_face_image(
            identity.id, str(stored_path), prepared.embedding.astype(np.float32).tobytes()
        )
        image_record.embedding_id = image_record.id
        session.commit()
        EventService(session).audit("ENROLL", f"Enrolled one reference image for {identity.identity_code}", identity.id)
        result = {"id": identity.id, "identity_code": identity.identity_code, "name": identity.name, "image_id": image_record.id}
    add_vector(image_record.id, identity.id, prepared.embedding)
    return result


@router.post("/recognize/image")
async def recognize_image(file: UploadFile = File(...), camera_id: int | None = None) -> dict:
    contents = await _read_upload(file, _MAX_IMAGE_BYTES)
    frame = _image_from_bytes(contents)
    threshold = float(_settings["recognition_threshold"])
    faces = _recognize_frame(frame, threshold, camera_id)
    return {"faces": faces, "face_count": len(faces), "threshold": threshold}


@router.post("/recognize/video")
async def recognize_video(file: UploadFile = File(...), camera_id: int | None = None) -> dict:
    contents = await _read_upload(file, _MAX_VIDEO_BYTES)
    suffix = Path(file.filename or "upload.mp4").suffix.lower() or ".mp4"
    temporary_path = Path(tempfile.gettempdir()) / f"faceview-{uuid4().hex}{suffix}"
    temporary_path.write_bytes(contents)
    capture = cv2.VideoCapture(str(temporary_path))
    if not capture.isOpened():
        temporary_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail="Unable to open uploaded video")
    threshold = float(_settings["recognition_threshold"])
    frame_skip = int(_settings["frame_skip"])
    frame_number = 0
    processed_frames = 0
    results = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_number % (frame_skip + 1) == 0:
                faces = _recognize_frame(frame, threshold, camera_id)
                results.extend(faces)
                processed_frames += 1
            frame_number += 1
    finally:
        capture.release()
        temporary_path.unlink(missing_ok=True)
    return {
        "frames_read": frame_number,
        "frames_processed": processed_frames,
        "faces_detected": len(results),
        "known_faces": sum(item["result"] == "KNOWN" for item in results),
        "unknown_faces": sum(item["result"] == "UNKNOWN" for item in results),
        "results": results,
        "threshold": threshold,
        "frame_skip": frame_skip,
    }


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

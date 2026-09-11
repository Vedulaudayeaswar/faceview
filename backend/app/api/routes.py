"""REST API for identity management, uploads, cameras, and recognition."""

import json
import tempfile
from pathlib import Path
from uuid import uuid4

import cv2
import numpy as np
from fastapi import APIRouter, File, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from sqlalchemy import func, select

from app.database.database import create_database, session_factory
from app.database.models import Camera, FaceImage, Identity, RecognitionEvent
from app.database.repositories import DuplicateIdentityError, IdentityRepository
from app.services.enrollment_service import EnrollmentError, EnrollmentService
from app.services.event_service import EventService
from app.models.embedder import embed_detected, normalize_embedding
from app.services.runtime import add_vector, get_runtime, remove_vectors

router = APIRouter(prefix="/api")
_engine = create_database("sqlite:///./data/face_recognition.db")
_sessions = session_factory(_engine)
_settings = {"recognition_threshold": 0.60, "frame_skip": 2, "detection_confidence": 0.50}
_MAX_IMAGE_BYTES = 10 * 1024 * 1024
_MAX_VIDEO_BYTES = 250 * 1024 * 1024
_VIDEO_RESULTS_DIR = Path("data/video-results")


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


def _recognize_frame(
    frame: np.ndarray,
    threshold: float,
    camera_id: int | None = None,
    annotate: bool = False,
) -> tuple[list[dict], np.ndarray | None]:
    detector, embedder, recognizer = get_runtime(_sessions)
    names = _identity_names()
    results = []
    annotated = frame.copy() if annotate else None
    for face in detector.detect(frame):
        try:
            embedding = normalize_embedding(face.embedding) if face.embedding is not None else embed_detected(embedder, frame, face).vector
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
        if annotated is not None:
            x, y, w, h = face.bbox
            color = (70, 220, 120) if outcome.result == "KNOWN" else (70, 70, 240)
            label = f"{result['name'] or 'UNKNOWN'} | ID: {result['identity_id'] or '-'} | {outcome.similarity:.1%}"
            cv2.rectangle(annotated, (x, y), (x + w, y + h), color, 2)
            cv2.rectangle(annotated, (x, max(0, y - 28)), (min(frame.shape[1], x + 420), y), color, -1)
            cv2.putText(annotated, label, (x + 5, max(19, y - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (15, 15, 15), 2)
        with _sessions() as session:
            EventService(session).record_recognition(camera_id, outcome.identity_id, outcome.similarity, outcome.result)
    return results, annotated


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
        repository = IdentityRepository(session)
        existing = repository.get_by_code(identity_code)
        if existing is not None and existing.status == "ACTIVE":
            # Additional reference photos improve pose/lighting robustness.
            # They share the identity and are independently searchable vectors.
            identity = existing
        else:
            try:
                identity = repository.create(identity_code, name, metadata or "{}")
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
    faces, _ = _recognize_frame(frame, threshold, camera_id)
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
    _VIDEO_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # VP8/WebM is broadly playable in browsers; mp4v often produces a blank
    # HTML5 video even though the file was written successfully.
    result_name = f"{uuid4().hex}.webm"
    result_path = _VIDEO_RESULTS_DIR / result_name
    fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(str(result_path), cv2.VideoWriter_fourcc(*"VP80"), fps, (width, height))
    if not writer.isOpened():
        capture.release()
        temporary_path.unlink(missing_ok=True)
        raise HTTPException(status_code=500, detail="Unable to create annotated result video")
    frame_number = 0
    processed_frames = 0
    results = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame_number % (frame_skip + 1) == 0:
                faces, annotated = _recognize_frame(frame, threshold, camera_id, annotate=True)
                results.extend(faces)
                processed_frames += 1
                writer.write(annotated if annotated is not None else frame)
            else:
                writer.write(frame)
            frame_number += 1
    finally:
        capture.release()
        writer.release()
        temporary_path.unlink(missing_ok=True)
    return {
        "frames_read": frame_number,
        "frames_processed": processed_frames,
        "faces_detected": len(results),
        "known_faces": sum(item["result"] == "KNOWN" for item in results),
        "unknown_faces": sum(item["result"] == "UNKNOWN" for item in results),
        "results": results,
        "annotated_video_url": f"/api/results/video/{result_name}",
        "threshold": threshold,
        "frame_skip": frame_skip,
    }


@router.get("/results/video/{filename}")
def get_result_video(filename: str) -> FileResponse:
    if Path(filename).name != filename or Path(filename).suffix.lower() not in {".mp4", ".webm"}:
        raise HTTPException(status_code=404, detail="Video result not found")
    path = _VIDEO_RESULTS_DIR / filename
    if not path.exists():
        raise HTTPException(status_code=404, detail="Video result not found")
    media_type = "video/webm" if path.suffix.lower() == ".webm" else "video/mp4"
    return FileResponse(path, media_type=media_type, filename=f"faceview-recognition{path.suffix.lower()}")


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

from fastapi.testclient import TestClient
from uuid import uuid4

from app.main import app


def test_identity_api_create_list_delete() -> None:
    client = TestClient(app)
    code = f"TEST-API-{uuid4().hex[:8]}"
    response = client.post("/api/identities", json={"identity_code": code, "name": "API Test"})
    assert response.status_code == 201
    identity_id = response.json()["id"]
    assert any(item["id"] == identity_id for item in client.get("/api/identities").json())
    assert client.delete(f"/api/identities/{identity_id}").status_code == 200


def test_settings_api() -> None:
    client = TestClient(app)
    response = client.put("/api/settings", json={"recognition_threshold": 0.72})
    assert response.status_code == 200
    assert response.json()["recognition_threshold"] == 0.72


def test_upload_recognition_rejects_invalid_image_without_enrolling() -> None:
    client = TestClient(app)
    response = client.post("/api/recognize/image", files={"file": ("bad.jpg", b"not-an-image", "image/jpeg")})
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid image file"


def test_upload_routes_are_documented() -> None:
    paths = app.openapi()["paths"]
    assert "/api/enroll/image" in paths
    assert "/api/recognize/image" in paths
    assert "/api/recognize/video" in paths
    assert "/api/webcam/stream" in paths

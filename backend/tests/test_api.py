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

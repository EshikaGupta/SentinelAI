from fastapi.testclient import TestClient

from app.api import app

client = TestClient(app)


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ask_rejects_empty_question():
    response = client.post("/ask", json={"question": ""})

    assert response.status_code == 422


def test_upload_rejects_unsupported_file():
    response = client.post(
        "/upload",
        files={"file": ("notes.exe", b"not a supported document")},
    )

    assert response.status_code == 400
from fastapi.testclient import TestClient

from app.main import app


def test_health_and_question_endpoints():
    with TestClient(app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json()["indexed_documents"] > 0

        response = client.post("/ask", json={"question": "software engineer in IT", "top_k": 2})
        assert response.status_code == 200
        assert response.json()["sources"]


def test_question_validation():
    with TestClient(app) as client:
        response = client.post("/ask", json={"question": "  "})
        assert response.status_code == 422


from fastapi.testclient import TestClient

from API.server import app


def test_health():
    with TestClient(app) as client:
        response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"healthy": "true"}


def test_welcome():
    with TestClient(app) as client:
        response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {"message": "FastAPI server is up."}

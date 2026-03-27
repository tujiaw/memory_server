import os

from fastapi.testclient import TestClient
import pytest


os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")
os.environ.setdefault("OPENAI_BASE_URL", "https://example.com/v1")
os.environ.setdefault("SECRET_KEY", "test-secret-key")
os.environ.setdefault("DATABASE_URL", "postgresql://memory:memory@localhost:5433/memory_server")
os.environ.setdefault(
    "SERVICE_CLIENTS_JSON",
    '{"svc-agent":{"secret":"top-secret","namespaces":["team-a"]}}',
)

import app.database.postgres as pgmod
import main


@pytest.fixture
def client(monkeypatch):
    async def fake_connect():
        return None

    async def fake_disconnect():
        return None

    monkeypatch.setattr(pgmod.postgres_db, "connect", fake_connect)
    monkeypatch.setattr(pgmod.postgres_db, "disconnect", fake_disconnect)
    return TestClient(main.app)


def test_root_exposes_internal_platform_endpoints(client):
    response = client.get("/")

    assert response.status_code == 200
    payload = response.json()
    assert payload["endpoints"]["token"] == "/api/v1/auth/token"
    assert payload["endpoints"]["subjects"] == "/api/v1/subjects"


def test_health_returns_503_when_dependency_is_unhealthy(client, monkeypatch):
    async def fake_collect_health_status():
        return {
            "status": "degraded",
            "services": {
                "postgresql": "healthy",
                "qdrant": "unhealthy",
                "openai_config": "healthy",
            },
        }

    monkeypatch.setattr(main, "collect_health_status", fake_collect_health_status)

    response = client.get("/health")

    assert response.status_code == 503
    assert response.json()["status"] == "degraded"


def test_http_exception_uses_unified_error_response(client):
    response = client.get("/api/v1/subjects/subject-1/context")

    assert response.status_code == 401
    payload = response.json()
    assert payload["success"] is False
    assert payload["code"] == "HTTP_401"

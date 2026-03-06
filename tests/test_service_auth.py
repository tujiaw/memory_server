import os
from datetime import timedelta

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest


os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")
os.environ.setdefault("OPENAI_API_BASE", "https://example.com/v1")
os.environ.setdefault("SECRET_KEY", "test-secret-key")
os.environ.setdefault(
    "SERVICE_CLIENTS_JSON",
    '{"svc-agent":{"secret":"top-secret","scopes":["memory:read","memory:write","context:read","context:write"],"namespaces":["team-a"]}}',
)

from app.api.auth_routes import router as auth_router
from app.core.deps import authorize_namespace
from app.services.auth_service import auth_service


def build_auth_app() -> TestClient:
    app = FastAPI()
    app.include_router(auth_router, prefix="/api/v1")
    return TestClient(app)


def test_validate_service_credentials_accepts_known_client():
    client = auth_service.validate_service_credentials("svc-agent", "top-secret")

    assert client is not None
    assert client.client_id == "svc-agent"
    assert client.namespaces == ["team-a"]
    assert "memory:write" in client.scopes


def test_create_service_token_returns_auth_context():
    token = auth_service.create_service_token(
        service_id="svc-agent",
        scopes=["memory:read", "memory:write"],
        namespaces=["team-a"],
        expires_delta=timedelta(minutes=5),
    )

    auth_context = auth_service.verify_token(token)

    assert auth_context is not None
    assert auth_context.service_id == "svc-agent"
    assert auth_context.scopes == ["memory:read", "memory:write"]
    assert auth_context.namespaces == ["team-a"]


def test_authorize_namespace_rejects_unauthorized_scope():
    token = auth_service.create_service_token(
        service_id="svc-agent",
        scopes=["memory:read"],
        namespaces=["team-a"],
    )
    auth_context = auth_service.verify_token(token)

    with pytest.raises(Exception):
        authorize_namespace(auth_context, namespace="team-a", required_scope="memory:write")


def test_issue_service_token_from_api():
    client = build_auth_app()

    response = client.post(
        "/api/v1/auth/token",
        json={"client_id": "svc-agent", "client_secret": "top-secret"},
    )

    assert response.status_code == 200

    payload = response.json()
    assert payload["service_id"] == "svc-agent"
    assert payload["token_type"] == "bearer"
    assert "memory:read" in payload["scopes"]
    assert payload["namespaces"] == ["team-a"]

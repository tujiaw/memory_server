import os
from datetime import timedelta, timezone

from fastapi import FastAPI, HTTPException
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
from app.api.auth_routes import router as auth_router
from app.services.auth_service import auth_service
from tests.fake_postgres import FakePostgresDB


def build_auth_app() -> TestClient:
    app = FastAPI()
    app.include_router(auth_router, prefix="/api/v1")
    return TestClient(app)


def test_validate_service_credentials_accepts_known_client():
    client = auth_service.validate_service_credentials("svc-agent", "top-secret")

    assert client is not None
    assert client.client_id == "svc-agent"
    assert client.namespaces == ["team-a"]


@pytest.mark.asyncio
async def test_create_service_token_returns_auth_context():
    token = auth_service.create_service_token(
        service_id="svc-agent",
        namespaces=["team-a"],
        expires_delta=timedelta(minutes=5),
    )

    auth_context = await auth_service.verify_token(token)

    assert auth_context is not None
    assert auth_context.service_id == "svc-agent"
    assert auth_context.namespaces == ["team-a"]


@pytest.mark.asyncio
async def test_verify_token_checks_postgresql_persistence(monkeypatch):
    fake_pg = FakePostgresDB()
    monkeypatch.setattr(pgmod, "postgres_db", fake_pg)

    token = await auth_service.issue_service_token(
        service_id="svc-agent",
        namespaces=["team-a"],
        expires_delta=timedelta(minutes=5),
    )

    auth_context = await auth_service.verify_token(token)

    assert auth_context is not None
    assert auth_context.service_id == "svc-agent"

    th = auth_service._hash_token(token)
    fake_pg.tokens.pop(th, None)

    auth_context_after_delete = await auth_service.verify_token(token)
    assert auth_context_after_delete is None


@pytest.mark.asyncio
async def test_verify_token_rejects_expired_persisted_token(monkeypatch):
    fake_pg = FakePostgresDB()
    monkeypatch.setattr(pgmod, "postgres_db", fake_pg)

    token = await auth_service.issue_service_token(
        service_id="svc-agent",
        namespaces=["team-a"],
        expires_delta=timedelta(minutes=-1),
    )

    auth_context = await auth_service.verify_token(token)
    assert auth_context is None


@pytest.mark.asyncio
async def test_verify_token_accepts_naive_postgresql_expiry_datetime(monkeypatch):
    fake_pg = FakePostgresDB()
    monkeypatch.setattr(pgmod, "postgres_db", fake_pg)

    token = await auth_service.issue_service_token(
        service_id="svc-agent",
        namespaces=["team-a"],
        expires_delta=timedelta(minutes=5),
    )
    stored_token = fake_pg.tokens[auth_service._hash_token(token)]
    stored_token["expires_at"] = stored_token["expires_at"].astimezone(timezone.utc).replace(tzinfo=None)

    auth_context = await auth_service.verify_token(token)

    assert auth_context is not None
    assert auth_context.service_id == "svc-agent"


@pytest.mark.asyncio
async def test_authorize_namespace_rejects_unauthorized_namespace():
    from app.core.deps import authorize_namespace

    token = auth_service.create_service_token(
        service_id="svc-agent",
        namespaces=["team-b"],
    )
    auth_context = await auth_service.verify_token(token)

    with pytest.raises(HTTPException):
        authorize_namespace(auth_context, namespace="team-a")


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
    assert payload["namespaces"] == ["team-a"]

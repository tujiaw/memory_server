import os
from copy import deepcopy

from fastapi.testclient import TestClient
import pytest


os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")
os.environ.setdefault("OPENAI_BASE_URL", "https://example.com/v1")
os.environ.setdefault("SECRET_KEY", "test-secret-key")
os.environ.setdefault("ADMIN_API_TOKEN", "test-admin-token")
os.environ.setdefault(
    "SERVICE_CLIENTS_JSON",
    '{"svc-agent":{"secret":"top-secret","namespaces":["team-a"]}}',
)

import main
from app.core.config import settings


class FakeCursor:
    def __init__(self, documents):
        self._documents = [deepcopy(document) for document in documents]

    async def to_list(self, length=None):
        if length is None:
            return list(self._documents)
        return list(self._documents[:length])


class FakeDeleteResult:
    def __init__(self, deleted_count):
        self.deleted_count = deleted_count


class FakeServiceClientCollection:
    def __init__(self):
        self.documents = {}

    async def find_one(self, query):
        if "_id" in query:
            document = self.documents.get(query["_id"])
            return deepcopy(document) if document is not None else None

        if "client_id" in query:
            for document in self.documents.values():
                if document.get("client_id") == query["client_id"]:
                    return deepcopy(document)
        return None

    async def insert_one(self, document):
        self.documents[document["_id"]] = deepcopy(document)

    def find(self, query=None):
        query = query or {}
        matched = []
        for document in self.documents.values():
            if all(document.get(key) == value for key, value in query.items()):
                matched.append(deepcopy(document))
        return FakeCursor(matched)

    async def update_one(self, query, update):
        document = await self.find_one(query)
        if document is None:
            return FakeDeleteResult(0)
        if "$set" in update:
            for key, value in update["$set"].items():
                document[key] = value
            self.documents[document["_id"]] = deepcopy(document)
        return FakeDeleteResult(1)

    async def delete_one(self, query):
        document = await self.find_one(query)
        if document is None:
            return FakeDeleteResult(0)
        self.documents.pop(document["_id"], None)
        return FakeDeleteResult(1)


@pytest.fixture
def admin_client(monkeypatch):
    fake_collection = FakeServiceClientCollection()

    async def fake_connect():
        return None

    async def fake_disconnect():
        return None

    monkeypatch.setattr(main.mongodb, "connect", fake_connect)
    monkeypatch.setattr(main.mongodb, "disconnect", fake_disconnect)
    monkeypatch.setattr(main.mongodb, "get_collection", lambda _: fake_collection)
    monkeypatch.setattr(settings, "ADMIN_API_TOKEN", "test-admin-token")

    return TestClient(main.app)


def test_admin_client_lifecycle(admin_client):
    create_response = admin_client.post(
        "/api/v1/admin/clients",
        headers={"X-Admin-Token": "test-admin-token"},
        json={
            "client_id": "svc-analytics",
            "namespaces": ["team-b"],
            "description": "analytics client",
        },
    )

    assert create_response.status_code == 201
    create_payload = create_response.json()
    assert create_payload["client_id"] == "svc-analytics"
    assert create_payload["client_secret"]
    assert create_payload["namespaces"] == ["team-b"]

    list_response = admin_client.get(
        "/api/v1/admin/clients",
        headers={"X-Admin-Token": "test-admin-token"},
    )

    assert list_response.status_code == 200
    listed_client = list_response.json()["clients"][0]
    assert listed_client["client_id"] == "svc-analytics"
    assert "client_secret" not in listed_client
    assert listed_client["description"] == "analytics client"

    update_response = admin_client.patch(
        "/api/v1/admin/clients/svc-analytics",
        headers={"X-Admin-Token": "test-admin-token"},
        json={"description": "updated analytics"},
    )

    assert update_response.status_code == 200
    update_payload = update_response.json()
    assert update_payload["description"] == "updated analytics"

    delete_response = admin_client.delete(
        "/api/v1/admin/clients/svc-analytics",
        headers={"X-Admin-Token": "test-admin-token"},
    )

    assert delete_response.status_code == 204


def test_reset_client_secret_returns_new_secret_and_invalidates_old(admin_client):
    create_response = admin_client.post(
        "/api/v1/admin/clients",
        headers={"X-Admin-Token": "test-admin-token"},
        json={
            "client_id": "svc-reset",
            "namespaces": ["team-x"],
        },
    )
    assert create_response.status_code == 201
    old_secret = create_response.json()["client_secret"]

    reset_response = admin_client.post(
        "/api/v1/admin/clients/svc-reset/reset-secret",
        headers={"X-Admin-Token": "test-admin-token"},
    )
    assert reset_response.status_code == 200
    new_secret = reset_response.json()["client_secret"]
    assert new_secret != old_secret

    old_token_resp = admin_client.post(
        "/api/v1/auth/token",
        json={"client_id": "svc-reset", "client_secret": old_secret},
    )
    assert old_token_resp.status_code == 401

    new_token_resp = admin_client.post(
        "/api/v1/auth/token",
        json={"client_id": "svc-reset", "client_secret": new_secret},
    )
    assert new_token_resp.status_code == 200


def test_issue_token_prefers_mongodb_client_over_env_fallback(admin_client):
    create_response = admin_client.post(
        "/api/v1/admin/clients",
        headers={"X-Admin-Token": "test-admin-token"},
        json={
            "client_id": "svc-agent",
            "namespaces": ["team-b"],
            "description": "mongodb override",
        },
    )

    assert create_response.status_code == 201
    client_secret = create_response.json()["client_secret"]

    token_response = admin_client.post(
        "/api/v1/auth/token",
        json={"client_id": "svc-agent", "client_secret": client_secret},
    )

    assert token_response.status_code == 200
    payload = token_response.json()
    assert payload["service_id"] == "svc-agent"
    assert payload["namespaces"] == ["team-b"]

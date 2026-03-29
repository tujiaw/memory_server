import os
from unittest.mock import MagicMock

from fastapi import FastAPI
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

from app.core.config import settings
from app.api.mem0_routes import router as memory_router
from app.api.user_routes import router as subject_router
from app.services.auth_service import auth_service
from app.services.mem0_service import mem0_service
from app.services.user_service import user_service


def build_app() -> TestClient:
    app = FastAPI()
    app.include_router(memory_router, prefix="/api/v1")
    app.include_router(subject_router, prefix="/api/v1")
    return TestClient(app)


def auth_headers(namespaces=None):
    token = auth_service.create_service_token(
        service_id="svc-agent",
        namespaces=namespaces or ["team-a"],
    )
    return {"Authorization": f"Bearer {token}"}


def test_add_memory_uses_subject_scope(monkeypatch):
    captured = {}

    async def fake_add_memory(**kwargs):
        captured.update(kwargs)
        return {"id": "mem-1", "text": "prefers python", "metadata": kwargs.get("metadata", {})}

    monkeypatch.setattr(mem0_service, "add_memory", fake_add_memory)

    client = build_app()
    response = client.post(
        "/api/v1/memories",
        headers=auth_headers(),
        json={
            "namespace": "team-a",
            "subject_id": "subject-1",
            "run_id": "session-1",
            "content": "prefers python",
            "metadata": {"category": "preference"},
        },
    )

    assert response.status_code == 201
    assert captured["namespace"] == "team-a"
    assert captured["subject_id"] == "subject-1"
    assert captured["run_id"] == "session-1"


def test_add_memory_rejects_unauthorized_namespace():
    client = build_app()
    response = client.post(
        "/api/v1/memories",
        headers=auth_headers(namespaces=["team-a"]),
        json={
            "namespace": "team-b",
            "subject_id": "subject-1",
            "content": "prefers python",
        },
    )

    assert response.status_code == 403


def test_search_memories_route_returns_items(monkeypatch):
    captured = {}

    async def fake_search_memories_scoped(**kwargs):
        captured.update(kwargs)
        return {
            "items": [
                {
                    "id": "m1",
                    "text": "用户喜欢 Python",
                    "vector_score": 0.91,
                    "lexical_score": None,
                    "match_sources": ["vector"],
                }
            ],
        }

    monkeypatch.setattr(
        mem0_service,
        "search_memories_scoped",
        fake_search_memories_scoped,
        raising=False,
    )

    client = build_app()
    response = client.post(
        "/api/v1/memories/search",
        headers=auth_headers(),
        json={
            "namespace": "team-a",
            "subject_id": "subject-1",
            "query": "用户喜欢什么",
            "limit": 5,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 1
    assert captured["mode"] == "hybrid"
    assert "vector_min_score" not in captured


def test_get_all_memories_route_returns_items(monkeypatch):
    async def fake_get_all_memories_scoped(**kwargs):
        return {
            "items": [{"id": "m1", "text": "用户在上海工作"}],
        }

    monkeypatch.setattr(
        mem0_service,
        "get_all_memories_scoped",
        fake_get_all_memories_scoped,
        raising=False,
    )

    client = build_app()
    response = client.get(
        "/api/v1/memories/team-a/subject-1",
        headers=auth_headers(),
    )

    assert response.status_code == 200
    data = response.json()
    assert data["count"] == 1


def test_subject_context_route_uses_subject_prefix(monkeypatch):
    async def fake_set_subject_context(**kwargs):
        return {
            "subject_id": kwargs["subject_id"],
            "namespace": kwargs["namespace"],
            "name": kwargs["context"].get("name"),
        }

    monkeypatch.setattr(mem0_service, "set_subject_context", fake_set_subject_context, raising=False)

    client = build_app()
    response = client.put(
        "/api/v1/subjects/subject-1/context",
        headers=auth_headers(),
        json={
            "namespace": "team-a",
            "name": "Alice",
            "role": "engineer",
            "preferences": {"language": "Python"},
        },
    )

    assert response.status_code == 200
    assert response.json()["data"]["subject_id"] == "subject-1"


class FakeMemoryClient:
    def __init__(self):
        self.add_calls = []
        self.search_calls = []

    def add(self, messages, user_id=None, agent_id=None, run_id=None, metadata=None, infer=True):
        self.add_calls.append(
            {
                "messages": messages,
                "user_id": user_id,
                "agent_id": agent_id,
                "run_id": run_id,
                "metadata": metadata,
                "infer": infer,
            }
        )
        return {
            "results": [
                {
                    "id": "mem-1",
                    "memory": "prefers python",
                    "metadata": metadata or {},
                    "user_id": user_id,
                    "agent_id": agent_id,
                    "run_id": run_id,
                }
            ]
        }

    def search(
        self,
        query,
        user_id=None,
        agent_id=None,
        run_id=None,
        limit=5,
        filters=None,
        threshold=None,
    ):
        self.search_calls.append(
            {
                "query": query,
                "user_id": user_id,
                "agent_id": agent_id,
                "run_id": run_id,
                "limit": limit,
                "filters": filters,
                "threshold": threshold,
            }
        )
        return {
            "results": [
                {
                    "id": "mem-1",
                    "memory": "prefers python",
                    "score": 0.88,
                    "metadata": {"category": "preference"},
                    "user_id": user_id,
                    "agent_id": agent_id,
                    "run_id": run_id,
                }
            ]
        }


@pytest.mark.asyncio
async def test_mem0_service_add_memory_passes_scope_and_context(monkeypatch):
    fake_client = FakeMemoryClient()
    tracked = {}

    def fake_get_memory_client():
        return fake_client

    monkeypatch.setattr(mem0_service, "_get_memory_client", fake_get_memory_client)

    async def fake_get_subject_context(namespace, subject_id):
        return {"name": "Alice"}

    async def fake_track(namespace, subject_id, write_count):
        tracked.update(
            {"namespace": namespace, "subject_id": subject_id, "write_count": write_count}
        )

    monkeypatch.setattr(mem0_service, "_get_subject_context", fake_get_subject_context, raising=False)
    monkeypatch.setattr(mem0_service, "_track_successful_write", fake_track, raising=False)

    result = await mem0_service.add_memory(
        namespace="team-a",
        subject_id="subject-1",
        content="prefers python",
        metadata={"category": "preference"},
        run_id="session-1",
    )

    assert fake_client.add_calls[0]["user_id"] == "subject-1"
    assert fake_client.add_calls[0]["agent_id"] == "team-a"
    assert fake_client.add_calls[0]["run_id"] == "session-1"
    assert tracked == {"namespace": "team-a", "subject_id": "subject-1", "write_count": 1}
    assert result["text"] == "prefers python"


def test_mem0_service_get_global_config_has_no_graph_store():
    config = mem0_service._get_global_config()
    assert "graph_store" not in config


def test_mem0_memory_client_forces_graph_disabled(monkeypatch):
    from mem0 import Memory

    def fake_from_config(_cfg):
        m = MagicMock()
        m.enable_graph = True
        return m

    monkeypatch.setattr(Memory, "from_config", staticmethod(fake_from_config))
    monkeypatch.setattr(mem0_service, "_memory_client", None)
    try:
        client = mem0_service._get_memory_client()
        assert client.enable_graph is False
    finally:
        mem0_service._memory_client = None


@pytest.mark.asyncio
async def test_mem0_service_search_passes_filters(monkeypatch):
    fake_client = FakeMemoryClient()

    def fake_get_memory_client():
        return fake_client

    monkeypatch.setattr(mem0_service, "_get_memory_client", fake_get_memory_client)
    async def fake_touch_subject(namespace, subject_id):
        return None

    monkeypatch.setattr(user_service, "touch_subject", fake_touch_subject)

    results = await mem0_service.search_memories(
        namespace="team-a",
        subject_id="subject-1",
        query="python",
        limit=3,
        mode="vector",
        run_id="session-1",
        filters={"category": "preference"},
    )

    assert fake_client.search_calls[0]["agent_id"] == "team-a"
    assert fake_client.search_calls[0]["user_id"] == "subject-1"
    assert fake_client.search_calls[0]["run_id"] == "session-1"
    assert fake_client.search_calls[0]["filters"] == {"category": "preference"}
    assert fake_client.search_calls[0]["threshold"] == settings.MEM0_SEARCH_VECTOR_MIN_SCORE
    assert results[0]["text"] == "prefers python"

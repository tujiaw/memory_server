import os

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest


os.environ.setdefault("OPENAI_API_KEY", "test-openai-key")
os.environ.setdefault("OPENAI_BASE_URL", "https://example.com/v1")
os.environ.setdefault("SECRET_KEY", "test-secret-key")
os.environ.setdefault(
    "SERVICE_CLIENTS_JSON",
    '{"svc-agent":{"secret":"top-secret","namespaces":["team-a"]}}',
)

from app.api.mem0_routes import router as memory_router
from app.api.user_routes import router as subject_router
from app.core.config import settings
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


def test_memory_context_returns_llm_ready_string(monkeypatch):
    captured = {}

    async def fake_get_context_for_llm(**kwargs):
        captured.update(kwargs)
        return {
            "context": "以下是关于该用户的关键记忆：\n\n1. 用户喜欢 Python\n2. 用户在上海工作",
            "count": 2,
            "query": "用户偏好",
            "enhanced_query": "用户的技术偏好和工作背景",
            "history_used": ["用户喜欢 Python"],
            "sources": [
                {"id": "m1", "text": "用户喜欢 Python", "score": 0.9},
                {"id": "m2", "text": "用户在上海工作", "score": 0.85},
            ],
            "relations": [
                {"source": "user1", "relationship": "works_in", "target": "上海"},
            ],
        }

    monkeypatch.setattr(mem0_service, "get_context_for_llm", fake_get_context_for_llm)

    client = build_app()
    response = client.post(
        "/api/v1/memories/context",
        headers=auth_headers(),
        json={
            "namespace": "team-a",
            "subject_id": "subject-1",
            "query": "用户偏好",
            "limit": 15,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert "以下是关于该用户的关键记忆" in data["context"]
    assert data["count"] == 2
    assert data["query"] == "用户偏好"
    assert data["enhanced_query"] == "用户的技术偏好和工作背景"
    assert data["history_used"] == ["用户喜欢 Python"]
    assert captured["min_score"] == 0.5
    assert captured["enable_query_rewrite"] is True
    assert captured["enable_graph_search"] is True
    assert len(data["sources"]) == 2
    assert data["relations"] == [{"source": "user1", "relationship": "works_in", "target": "上海"}]
    assert "prompt_block" not in data
    assert "usage_hint" not in data


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


def test_search_memories_route_returns_relations(monkeypatch):
    async def fake_search_memories_with_relations(**kwargs):
        return {
            "items": [{"id": "m1", "text": "用户喜欢 Python", "score": 0.91}],
            "relations": [{"source": "user1", "relationship": "likes", "target": "Python"}],
        }

    monkeypatch.setattr(
        mem0_service,
        "search_memories_with_relations",
        fake_search_memories_with_relations,
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
    assert data["relations"] == [{"source": "user1", "relationship": "likes", "target": "Python"}]


def test_get_all_memories_route_returns_relations(monkeypatch):
    async def fake_get_all_memories_with_relations(**kwargs):
        return {
            "items": [{"id": "m1", "text": "用户在上海工作"}],
            "relations": [{"source": "user1", "relationship": "works_in", "target": "上海"}],
        }

    monkeypatch.setattr(
        mem0_service,
        "get_all_memories_with_relations",
        fake_get_all_memories_with_relations,
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
    assert data["relations"] == [{"source": "user1", "relationship": "works_in", "target": "上海"}]


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

    def search(self, query, user_id=None, agent_id=None, run_id=None, limit=5, filters=None):
        self.search_calls.append(
            {
                "query": query,
                "user_id": user_id,
                "agent_id": agent_id,
                "run_id": run_id,
                "limit": limit,
                "filters": filters,
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

    monkeypatch.setattr(mem0_service, "get_memory_client", lambda namespace, subject_id: fake_client)

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
    assert result["relations"] == []


def test_mem0_service_get_config_includes_graph_store_when_neo4j_is_configured(monkeypatch):
    monkeypatch.setattr(settings, "NEO4J_URL", "bolt://localhost:7687", raising=False)
    monkeypatch.setattr(settings, "NEO4J_USERNAME", "neo4j", raising=False)
    monkeypatch.setattr(settings, "NEO4J_PASSWORD", "password", raising=False)

    config = mem0_service._get_config(namespace="team-a", subject_id="subject-1")

    assert config["graph_store"]["provider"] == "neo4j"
    assert config["graph_store"]["config"]["url"] == "bolt://localhost:7687"
    assert config["graph_store"]["config"]["username"] == "neo4j"
    assert config["graph_store"]["config"]["password"] == "password"


@pytest.mark.asyncio
async def test_mem0_service_search_passes_filters(monkeypatch):
    fake_client = FakeMemoryClient()

    monkeypatch.setattr(mem0_service, "get_memory_client", lambda namespace, subject_id: fake_client)
    async def fake_touch_subject(namespace, subject_id):
        return None

    monkeypatch.setattr(user_service, "touch_subject", fake_touch_subject)

    results = await mem0_service.search_memories(
        namespace="team-a",
        subject_id="subject-1",
        query="python",
        limit=3,
        run_id="session-1",
        filters={"category": "preference"},
    )

    assert fake_client.search_calls[0]["agent_id"] == "team-a"
    assert fake_client.search_calls[0]["user_id"] == "subject-1"
    assert fake_client.search_calls[0]["run_id"] == "session-1"
    assert fake_client.search_calls[0]["filters"] == {"category": "preference"}
    assert results[0]["text"] == "prefers python"


@pytest.mark.asyncio
async def test_mem0_service_context_uses_llm_rewritten_query(monkeypatch):
    captured = {}

    async def fake_search_memories_with_relations(**kwargs):
        captured["query"] = kwargs["query"]
        return {
            "items": [
                {
                    "id": "m1",
                    "text": "用户喜欢 Python",
                    "score": 0.95,
                    "metadata": {"category": "preference"},
                    "created_at": "2026-03-06T10:00:00+00:00",
                },
                {
                    "id": "m2",
                    "text": "用户在上海工作",
                    "score": 0.88,
                    "metadata": {"category": "profile"},
                    "created_at": "2026-03-05T10:00:00+00:00",
                },
            ],
            "relations": [{"source": "user1", "relationship": "works_in", "target": "上海"}],
        }

    async def fake_get_all_memories_with_relations(**kwargs):
        return {
            "items": [
                {
                    "id": "m2",
                    "text": "用户在上海工作",
                    "score": None,
                    "metadata": {"category": "profile"},
                    "created_at": "2026-03-05T10:00:00+00:00",
                },
                {
                    "id": "m3",
                    "text": "最近在评估 LangGraph 方案",
                    "score": None,
                    "metadata": {"category": "recent"},
                    "created_at": "2026-03-06T12:00:00+00:00",
                },
            ],
            "relations": [{"source": "user1", "relationship": "evaluating", "target": "LangGraph"}],
        }

    async def fake_rewrite_query_with_llm(query, history_items):
        captured["rewrite_input"] = {
            "query": query,
            "history_items": history_items,
        }
        return "用户当前的技术偏好、所在城市和最近关注项目"

    monkeypatch.setattr(mem0_service, "search_memories_with_relations", fake_search_memories_with_relations)
    monkeypatch.setattr(mem0_service, "get_all_memories_with_relations", fake_get_all_memories_with_relations)
    monkeypatch.setattr(mem0_service, "_rewrite_query_with_llm", fake_rewrite_query_with_llm, raising=False)

    result = await mem0_service.get_context_for_llm(
        namespace="team-a",
        subject_id="subject-1",
        query="用户背景和最近在做什么",
        limit=10,
    )

    assert "以下是与当前问题最相关的用户记忆" in result["context"]
    assert len(result["sources"]) == 3
    assert result["query"] == "用户背景和最近在做什么"
    assert captured["rewrite_input"] == {
        "query": "用户背景和最近在做什么",
        "history_items": ["最近在评估 LangGraph 方案", "用户在上海工作"],
    }
    assert result["enhanced_query"] == "用户当前的技术偏好、所在城市和最近关注项目"
    assert captured["query"] == result["enhanced_query"]
    assert result["history_used"] == ["最近在评估 LangGraph 方案", "用户在上海工作"]
    assert [item["id"] for item in result["sources"]] == ["m1", "m2", "m3"]
    assert result["relations"] == [
        {"source": "user1", "relationship": "works_in", "target": "上海"},
        {"source": "user1", "relationship": "evaluating", "target": "LangGraph"},
    ]
    assert "prompt_block" not in result
    assert "usage_hint" not in result
    assert "以下是与当前问题最相关的用户记忆" in result["context"]
    assert "以下是最近记忆补充" in result["context"]
    assert "以下是可用于推理的关系图谱（三元组）" in result["context"]
    assert "user1 --works_in--> 上海" in result["context"]
    assert "回答规则：" in result["context"]
    assert "用户喜欢 Python" in result["context"]
    assert "最近在评估 LangGraph 方案" in result["context"]


@pytest.mark.asyncio
async def test_mem0_service_context_filters_relevant_memories_by_min_score(monkeypatch):
    async def fake_search_memories_with_relations(**kwargs):
        return {
            "items": [
                {
                    "id": "m1",
                    "text": "高相关信息",
                    "score": 0.91,
                    "created_at": "2026-03-06T10:00:00+00:00",
                },
                {
                    "id": "m2",
                    "text": "低相关信息",
                    "score": 0.42,
                    "created_at": "2026-03-05T10:00:00+00:00",
                },
            ],
            "relations": [],
        }

    async def fake_get_all_memories_with_relations(**kwargs):
        return {
            "items": [
                {
                    "id": "m3",
                    "text": "最近记忆",
                    "created_at": "2026-03-06T12:00:00+00:00",
                }
            ],
            "relations": [],
        }

    async def fake_rewrite_query_with_llm(query, history_items):
        return "增强后的 query"

    monkeypatch.setattr(mem0_service, "search_memories_with_relations", fake_search_memories_with_relations)
    monkeypatch.setattr(mem0_service, "get_all_memories_with_relations", fake_get_all_memories_with_relations)
    monkeypatch.setattr(mem0_service, "_rewrite_query_with_llm", fake_rewrite_query_with_llm, raising=False)

    result = await mem0_service.get_context_for_llm(
        namespace="team-a",
        subject_id="subject-1",
        query="查询",
        limit=10,
        min_score=0.5,
    )

    assert [item["id"] for item in result["sources"]] == ["m1", "m3"]
    assert "低相关信息" not in result["context"]
    assert "高相关信息" in result["context"]


@pytest.mark.asyncio
async def test_mem0_service_context_drops_low_score_relevant_results_when_recent_fallback_exists(monkeypatch):
    async def fake_search_memories_with_relations(**kwargs):
        return {
            "items": [
                {
                    "id": "m1",
                    "text": "最好的低分结果",
                    "score": 0.49,
                    "created_at": "2026-03-06T10:00:00+00:00",
                },
                {
                    "id": "m2",
                    "text": "次优低分结果",
                    "score": 0.31,
                    "created_at": "2026-03-05T10:00:00+00:00",
                },
                {
                    "id": "m3",
                    "text": "更差的低分结果",
                    "score": 0.12,
                    "created_at": "2026-03-04T10:00:00+00:00",
                },
            ],
            "relations": [],
        }

    async def fake_get_all_memories_with_relations(**kwargs):
        return {
            "items": [
                {
                    "id": "m4",
                    "text": "最近补充记忆",
                    "created_at": "2026-03-07T10:00:00+00:00",
                }
            ],
            "relations": [],
        }

    async def fake_rewrite_query_with_llm(query, history_items):
        return "增强后的 query"

    monkeypatch.setattr(mem0_service, "search_memories_with_relations", fake_search_memories_with_relations)
    monkeypatch.setattr(mem0_service, "get_all_memories_with_relations", fake_get_all_memories_with_relations)
    monkeypatch.setattr(mem0_service, "_rewrite_query_with_llm", fake_rewrite_query_with_llm, raising=False)

    result = await mem0_service.get_context_for_llm(
        namespace="team-a",
        subject_id="subject-1",
        query="查询",
        limit=10,
        min_score=0.5,
    )

    assert [item["id"] for item in result["sources"]] == ["m4"]
    assert "最好的低分结果" not in result["context"]
    assert "次优低分结果" not in result["context"]
    assert "更差的低分结果" not in result["context"]
    assert "最近补充记忆" in result["context"]


@pytest.mark.asyncio
async def test_mem0_service_context_falls_back_to_original_query_when_rewrite_fails(monkeypatch):
    captured = {}

    async def fake_search_memories_with_relations(**kwargs):
        captured["query"] = kwargs["query"]
        return {"items": [], "relations": []}

    async def fake_get_all_memories_with_relations(**kwargs):
        return {
            "items": [
                {
                    "id": "m1",
                    "text": "用户喜欢 Python",
                    "created_at": "2026-03-06T10:00:00+00:00",
                }
            ],
            "relations": [],
        }

    async def fake_rewrite_query_with_llm(query, history_items):
        raise RuntimeError("boom")

    monkeypatch.setattr(mem0_service, "search_memories_with_relations", fake_search_memories_with_relations)
    monkeypatch.setattr(mem0_service, "get_all_memories_with_relations", fake_get_all_memories_with_relations)
    monkeypatch.setattr(mem0_service, "_rewrite_query_with_llm", fake_rewrite_query_with_llm, raising=False)

    result = await mem0_service.get_context_for_llm(
        namespace="team-a",
        subject_id="subject-1",
        query=" 用户喜欢什么语言 ",
        limit=5,
    )

    assert captured["query"] == "用户喜欢什么语言"
    assert result["query"] == "用户喜欢什么语言"
    assert result["enhanced_query"] == "用户喜欢什么语言"
    assert result["history_used"] == []


@pytest.mark.asyncio
async def test_mem0_service_context_falls_back_to_original_query_when_rewrite_returns_empty(monkeypatch):
    captured = {}

    async def fake_search_memories_with_relations(**kwargs):
        captured["query"] = kwargs["query"]
        return {"items": [], "relations": []}

    async def fake_get_all_memories_with_relations(**kwargs):
        return {
            "items": [
                {
                    "id": "m1",
                    "text": "用户最近在调试 JWT 鉴权",
                    "created_at": "2026-03-06T10:00:00+00:00",
                }
            ],
            "relations": [],
        }

    async def fake_rewrite_query_with_llm(query, history_items):
        return "   "

    monkeypatch.setattr(mem0_service, "search_memories_with_relations", fake_search_memories_with_relations)
    monkeypatch.setattr(mem0_service, "get_all_memories_with_relations", fake_get_all_memories_with_relations)
    monkeypatch.setattr(mem0_service, "_rewrite_query_with_llm", fake_rewrite_query_with_llm, raising=False)

    result = await mem0_service.get_context_for_llm(
        namespace="team-a",
        subject_id="subject-1",
        query="最近在做什么",
        limit=5,
    )

    assert captured["query"] == "最近在做什么"
    assert result["enhanced_query"] == "最近在做什么"
    assert result["history_used"] == []


@pytest.mark.asyncio
async def test_mem0_service_context_without_query_returns_recent_memories_only(monkeypatch):
    async def fake_search_memories_with_relations(**kwargs):
        raise AssertionError("search_memories_with_relations should not be called when query is empty")

    async def fake_rewrite_query_with_llm(query, history_items):
        raise AssertionError("_rewrite_query_with_llm should not be called when query is empty")

    async def fake_get_all_memories_with_relations(**kwargs):
        return {
            "items": [
                {
                    "id": "m3",
                    "text": "最近在评估 LangGraph 方案",
                    "score": None,
                    "metadata": {"category": "recent"},
                    "created_at": "2026-03-06T12:00:00+00:00",
                }
            ],
            "relations": [],
        }

    monkeypatch.setattr(mem0_service, "search_memories_with_relations", fake_search_memories_with_relations)
    monkeypatch.setattr(mem0_service, "get_all_memories_with_relations", fake_get_all_memories_with_relations)
    monkeypatch.setattr(mem0_service, "_rewrite_query_with_llm", fake_rewrite_query_with_llm, raising=False)

    result = await mem0_service.get_context_for_llm(
        namespace="team-a",
        subject_id="subject-1",
        query="  ",
        limit=10,
    )

    assert "以下是该用户的可用记忆" in result["context"]
    assert "回答规则：" in result["context"]
    assert result["query"] is None
    assert result["enhanced_query"] is None
    assert result["history_used"] == []
    assert result["count"] == 1


@pytest.mark.asyncio
async def test_mem0_service_context_can_disable_query_rewrite(monkeypatch):
    captured = {}

    async def fake_search_memories_with_relations(**kwargs):
        captured["query"] = kwargs["query"]
        return {"items": [], "relations": []}

    async def fake_get_all_memories_with_relations(**kwargs):
        return {
            "items": [
                {
                    "id": "m1",
                    "text": "用户喜欢 Python",
                    "created_at": "2026-03-06T10:00:00+00:00",
                }
            ],
            "relations": [],
        }

    async def fake_rewrite_query_with_llm(query, history_items):
        raise AssertionError("_rewrite_query_with_llm should not be called when rewrite is disabled")

    monkeypatch.setattr(mem0_service, "search_memories_with_relations", fake_search_memories_with_relations)
    monkeypatch.setattr(mem0_service, "get_all_memories_with_relations", fake_get_all_memories_with_relations)
    monkeypatch.setattr(mem0_service, "_rewrite_query_with_llm", fake_rewrite_query_with_llm, raising=False)

    result = await mem0_service.get_context_for_llm(
        namespace="team-a",
        subject_id="subject-1",
        query="用户喜欢什么语言",
        limit=5,
        enable_query_rewrite=False,
    )

    assert captured["query"] == "用户喜欢什么语言"
    assert result["enhanced_query"] == "用户喜欢什么语言"
    assert result["history_used"] == []


@pytest.mark.asyncio
async def test_mem0_service_context_can_disable_graph_search(monkeypatch):
    async def fail_get_all_memories_with_relations(**kwargs):
        raise AssertionError("graph get_all should not be called when graph search is disabled")

    async def fail_search_memories_with_relations(**kwargs):
        raise AssertionError("graph search should not be called when graph search is disabled")

    async def fake_get_all_memories_vector_only(**kwargs):
        return {
            "items": [
                {
                    "id": "m1",
                    "text": "用户在上海工作",
                    "created_at": "2026-03-06T10:00:00+00:00",
                }
            ],
            "relations": [],
        }

    async def fake_search_memories_vector_only(**kwargs):
        return {
            "items": [
                {
                    "id": "m2",
                    "text": "用户喜欢 Python",
                    "score": 0.91,
                    "created_at": "2026-03-06T11:00:00+00:00",
                }
            ],
            "relations": [],
        }

    monkeypatch.setattr(mem0_service, "get_all_memories_with_relations", fail_get_all_memories_with_relations)
    monkeypatch.setattr(mem0_service, "search_memories_with_relations", fail_search_memories_with_relations)
    monkeypatch.setattr(mem0_service, "_get_all_memories_vector_only", fake_get_all_memories_vector_only, raising=False)
    monkeypatch.setattr(mem0_service, "_search_memories_vector_only", fake_search_memories_vector_only, raising=False)

    result = await mem0_service.get_context_for_llm(
        namespace="team-a",
        subject_id="subject-1",
        query="用户背景",
        limit=5,
        enable_graph_search=False,
    )

    assert result["relations"] == []
    assert [item["id"] for item in result["sources"]] == ["m2", "m1"]

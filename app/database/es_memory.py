"""Elasticsearch：与 mem0 共用同一索引（dense_vector + metadata.data 全文 BM25）。"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List, Optional

from elasticsearch import Elasticsearch

from app.core.config import settings

logger = logging.getLogger(__name__)


def _es_hosts() -> List[str]:
    h, p = settings.ELASTICSEARCH_HOST, settings.ELASTICSEARCH_PORT
    if p is None:
        return [h]
    return [f"{h}:{p}"]


def make_elasticsearch_client() -> Elasticsearch:
    return Elasticsearch(
        hosts=_es_hosts(),
        basic_auth=(settings.ELASTICSEARCH_USER, settings.ELASTICSEARCH_PASSWORD),
        verify_certs=settings.ELASTICSEARCH_VERIFY_CERTS,
        request_timeout=30,
    )


def ensure_mem0_index(client: Elasticsearch, index_name: str, vector_dims: int) -> None:
    """在 mem0 首次写入前创建索引；mapping 含 vector + metadata.data（BM25）。"""
    if client.indices.exists(index=index_name):
        return
    body: Dict[str, Any] = {
        "settings": {
            "index": {
                "number_of_shards": 1,
                "number_of_replicas": 0,
                "refresh_interval": "1s",
            }
        },
        "mappings": {
            "properties": {
                "vector": {
                    "type": "dense_vector",
                    "dims": vector_dims,
                    "index": True,
                    "similarity": "cosine",
                },
                "metadata": {
                    "type": "object",
                    "dynamic": True,
                    "properties": {
                        "user_id": {"type": "keyword"},
                        "agent_id": {"type": "keyword"},
                        "run_id": {"type": "keyword"},
                        "data": {"type": "text"},
                        "hash": {"type": "keyword"},
                        "created_at": {"type": "date", "ignore_malformed": True},
                        "updated_at": {"type": "date", "ignore_malformed": True},
                    },
                },
            }
        },
    }
    client.indices.create(
        index=index_name,
        settings=body["settings"],
        mappings=body["mappings"],
    )
    logger.info("Created Elasticsearch index %s", index_name)


def _filter_clauses(
    namespace: str,
    subject_id: str,
    run_id: Optional[str],
    filters: Optional[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    clauses: List[Dict[str, Any]] = [
        {"term": {"metadata.user_id": subject_id}},
        {"term": {"metadata.agent_id": namespace}},
    ]
    if run_id is not None:
        clauses.append({"term": {"metadata.run_id": run_id}})
    if filters:
        for key, value in filters.items():
            if value is None:
                continue
            if isinstance(value, (str, int, float, bool)):
                clauses.append({"term": {f"metadata.{key}": value}})
            else:
                clauses.append({"match": {f"metadata.{key}": json.dumps(value, sort_keys=True)}})
    return clauses


def search_bm25_lexical(
    client: Elasticsearch,
    index_name: str,
    namespace: str,
    subject_id: str,
    query: str,
    run_id: Optional[str],
    filters: Optional[Dict[str, Any]],
    limit: int,
) -> List[Dict[str, Any]]:
    """BM25 检索 metadata.data，过滤维度与 mem0 向量检索一致。"""
    if not query.strip():
        return []
    must_query: Dict[str, Any] = {
        "multi_match": {
            "query": query,
            "fields": ["metadata.data"],
            "type": "best_fields",
            "operator": "or",
        }
    }
    body = {
        "query": {
            "bool": {
                "must": [must_query],
                "filter": _filter_clauses(namespace, subject_id, run_id, filters),
            }
        },
        "size": limit,
    }
    resp = client.search(index=index_name, query=body["query"], size=limit)
    out: List[Dict[str, Any]] = []
    for hit in resp.get("hits", {}).get("hits", []):
        src = hit.get("_source") or {}
        meta = src.get("metadata") or {}
        text = meta.get("data") or ""
        out.append(
            {
                "id": hit["_id"],
                "text": text,
                "metadata": {k: v for k, v in meta.items() if k != "data"},
                "score": float(hit.get("_score") or 0.0),
                "namespace": namespace,
                "subject_id": subject_id,
                "run_id": meta.get("run_id"),
                "created_at": meta.get("created_at"),
                "updated_at": meta.get("updated_at"),
            }
        )
    return out


def cluster_health_snippet(client: Elasticsearch) -> str:
    try:
        h = client.cluster.health(request_timeout=5)
        return str(h.get("status", "unknown"))
    except Exception:
        return "unhealthy"

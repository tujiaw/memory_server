import json
import asyncio
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from mem0 import Memory

from app.core.config import settings
from app.database import postgres as pg_database
from app.database.sql import MEMORY_LEXICAL_DELETE, MEMORY_LEXICAL_UPDATE, MEMORY_LEXICAL_UPSERT
from app.services.user_service import user_service


class Mem0Service:
    """Service for managing subject-scoped memories on top of mem0."""

    def __init__(self):
        self._memory_client: Optional[Memory] = None

    @staticmethod
    def _get_global_config() -> dict:
        """单集合向量库 + LLM；不包含 graph_store，知识图谱检索保持关闭。"""
        return {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": settings.QDRANT_MEM0_COLLECTION,
                    "host": settings.QDRANT_HOST,
                    "port": settings.QDRANT_PORT,
                    "embedding_model_dims": settings.VECTOR_SIZE,
                },
            },
            "llm": {
                "provider": "openai",
                "config": {
                    "model": settings.OPENAI_MODEL,
                    "temperature": 0.2,
                    "max_tokens": 1500,
                },
            },
            "embedder": {
                "provider": "openai",
                "config": {
                    "model": settings.OPENAI_EMBEDDING_MODEL,
                },
            },
            "version": "v1.1",
        }

    def _get_memory_client(self) -> Memory:
        """Get or create the global Memory client."""
        if self._memory_client is None:
            client = Memory.from_config(self._get_global_config())
            client.enable_graph = False
            self._memory_client = client
        return self._memory_client

    async def _get_subject_context(self, namespace: str, subject_id: str) -> Dict[str, Any]:
        subject = await user_service.get_subject_context(namespace, subject_id)
        return subject or {}

    async def _track_successful_write(self, namespace: str, subject_id: str, write_count: int) -> None:
        await user_service.touch_subject(namespace, subject_id)
        if write_count > 0:
            await user_service.increment_memory_count(namespace, subject_id, amount=write_count)

    @staticmethod
    def _build_metadata(
        metadata: Optional[Dict[str, Any]],
        subject_context: Dict[str, Any],
    ) -> Dict[str, Any]:
        merged = dict(metadata or {})
        if subject_context.get("name") and "subject_name" not in merged:
            merged["subject_name"] = subject_context["name"]
        if subject_context.get("role") and "subject_role" not in merged:
            merged["subject_role"] = subject_context["role"]
        return merged

    @staticmethod
    def _context_message(subject_context: Dict[str, Any]) -> Optional[Dict[str, str]]:
        if not subject_context:
            return None

        compact_context = {
            key: value
            for key, value in subject_context.items()
            if key in {"name", "role", "preferences", "custom_data"} and value
        }
        if not compact_context:
            return None

        return {
            "role": "system",
            "content": f"Subject context: {compact_context}",
        }

    def _prepare_messages(self, content: str, subject_context: Dict[str, Any]) -> List[Dict[str, str]]:
        messages = [{"role": "user", "content": content}]
        context_message = self._context_message(subject_context)
        if context_message:
            return [context_message, *messages]
        return messages

    def _prepare_conversation_messages(
        self,
        messages: List[Dict[str, str]],
        subject_context: Dict[str, Any],
    ) -> List[Dict[str, str]]:
        context_message = self._context_message(subject_context)
        if context_message:
            return [context_message, *messages]
        return messages

    @staticmethod
    def _dt_to_api_str(value: Any) -> Optional[str]:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value.isoformat()
        if isinstance(value, str):
            return value
        return str(value)

    @staticmethod
    def _normalize_memory_result(result: Dict[str, Any]) -> Dict[str, Any]:
        metadata = result.get("metadata")
        out: Dict[str, Any] = {
            "id": result.get("id"),
            "text": result.get("memory") or result.get("text", ""),
            "metadata": metadata,
            "namespace": result.get("agent_id") or result.get("namespace"),
            "subject_id": result.get("user_id") or result.get("subject_id"),
            "run_id": result.get("run_id"),
            "created_at": Mem0Service._dt_to_api_str(result.get("created_at")),
            "updated_at": Mem0Service._dt_to_api_str(result.get("updated_at")),
        }
        if "score" in result:
            out["score"] = result["score"]
        if "vector_score" in result:
            out["vector_score"] = result["vector_score"]
        if "lexical_score" in result:
            out["lexical_score"] = result["lexical_score"]
        if result.get("match_sources") is not None:
            out["match_sources"] = result["match_sources"]
        return out

    @staticmethod
    def _finalize_search_item(row: Dict[str, Any]) -> Dict[str, Any]:
        row = {k: v for k, v in row.items() if k != "_norm_score"}
        row.pop("score", None)
        row.pop("_hybrid_fusion", None)
        return Mem0Service._normalize_memory_result(row)

    def _normalize_results(self, payload: Any) -> List[Dict[str, Any]]:
        if isinstance(payload, dict):
            results = payload.get("results", [])
        else:
            results = payload or []
        return [self._normalize_memory_result(result) for result in results]

    @staticmethod
    def _score_to_unit_interval(raw: Any) -> float:
        """供融合用：余弦类 [0,1] 保持；BM25 等无界正分压缩到 (0,1)，避免单条时 _norm_score 缺失变 0。"""
        try:
            x = float(raw)
        except (TypeError, ValueError):
            return 0.0
        if x <= 0:
            return 0.0
        if x <= 1.0:
            return x
        return x / (x + 1.0)

    @staticmethod
    def _normalize_scores(items: List[Dict[str, Any]], score_key: str = "score") -> List[Dict[str, Any]]:
        if not items:
            return items
        scores = [item.get(score_key) or 0 for item in items]
        lo, hi = min(scores), max(scores)
        if hi <= lo:
            return [
                {**item, "_norm_score": Mem0Service._score_to_unit_interval(item.get(score_key))}
                for item in items
            ]
        return [
            {**item, "_norm_score": ((item.get(score_key) or 0) - lo) / (hi - lo)}
            for item in items
        ]

    @staticmethod
    def _merge_hybrid_results(
        vector_items: List[Dict[str, Any]],
        bm25_items: List[Dict[str, Any]],
        vector_weight: float,
        lexical_weight: float,
        limit: int,
    ) -> List[Dict[str, Any]]:
        vector_norm = Mem0Service._normalize_scores(vector_items)
        bm25_norm = Mem0Service._normalize_scores(bm25_items)
        id_to_vector = {item.get("id"): item for item in vector_norm}
        id_to_bm25 = {item.get("id"): item for item in bm25_norm}
        all_ids = set(id_to_vector) | set(id_to_bm25)
        merged: List[Dict[str, Any]] = []
        for mid in all_ids:
            v_item = id_to_vector.get(mid)
            b_item = id_to_bm25.get(mid)
            v_n = v_item.get("_norm_score", 0) if v_item else 0.0
            b_n = b_item.get("_norm_score", 0) if b_item else 0.0
            fusion = vector_weight * v_n + lexical_weight * b_n
            base_src = v_item if v_item else b_item
            base = {k: v for k, v in (base_src or {}).items() if k != "_norm_score"}
            v_raw = None
            b_raw = None
            if v_item is not None and v_item.get("score") is not None:
                v_raw = float(v_item["score"])
            if b_item is not None and b_item.get("score") is not None:
                b_raw = float(b_item["score"])
            match_sources: List[str] = []
            if v_item:
                match_sources.append("vector")
            if b_item:
                match_sources.append("bm25")
            merged.append(
                {
                    **base,
                    "_hybrid_fusion": fusion,
                    "vector_score": v_raw,
                    "lexical_score": b_raw,
                    "match_sources": match_sources,
                }
            )
        merged.sort(key=lambda x: x.get("_hybrid_fusion") or 0.0, reverse=True)
        return merged[:limit]

    @staticmethod
    def _deduplicate_memory_items(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        deduplicated: List[Dict[str, Any]] = []
        seen_keys = set()

        for item in items:
            text = (item.get("text") or "").strip()
            if not text:
                continue

            dedupe_key = (item.get("id"), text)
            if dedupe_key in seen_keys:
                continue

            seen_keys.add(dedupe_key)
            deduplicated.append(item)

        return deduplicated

    @staticmethod
    def _normalized_hybrid_weights() -> tuple[float, float]:
        vw = settings.MEM0_HYBRID_VECTOR_WEIGHT
        lw = settings.MEM0_HYBRID_BM25_WEIGHT
        total = vw + lw
        if total <= 0:
            return 0.7, 0.3
        return vw / total, lw / total

    @staticmethod
    def _effective_run_id(run_id: Optional[str]) -> Optional[str]:
        if run_id is None:
            return None
        if isinstance(run_id, str) and not run_id.strip():
            return None
        return run_id

    async def _persist_memory_lexical_batch(
        self,
        namespace: str,
        subject_id: str,
        run_id: Optional[str],
        items: List[Dict[str, Any]],
    ) -> None:
        if pg_database.postgres_db.pool is None:
            return
        now = datetime.now(timezone.utc)
        for item in items:
            mid = item.get("id")
            if mid is None:
                continue
            text = (item.get("text") or "").strip()
            if not text:
                continue
            meta = item.get("metadata") or {}

            def _coerce_dt(value: Any) -> datetime:
                if isinstance(value, datetime):
                    return value
                if isinstance(value, str):
                    try:
                        return datetime.fromisoformat(value.replace("Z", "+00:00"))
                    except ValueError:
                        return now
                return now

            ca = _coerce_dt(item.get("created_at"))
            ua = _coerce_dt(item.get("updated_at"))
            await pg_database.postgres_db.execute(
                MEMORY_LEXICAL_UPSERT,
                str(mid),
                namespace,
                subject_id,
                run_id,
                text,
                json.dumps(meta),
                ca,
                ua,
            )

    async def _delete_memory_lexical(self, memory_id: str) -> None:
        if pg_database.postgres_db.pool is None:
            return
        await pg_database.postgres_db.execute(MEMORY_LEXICAL_DELETE, memory_id)

    async def _update_memory_lexical(self, memory_id: str, content: str) -> None:
        if pg_database.postgres_db.pool is None:
            return
        now = datetime.now(timezone.utc)
        await pg_database.postgres_db.execute(MEMORY_LEXICAL_UPDATE, memory_id, content, now)

    async def _search_bm25_in_db(
        self,
        namespace: str,
        subject_id: str,
        query: str,
        run_id: Optional[str],
        filters: Optional[Dict[str, Any]],
        limit: int,
    ) -> List[Dict[str, Any]]:
        if pg_database.postgres_db.pool is None or not query.strip():
            return []
        conditions = ["namespace = $1", "subject_id = $2"]
        args: List[Any] = [namespace, subject_id]
        n = 3
        if run_id is not None:
            conditions.append(f"run_id = ${n}")
            args.append(run_id)
            n += 1
        if filters:
            conditions.append(f"metadata @> ${n}::jsonb")
            args.append(json.dumps(filters))
            n += 1
        # Lindera 中文分词索引 + 词项按 OR 匹配（与默认 @@@ 字符串的 AND 语义不同，避免「今天的天气」因文档无「的」而零命中）
        conditions.append(
            f"content @@@ paradedb.match('content', ${n}, conjunction_mode => false)"
        )
        args.append(query)
        n += 1
        sql = f"""
        SELECT memory_id, content, metadata, namespace, subject_id, run_id, created_at, updated_at,
               paradedb.score(memory_id) AS bm25_score
        FROM memory_lexical
        WHERE {" AND ".join(conditions)}
        ORDER BY paradedb.score(memory_id) DESC NULLS LAST
        LIMIT ${n}
        """
        args.append(limit)
        rows = await pg_database.postgres_db.fetch(sql, *args)
        out: List[Dict[str, Any]] = []
        for r in rows:
            meta = r["metadata"]
            if isinstance(meta, str):
                meta = json.loads(meta)
            score = r["bm25_score"]
            out.append(
                {
                    "id": r["memory_id"],
                    "text": r["content"],
                    "metadata": meta,
                    "score": float(score) if score is not None else 0.0,
                    "namespace": r["namespace"],
                    "subject_id": r["subject_id"],
                    "run_id": r["run_id"],
                    "created_at": r["created_at"],
                    "updated_at": r["updated_at"],
                }
            )
        return out

    async def add_memory(
        self,
        namespace: str,
        subject_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        run_id: Optional[str] = None,
        infer: bool = True,
    ) -> Dict[str, Any]:
        """Add a new memory for a subject."""
        memory_client = self._get_memory_client()
        subject_context = await self._get_subject_context(namespace, subject_id)
        result = await asyncio.to_thread(
            memory_client.add,
            messages=self._prepare_messages(content, subject_context),
            user_id=subject_id,
            agent_id=namespace,
            run_id=run_id,
            metadata=self._build_metadata(metadata, subject_context),
            infer=infer,
        )
        normalized_results = self._normalize_results(result)
        await self._track_successful_write(namespace, subject_id, len(normalized_results))
        scoped = [
            {**item, "namespace": namespace, "subject_id": subject_id, "run_id": run_id}
            for item in normalized_results
        ]
        await self._persist_memory_lexical_batch(namespace, subject_id, run_id, scoped)
        if normalized_results:
            return normalized_results[0]
        return {
            "text": content,
            "metadata": metadata or {},
            "namespace": namespace,
            "subject_id": subject_id,
            "run_id": run_id,
        }

    async def search_memories_scoped(
        self,
        namespace: str,
        subject_id: str,
        query: str,
        limit: int = 10,
        fusion: str = "hybrid",
        run_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        memory_client = self._get_memory_client()
        run_id = self._effective_run_id(run_id)
        vector_min_score = settings.MEM0_SEARCH_VECTOR_MIN_SCORE
        lexical_min_score = settings.MEM0_SEARCH_LEXICAL_MIN_SCORE
        min_fusion_score = settings.MEM0_SEARCH_MIN_FUSION_SCORE

        async def _vector_hits(search_limit: int, threshold: float) -> List[Dict[str, Any]]:
            payload = await asyncio.to_thread(
                memory_client.search,
                query=query,
                user_id=subject_id,
                agent_id=namespace,
                run_id=run_id,
                limit=search_limit,
                filters=filters,
                threshold=threshold,
            )
            return self._normalize_results(payload)

        if fusion == "vector":
            items_raw = await _vector_hits(limit, vector_min_score)
            await user_service.touch_subject(namespace, subject_id)
            out: List[Dict[str, Any]] = []
            for it in items_raw:
                vs = it.get("score")
                vf = float(vs) if vs is not None else None
                row = {**it, "vector_score": vf, "lexical_score": None, "match_sources": ["vector"]}
                out.append(self._finalize_search_item(row))
            return {"items": out}

        if fusion == "lexical":
            await user_service.touch_subject(namespace, subject_id)
            bm25_items = await self._search_bm25_in_db(
                namespace=namespace,
                subject_id=subject_id,
                query=query,
                run_id=run_id,
                filters=filters,
                limit=limit,
            )
            if lexical_min_score is not None:
                bm25_items = [x for x in bm25_items if (x.get("score") or 0) >= lexical_min_score]
            out_lex: List[Dict[str, Any]] = []
            for it in bm25_items:
                raw = float(it.get("score") or 0)
                row = {
                    **it,
                    "vector_score": None,
                    "lexical_score": raw,
                    "match_sources": ["bm25"],
                }
                out_lex.append(self._finalize_search_item(row))
            return {"items": out_lex}

        vw, lw = self._normalized_hybrid_weights()
        items_vec = await _vector_hits(limit, vector_min_score)
        bm25_items = await self._search_bm25_in_db(
            namespace=namespace,
            subject_id=subject_id,
            query=query,
            run_id=run_id,
            filters=filters,
            limit=limit,
        )
        if lexical_min_score is not None:
            bm25_items = [x for x in bm25_items if (x.get("score") or 0) >= lexical_min_score]

        await user_service.touch_subject(namespace, subject_id)
        if not items_vec and not bm25_items:
            return {"items": []}
        merged = self._merge_hybrid_results(items_vec, bm25_items, vw, lw, limit)
        if min_fusion_score is not None:
            merged = [x for x in merged if (x.get("_hybrid_fusion") or 0) >= min_fusion_score]
        return {"items": [self._finalize_search_item(m) for m in merged]}

    async def search_memories(
        self,
        namespace: str,
        subject_id: str,
        query: str,
        limit: int = 10,
        fusion: str = "hybrid",
        run_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """向量 / BM25 / 加权融合检索。"""
        result = await self.search_memories_scoped(
            namespace=namespace,
            subject_id=subject_id,
            query=query,
            limit=limit,
            fusion=fusion,
            run_id=run_id,
            filters=filters,
        )
        return result["items"]

    async def get_all_memories_scoped(
        self,
        namespace: str,
        subject_id: str,
        limit: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        memory_client = self._get_memory_client()
        run_id = self._effective_run_id(run_id)
        results = await asyncio.to_thread(
            memory_client.get_all,
            user_id=subject_id,
            agent_id=namespace,
            run_id=run_id,
            limit=limit or 100,
        )
        return {"items": self._normalize_results(results)}

    async def get_all_memories(
        self,
        namespace: str,
        subject_id: str,
        limit: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get all memories for a subject."""
        result = await self.get_all_memories_scoped(
            namespace=namespace,
            subject_id=subject_id,
            limit=limit,
            run_id=run_id,
        )
        return result["items"]

    async def update_memory(
        self,
        namespace: str,
        subject_id: str,
        memory_id: str,
        content: str,
    ) -> Dict[str, Any]:
        """Update an existing memory."""
        memory_client = self._get_memory_client()
        result = await asyncio.to_thread(
            memory_client.update,
            memory_id=memory_id,
            data=content,
        )
        await user_service.touch_subject(namespace, subject_id)
        await self._update_memory_lexical(str(memory_id), content)
        return {
            "id": memory_id,
            "text": content,
            "metadata": None,
            "namespace": namespace,
            "subject_id": subject_id,
            "run_id": None,
            "created_at": None,
            "updated_at": None,
            "message": result.get("message"),
        }

    async def delete_memory(self, namespace: str, subject_id: str, memory_id: str) -> bool:
        """Delete a memory by ID."""
        memory_client = self._get_memory_client()
        await asyncio.to_thread(memory_client.delete, memory_id=memory_id)
        await user_service.touch_subject(namespace, subject_id)
        await self._delete_memory_lexical(str(memory_id))
        return True

    async def add_memories_batch(
        self,
        namespace: str,
        subject_id: str,
        memories: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
        run_id: Optional[str] = None,
        infer: bool = True,
    ) -> Dict[str, Any]:
        """Add multiple memories in batch."""

        async def _add_single(item: Dict[str, Any]):
            item_content = item["content"]
            item_metadata = dict(metadata or {})
            item_metadata.update(item.get("metadata") or {})
            try:
                return await self.add_memory(
                    namespace=namespace,
                    subject_id=subject_id,
                    content=item_content,
                    metadata=item_metadata,
                    run_id=run_id,
                    infer=infer,
                )
            except Exception as exc:
                print(f"Failed to add memory: {exc}")
                return None

        tasks = [_add_single(item) for item in memories]
        results_raw = await asyncio.gather(*tasks)
        results = [r for r in results_raw if r is not None]
        failed = len(memories) - len(results)

        return {
            "added_count": len(results),
            "failed_count": failed,
            "data": results,
        }

    async def add_conversation_memory(
        self,
        namespace: str,
        subject_id: str,
        messages: List[Dict[str, str]],
        metadata: Optional[Dict[str, Any]] = None,
        run_id: Optional[str] = None,
        infer: bool = True,
    ) -> Dict[str, Any]:
        """Extract and store memories from structured conversation messages."""
        memory_client = self._get_memory_client()
        subject_context = await self._get_subject_context(namespace, subject_id)
        result = await asyncio.to_thread(
            memory_client.add,
            messages=self._prepare_conversation_messages(messages, subject_context),
            user_id=subject_id,
            agent_id=namespace,
            run_id=run_id,
            metadata=self._build_metadata(metadata, subject_context),
            infer=infer,
        )
        normalized_results = self._normalize_results(result)
        await self._track_successful_write(namespace, subject_id, len(normalized_results))
        scoped = [
            {**item, "namespace": namespace, "subject_id": subject_id, "run_id": run_id}
            for item in normalized_results
        ]
        await self._persist_memory_lexical_batch(namespace, subject_id, run_id, scoped)
        return {
            "items": normalized_results,
            "count": len(normalized_results),
        }

    async def set_subject_context(
        self,
        namespace: str,
        subject_id: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Upsert subject context without creating a synthetic memory."""
        result = await user_service.upsert_subject_context(
            namespace=namespace,
            subject_id=subject_id,
            name=context.get("name"),
            email=context.get("email"),
            role=context.get("role"),
            preferences=context.get("preferences"),
            custom_data=context.get("custom_data"),
        )
        return result

    async def get_subject_context(self, namespace: str, subject_id: str) -> Dict[str, Any]:
        """Get stored subject context from PostgreSQL."""
        return await self._get_subject_context(namespace, subject_id)

    async def get_subject_stats(self, namespace: str, subject_id: str) -> Dict[str, Any]:
        """Get statistics about a subject's memories."""
        subject_stats = await user_service.get_subject_stats(namespace, subject_id)
        if not subject_stats:
            return {
                "namespace": namespace,
                "subject_id": subject_id,
                "total_memories": 0,
                "collection_name": settings.QDRANT_MEM0_COLLECTION,
            }
        all_memories = await self.get_all_memories(namespace=namespace, subject_id=subject_id)
        return {
            "namespace": namespace,
            "subject_id": subject_id,
            "name": subject_stats.get("name"),
            "email": subject_stats.get("email"),
            "total_memories": len(all_memories),
            "collection_name": settings.QDRANT_MEM0_COLLECTION,
            "created_at": subject_stats.get("created_at"),
            "last_active": subject_stats.get("last_active"),
        }

    def reset_subject(self, namespace: str, subject_id: str) -> bool:
        """Reset operation is not applicable with a global memory client."""
        return True


mem0_service = Mem0Service()

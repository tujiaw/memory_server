import re
import logging
from time import perf_counter
from typing import Any, Dict, List, Optional

from mem0 import Memory
from openai import AsyncOpenAI

from app.core.config import settings
from app.services.user_service import user_service

logger = logging.getLogger(__name__)


class Mem0Service:
    """Service for managing subject-scoped memories on top of mem0."""

    def __init__(self):
        self._memories: Dict[str, Memory] = {}
        self._vector_only_memories: Dict[str, Memory] = {}
        self._openai_client: Optional[AsyncOpenAI] = None

    @staticmethod
    def _scope_key(namespace: str, subject_id: str) -> str:
        return f"{namespace}:{subject_id}"

    @staticmethod
    def _collection_name(namespace: str, subject_id: str) -> str:
        raw_name = f"{namespace}_{subject_id}"
        safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", raw_name)
        return f"mem0_{safe_name}"

    def _get_config(self, namespace: str, subject_id: str) -> dict:
        """Generate mem0 configuration for a specific subject scope."""
        config = {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": self._collection_name(namespace, subject_id),
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
        if settings.NEO4J_URL and settings.NEO4J_USERNAME and settings.NEO4J_PASSWORD:
            config["graph_store"] = {
                "provider": "neo4j",
                "config": {
                    "url": settings.NEO4J_URL,
                    "username": settings.NEO4J_USERNAME,
                    "password": settings.NEO4J_PASSWORD,
                },
            }
            if settings.MEM0_GRAPH_CUSTOM_PROMPT:
                config["graph_store"]["custom_prompt"] = settings.MEM0_GRAPH_CUSTOM_PROMPT
        return config

    def get_memory_client(self, namespace: str, subject_id: str) -> Memory:
        """Get or create a Memory client for a subject scope."""
        scope_key = self._scope_key(namespace, subject_id)
        if scope_key not in self._memories:
            config = self._get_config(namespace, subject_id)
            self._memories[scope_key] = Memory.from_config(config)
        return self._memories[scope_key]

    def get_vector_only_memory_client(self, namespace: str, subject_id: str) -> Memory:
        """Get or create a vector-only Memory client (without graph store)."""
        scope_key = self._scope_key(namespace, subject_id)
        if scope_key not in self._vector_only_memories:
            config = dict(self._get_config(namespace, subject_id))
            config.pop("graph_store", None)
            self._vector_only_memories[scope_key] = Memory.from_config(config)
        return self._vector_only_memories[scope_key]

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
    def _normalize_memory_result(result: Dict[str, Any]) -> Dict[str, Any]:
        metadata = result.get("metadata")
        return {
            "id": result.get("id"),
            "text": result.get("memory") or result.get("text", ""),
            "metadata": metadata,
            "score": result.get("score"),
            "namespace": result.get("agent_id"),
            "subject_id": result.get("user_id"),
            "run_id": result.get("run_id"),
            "created_at": result.get("created_at"),
            "updated_at": result.get("updated_at"),
        }

    def _normalize_results(self, payload: Any) -> List[Dict[str, Any]]:
        if isinstance(payload, dict):
            results = payload.get("results", [])
        else:
            results = payload or []
        return [self._normalize_memory_result(result) for result in results]

    @staticmethod
    def _normalize_relation_item(result: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "source": result.get("source"),
            "relationship": result.get("relationship") or result.get("relatationship"),
            "target": result.get("target") or result.get("destination"),
        }

    @staticmethod
    def _extract_relation_candidates(relations_payload: Any) -> List[Dict[str, Any]]:
        if isinstance(relations_payload, list):
            return [item for item in relations_payload if isinstance(item, dict)]

        if not isinstance(relations_payload, dict):
            return []

        # mem0 add() graph output shape:
        # {"deleted_entities": [...], "added_entities": [[{...}], ...]}
        candidates: List[Dict[str, Any]] = []
        added_entities = relations_payload.get("added_entities", [])
        for entity_group in added_entities:
            if isinstance(entity_group, list):
                candidates.extend(item for item in entity_group if isinstance(item, dict))
            elif isinstance(entity_group, dict):
                candidates.append(entity_group)
        return candidates

    def _normalize_relations(self, payload: Any) -> List[Dict[str, Any]]:
        if not isinstance(payload, dict):
            return []

        relations = self._extract_relation_candidates(payload.get("relations", []))
        normalized_relations: List[Dict[str, Any]] = []
        seen_keys = set()
        for relation in relations:
            normalized_relation = self._normalize_relation_item(relation)
            dedupe_key = (
                normalized_relation.get("source"),
                normalized_relation.get("relationship"),
                normalized_relation.get("target"),
            )
            if dedupe_key in seen_keys:
                continue
            seen_keys.add(dedupe_key)
            normalized_relations.append(normalized_relation)
        return normalized_relations

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
    def _sort_memories_by_recency(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        def sort_key(item: Dict[str, Any]) -> str:
            return str(item.get("updated_at") or item.get("created_at") or "")

        return sorted(items, key=sort_key, reverse=True)

    @staticmethod
    def _memory_dedupe_key(item: Dict[str, Any]) -> tuple:
        return (item.get("id"), (item.get("text") or "").strip())

    def _merge_context_sources(
        self,
        relevant_memories: List[Dict[str, Any]],
        recent_memories: List[Dict[str, Any]],
    ) -> Dict[str, List[Dict[str, Any]]]:
        deduplicated_relevant = self._deduplicate_memory_items(relevant_memories)
        deduplicated_recent = self._deduplicate_memory_items(recent_memories)
        relevant_keys = {
            self._memory_dedupe_key(item)
            for item in deduplicated_relevant
        }
        recent_only = [
            item
            for item in deduplicated_recent
            if self._memory_dedupe_key(item) not in relevant_keys
        ]
        return {
            "relevant": deduplicated_relevant,
            "recent": recent_only,
        }

    @staticmethod
    def _filter_relevant_memories_by_score(
        relevant_memories: List[Dict[str, Any]],
        min_score: float,
        fallback_count: int = 0,
    ) -> List[Dict[str, Any]]:
        deduplicated_relevant = Mem0Service._deduplicate_memory_items(relevant_memories)
        filtered_items = [
            item
            for item in deduplicated_relevant
            if (item.get("score") or 0) >= min_score
        ]
        if filtered_items:
            return filtered_items
        return deduplicated_relevant[:fallback_count]

    @staticmethod
    def _build_memory_section(title: str, items: List[Dict[str, Any]]) -> Optional[str]:
        lines = [f"{index + 1}. {item['text']}" for index, item in enumerate(items) if item.get("text")]
        if not lines:
            return None
        return title + "\n\n" + "\n".join(lines)

    @staticmethod
    def _build_relations_section(relations: List[Dict[str, Any]]) -> Optional[str]:
        lines: List[str] = []
        for index, relation in enumerate(relations):
            source = relation.get("source")
            relationship = relation.get("relationship")
            target = relation.get("target")
            if not source or not relationship or not target:
                continue
            lines.append(f"{index + 1}. {source} --{relationship}--> {target}")
        if not lines:
            return None
        return "以下是可用于推理的关系图谱（三元组）：\n\n" + "\n".join(lines)

    @staticmethod
    def _build_context_usage_guideline() -> str:
        return (
            "回答规则：\n"
            "1. 优先使用关系图谱中的结构化事实。\n"
            "2. 若与记忆文本存在冲突，优先采用更近期、更具体的信息。\n"
            "3. 若证据不足，明确说明不确定并向用户追问。\n"
            "4. 不要编造上下文中不存在的事实。"
        )

    def _build_context_text(
        self,
        query: Optional[str],
        relevant_items: List[Dict[str, Any]],
        recent_items: List[Dict[str, Any]],
        relations: List[Dict[str, Any]],
    ) -> str:
        all_items = relevant_items + recent_items
        if not all_items and not relations:
            return "当前没有可用记忆。请仅基于当前用户输入进行回答。"

        relation_section = self._build_relations_section(relations)
        guidance_section = self._build_context_usage_guideline()

        if not query or not query.strip():
            sections: List[str] = []
            recent_section = self._build_memory_section("以下是该用户的可用记忆：", recent_items)
            if recent_section:
                sections.append(recent_section)
            if relation_section:
                sections.append(relation_section)
            if not sections:
                return "当前没有可用记忆。请仅基于当前用户输入进行回答。"
            sections.append(guidance_section)
            return "\n\n".join(sections)

        sections: List[str] = []
        relevant_section = self._build_memory_section("以下是与当前问题最相关的用户记忆：", relevant_items)
        if relevant_section:
            sections.append(relevant_section)

        recent_section = self._build_memory_section("以下是最近记忆补充：", recent_items)
        if recent_section:
            sections.append(recent_section)

        if relation_section:
            sections.append(relation_section)

        if not sections:
            return "当前没有可用记忆。请仅基于当前用户输入进行回答。"
        sections.append(guidance_section)
        return "\n\n".join(sections)

    @staticmethod
    def _select_history_for_query_enhancement(
        recent_memories: List[Dict[str, Any]],
        max_items: int = 3,
    ) -> List[str]:
        history_items: List[str] = []

        for item in recent_memories:
            text = (item.get("text") or "").strip()
            if not text:
                continue

            history_items.append(text)
            if len(history_items) >= max_items:
                break

        return history_items

    def _get_openai_client(self) -> AsyncOpenAI:
        if self._openai_client is None:
            self._openai_client = AsyncOpenAI(
                api_key=settings.OPENAI_API_KEY,
                base_url=settings.OPENAI_BASE_URL,
            )
        return self._openai_client

    @staticmethod
    def _build_query_rewrite_prompt(query: str, history_items: List[str]) -> str:
        history_block = "\n".join(f"- {item}" for item in history_items)
        return (
            "你是一个检索查询改写助手。"
            "请结合用户原始 query 和最近历史，生成一个更适合 memory 检索的增强 query。"
            "只返回一行增强后的 query，不要解释，不要返回 JSON。\n\n"
            f"原始 query:\n{query}\n\n"
            f"最近历史:\n{history_block}"
        )

    async def _rewrite_query_with_llm(self, query: str, history_items: List[str]) -> str:
        if not history_items:
            return query.strip()

        client = self._get_openai_client()
        response = await client.chat.completions.create(
            model=settings.OPENAI_MODEL,
            temperature=0,
            messages=[
                {
                    "role": "system",
                    "content": "你负责把用户问题改写成更适合语义检索的简洁查询。",
                },
                {
                    "role": "user",
                    "content": self._build_query_rewrite_prompt(query.strip(), history_items),
                },
            ],
        )
        message = response.choices[0].message.content if response.choices else None
        return (message or "").strip()

    # ========================================================================
    # Basic Memory Operations
    # ========================================================================

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
        memory_client = self.get_memory_client(namespace, subject_id)
        subject_context = await self._get_subject_context(namespace, subject_id)
        result = memory_client.add(
            messages=self._prepare_messages(content, subject_context),
            user_id=subject_id,
            agent_id=namespace,
            run_id=run_id,
            metadata=self._build_metadata(metadata, subject_context),
            infer=infer,
        )
        normalized_results = self._normalize_results(result)
        normalized_relations = self._normalize_relations(result)
        await self._track_successful_write(namespace, subject_id, len(normalized_results))
        if normalized_results:
            return {
                **normalized_results[0],
                "relations": normalized_relations,
            }
        return {
            "text": content,
            "metadata": metadata or {},
            "namespace": namespace,
            "subject_id": subject_id,
            "run_id": run_id,
            "relations": normalized_relations,
        }

    async def search_memories_with_relations(
        self,
        namespace: str,
        subject_id: str,
        query: str,
        limit: int = 5,
        run_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        memory_client = self.get_memory_client(namespace, subject_id)
        results = memory_client.search(
            query=query,
            user_id=subject_id,
            agent_id=namespace,
            run_id=run_id,
            limit=limit,
            filters=filters,
        )
        await user_service.touch_subject(namespace, subject_id)
        return {
            "items": self._normalize_results(results),
            "relations": self._normalize_relations(results),
        }

    async def search_memories(
        self,
        namespace: str,
        subject_id: str,
        query: str,
        limit: int = 5,
        run_id: Optional[str] = None,
        filters: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Search memories using semantic similarity."""
        result = await self.search_memories_with_relations(
            namespace=namespace,
            subject_id=subject_id,
            query=query,
            limit=limit,
            run_id=run_id,
            filters=filters,
        )
        return result["items"]

    async def get_all_memories_with_relations(
        self,
        namespace: str,
        subject_id: str,
        limit: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        memory_client = self.get_memory_client(namespace, subject_id)
        results = memory_client.get_all(
            user_id=subject_id,
            agent_id=namespace,
            run_id=run_id,
            limit=limit or 100,
        )
        return {
            "items": self._normalize_results(results),
            "relations": self._normalize_relations(results),
        }

    async def get_all_memories(
        self,
        namespace: str,
        subject_id: str,
        limit: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get all memories for a subject."""
        result = await self.get_all_memories_with_relations(
            namespace=namespace,
            subject_id=subject_id,
            limit=limit,
            run_id=run_id,
        )
        return result["items"]

    async def _search_memories_vector_only(
        self,
        namespace: str,
        subject_id: str,
        query: str,
        limit: int,
        run_id: Optional[str],
    ) -> Dict[str, Any]:
        memory_client = self.get_vector_only_memory_client(namespace, subject_id)
        results = memory_client.search(
            query=query,
            user_id=subject_id,
            agent_id=namespace,
            run_id=run_id,
            limit=limit,
        )
        await user_service.touch_subject(namespace, subject_id)
        return {
            "items": self._normalize_results(results),
            "relations": [],
        }

    async def _get_all_memories_vector_only(
        self,
        namespace: str,
        subject_id: str,
        limit: int,
        run_id: Optional[str],
    ) -> Dict[str, Any]:
        memory_client = self.get_vector_only_memory_client(namespace, subject_id)
        results = memory_client.get_all(
            user_id=subject_id,
            agent_id=namespace,
            run_id=run_id,
            limit=limit,
        )
        return {
            "items": self._normalize_results(results),
            "relations": [],
        }

    async def get_context_for_llm(
        self,
        namespace: str,
        subject_id: str,
        query: Optional[str] = None,
        limit: int = 15,
        min_score: float = 0.5,
        run_id: Optional[str] = None,
        enable_query_rewrite: bool = True,
        enable_graph_search: bool = True,
    ) -> Dict[str, Any]:
        """获取适合直接注入 LLM 的 RAG 风格上下文。"""
        total_start = perf_counter()
        normalized_query = query.strip() if query and query.strip() else None
        enhanced_query: Optional[str] = None
        history_used: List[str] = []
        task_relevant_memories: List[Dict[str, Any]] = []
        relevant_relations: List[Dict[str, Any]] = []

        recent_fetch_start = perf_counter()
        if enable_graph_search:
            recent_memory_result = await self.get_all_memories_with_relations(
                namespace=namespace,
                subject_id=subject_id,
                limit=limit,
                run_id=run_id,
            )
        else:
            recent_memory_result = await self._get_all_memories_vector_only(
                namespace=namespace,
                subject_id=subject_id,
                limit=limit,
                run_id=run_id,
            )
        recent_fetch_ms = (perf_counter() - recent_fetch_start) * 1000
        recent_memories = recent_memory_result["items"]
        recent_memories = self._sort_memories_by_recency(self._deduplicate_memory_items(recent_memories))
        recent_relations = recent_memory_result["relations"]
        rewrite_ms = 0.0
        relevant_fetch_ms = 0.0

        if normalized_query:
            candidate_history = self._select_history_for_query_enhancement(recent_memories)
            rewrite_start = perf_counter()
            try:
                if enable_query_rewrite:
                    rewritten_query = await self._rewrite_query_with_llm(normalized_query, candidate_history)
                    enhanced_query = rewritten_query.strip() or normalized_query
                    if enhanced_query != normalized_query:
                        history_used = candidate_history
                else:
                    enhanced_query = normalized_query
                    history_used = []
            except Exception:
                enhanced_query = normalized_query
                history_used = []
            rewrite_ms = (perf_counter() - rewrite_start) * 1000

            relevant_fetch_start = perf_counter()
            if enable_graph_search:
                relevant_memory_result = await self.search_memories_with_relations(
                    namespace=namespace,
                    subject_id=subject_id,
                    query=enhanced_query,
                    limit=limit,
                    run_id=run_id,
                )
            else:
                relevant_memory_result = await self._search_memories_vector_only(
                    namespace=namespace,
                    subject_id=subject_id,
                    query=enhanced_query,
                    limit=limit,
                    run_id=run_id,
                )
            relevant_fetch_ms = (perf_counter() - relevant_fetch_start) * 1000
            task_relevant_memories = relevant_memory_result["items"]
            task_relevant_memories = self._filter_relevant_memories_by_score(
                task_relevant_memories,
                min_score=min_score,
            )
            relevant_relations = relevant_memory_result["relations"]

        merge_start = perf_counter()
        grouped_sources = self._merge_context_sources(task_relevant_memories, recent_memories)
        merged_items = grouped_sources["relevant"] + grouped_sources["recent"]
        merged_relations = self._normalize_relations({"relations": relevant_relations + recent_relations})
        merge_ms = (perf_counter() - merge_start) * 1000
        result: Dict[str, Any] = {
            "context": self._build_context_text(
                query=normalized_query,
                relevant_items=grouped_sources["relevant"],
                recent_items=grouped_sources["recent"],
                relations=merged_relations,
            ),
            "count": len(merged_items),
            "query": normalized_query,
            "enhanced_query": enhanced_query,
            "history_used": history_used,
            "sources": merged_items,
            "relations": merged_relations,
        }
        total_ms = (perf_counter() - total_start) * 1000
        logger.info(
            (
                "get_context_for_llm timing | namespace=%s subject_id=%s has_query=%s "
                "enable_query_rewrite=%s enable_graph_search=%s "
                "recent_fetch_ms=%.2f rewrite_ms=%.2f relevant_fetch_ms=%.2f merge_ms=%.2f total_ms=%.2f "
                "sources=%d relations=%d"
            ),
            namespace,
            subject_id,
            bool(normalized_query),
            enable_query_rewrite,
            enable_graph_search,
            recent_fetch_ms,
            rewrite_ms,
            relevant_fetch_ms,
            merge_ms,
            total_ms,
            len(merged_items),
            len(merged_relations),
        )
        return result

    async def update_memory(
        self,
        namespace: str,
        subject_id: str,
        memory_id: str,
        content: str,
    ) -> Dict[str, Any]:
        """Update an existing memory."""
        memory_client = self.get_memory_client(namespace, subject_id)
        result = memory_client.update(memory_id=memory_id, data=content)
        await user_service.touch_subject(namespace, subject_id)
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
        memory_client = self.get_memory_client(namespace, subject_id)
        memory_client.delete(memory_id=memory_id)
        await user_service.touch_subject(namespace, subject_id)
        return True

    # ========================================================================
    # Batch Operations
    # ========================================================================

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
        subject_context = await self._get_subject_context(namespace, subject_id)
        results = []
        failed = 0
        for item in memories:
            item_content = item["content"]
            item_metadata = dict(metadata or {})
            item_metadata.update(item.get("metadata") or {})
            try:
                result = await self.add_memory(
                    namespace=namespace,
                    subject_id=subject_id,
                    content=item_content,
                    metadata=self._build_metadata(item_metadata, subject_context),
                    run_id=run_id,
                    infer=infer,
                )
                results.append(result)
            except Exception as exc:
                print(f"Failed to add memory: {exc}")
                failed += 1

        return {
            "added_count": len(results),
            "failed_count": failed,
            "data": results,
        }

    # ========================================================================
    # Conversation Memory
    # ========================================================================

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
        memory_client = self.get_memory_client(namespace, subject_id)
        subject_context = await self._get_subject_context(namespace, subject_id)
        result = memory_client.add(
            messages=self._prepare_conversation_messages(messages, subject_context),
            user_id=subject_id,
            agent_id=namespace,
            run_id=run_id,
            metadata=self._build_metadata(metadata, subject_context),
            infer=infer,
        )
        normalized_results = self._normalize_results(result)
        normalized_relations = self._normalize_relations(result)
        await self._track_successful_write(namespace, subject_id, len(normalized_results))
        return {
            "items": normalized_results,
            "count": len(normalized_results),
            "relations": normalized_relations,
        }

    # ========================================================================
    # User Context Management
    # ========================================================================

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
        """Get stored subject context from MongoDB."""
        return await self._get_subject_context(namespace, subject_id)

    # ========================================================================
    # Statistics and Utilities
    # ========================================================================

    async def get_subject_stats(self, namespace: str, subject_id: str) -> Dict[str, Any]:
        """Get statistics about a subject's memories."""
        subject_stats = await user_service.get_subject_stats(namespace, subject_id)
        if not subject_stats:
            return {
                "namespace": namespace,
                "subject_id": subject_id,
                "total_memories": 0,
                "collection_name": self._collection_name(namespace, subject_id),
            }
        all_memories = await self.get_all_memories(namespace=namespace, subject_id=subject_id)
        return {
            "namespace": namespace,
            "subject_id": subject_id,
            "name": subject_stats.get("name"),
            "email": subject_stats.get("email"),
            "total_memories": len(all_memories),
            "collection_name": self._collection_name(namespace, subject_id),
            "created_at": subject_stats.get("created_at"),
            "last_active": subject_stats.get("last_active"),
        }

    def reset_subject(self, namespace: str, subject_id: str) -> bool:
        """Remove a subject memory client from the local cache."""
        scope_key = self._scope_key(namespace, subject_id)
        if scope_key in self._memories:
            del self._memories[scope_key]
        if scope_key in self._vector_only_memories:
            del self._vector_only_memories[scope_key]
        return True


# Global instance
mem0_service = Mem0Service()

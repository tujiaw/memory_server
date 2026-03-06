import re
from typing import Any, Dict, List, Optional

from mem0 import Memory

from app.core.config import settings
from app.services.user_service import user_service


class Mem0Service:
    """Service for managing subject-scoped memories on top of mem0."""

    def __init__(self):
        self._memories: Dict[str, Memory] = {}

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
        return {
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

    def get_memory_client(self, namespace: str, subject_id: str) -> Memory:
        """Get or create a Memory client for a subject scope."""
        scope_key = self._scope_key(namespace, subject_id)
        if scope_key not in self._memories:
            config = self._get_config(namespace, subject_id)
            self._memories[scope_key] = Memory.from_config(config)
        return self._memories[scope_key]

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
        await self._track_successful_write(namespace, subject_id, len(normalized_results))
        if normalized_results:
            return normalized_results[0]
        return {
            "text": content,
            "metadata": metadata or {},
            "namespace": namespace,
            "subject_id": subject_id,
            "run_id": run_id,
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
        return self._normalize_results(results)

    async def get_all_memories(
        self,
        namespace: str,
        subject_id: str,
        limit: Optional[int] = None,
        run_id: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Get all memories for a subject."""
        memory_client = self.get_memory_client(namespace, subject_id)
        results = memory_client.get_all(
            user_id=subject_id,
            agent_id=namespace,
            run_id=run_id,
            limit=limit or 100,
        )
        return self._normalize_results(results)

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
        await self._track_successful_write(namespace, subject_id, len(normalized_results))
        return {
            "items": normalized_results,
            "count": len(normalized_results),
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
        return True


# Global instance
mem0_service = Mem0Service()

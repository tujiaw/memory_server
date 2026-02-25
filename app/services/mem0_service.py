from typing import Optional, Dict, Any, List
from mem0 import Memory
from app.core.config import settings
from app.services.user_service import user_service


class Mem0Service:
    """
    Service for managing user memories using mem0 with Qdrant backend.

    Each user gets their own isolated collection in Qdrant (mem0_{user_id}).
    User profiles are managed in MongoDB.
    """

    def __init__(self):
        self._memories: Dict[str, Memory] = {}

    def _get_config(self, user_id: str) -> dict:
        """Generate mem0 configuration for a specific user."""
        return {
            "vector_store": {
                "provider": "qdrant",
                "config": {
                    "collection_name": f"mem0_{user_id}",
                    "host": settings.QDRANT_HOST,
                    "port": settings.QDRANT_PORT,
                    "embedding_model_dims": 1536,
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

    def get_memory_client(self, user_id: str) -> Memory:
        """Get or create a Memory client for a specific user."""
        if user_id not in self._memories:
            config = self._get_config(user_id)
            self._memories[user_id] = Memory.from_config(config=config)
        return self._memories[user_id]

    async def _get_user_info(self, user_id: str) -> Dict[str, Any]:
        """Get user info from MongoDB for personalized memory extraction."""
        user = await user_service.get_user(user_id)

        if user:
            return {
                "name": user.get("name"),
                "role": user.get("role"),
                "preferences": user.get("preferences", {}),
            }

        return {}

    # ========================================================================
    # Basic Memory Operations
    # ========================================================================

    async def add_memory(
        self,
        user_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        user_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Add a new memory for a user."""
        memory_client = self.get_memory_client(user_id)

        # Merge user info from MongoDB if not provided
        if not user_info:
            user_info = await self._get_user_info(user_id)

        result = memory_client.add(
            content=content,
            metadata=metadata or {},
            user_info=user_info or {},
        )

        # Update user stats
        await user_service.update_last_active(user_id)
        await user_service.increment_memory_count(user_id)

        return result

    async def search_memories(
        self,
        user_id: str,
        query: str,
        limit: int = 5,
    ) -> List[Dict[str, Any]]:
        """Search memories using semantic similarity."""
        memory_client = self.get_memory_client(user_id)

        results = memory_client.search(
            query=query,
            limit=limit,
        )

        await user_service.update_last_active(user_id)

        return results

    async def get_all_memories(
        self,
        user_id: str,
        limit: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        """Get all memories for a user."""
        memory_client = self.get_memory_client(user_id)

        results = memory_client.get_all()

        if limit:
            return results[:limit]

        return results

    async def update_memory(
        self,
        user_id: str,
        memory_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Update an existing memory."""
        memory_client = self.get_memory_client(user_id)

        result = memory_client.update(
            memory_id=memory_id,
            content=content,
            metadata=metadata,
        )

        await user_service.update_last_active(user_id)

        return result

    async def delete_memory(self, user_id: str, memory_id: str) -> bool:
        """Delete a memory by ID."""
        memory_client = self.get_memory_client(user_id)
        memory_client.delete(memory_id=memory_id)

        await user_service.update_last_active(user_id)

        return True

    # ========================================================================
    # Batch Operations
    # ========================================================================

    async def add_memories_batch(
        self,
        user_id: str,
        memories: List[str],
        metadata: Optional[Dict[str, Any]] = None,
        user_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Add multiple memories in batch."""
        memory_client = self.get_memory_client(user_id)

        # Merge user info from MongoDB if not provided
        if not user_info:
            user_info = await self._get_user_info(user_id)

        results = []
        failed = 0

        for content in memories:
            try:
                result = memory_client.add(
                    content=content,
                    metadata=metadata or {},
                    user_info=user_info or {},
                )
                results.append(result)
            except Exception as e:
                failed += 1

        # Update user stats
        await user_service.update_last_active(user_id)

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
        user_id: str,
        messages: List[Dict[str, str]],
        user_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Extract and store memories from a conversation.

        mem0 will analyze the conversation and extract important facts/preferences.
        """
        memory_client = self.get_memory_client(user_id)

        # Merge user info from MongoDB if not provided
        if not user_info:
            user_info = await self._get_user_info(user_id)

        # Convert messages to mem0 format
        conversation_text = "\n".join([
            f"{msg['role']}: {msg['content']}"
            for msg in messages
        ])

        result = memory_client.add(
            content=conversation_text,
            metadata={"type": "conversation"},
            user_info=user_info or {},
        )

        # Update user stats
        await user_service.update_last_active(user_id)
        await user_service.increment_memory_count(user_id)

        return result

    # ========================================================================
    # User Context Management
    # ========================================================================

    async def set_user_context(
        self,
        user_id: str,
        context: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Store user profile/context as a special memory.

        This helps mem0 provide personalized responses.
        Also updates the user in MongoDB.
        """
        # Update user in MongoDB
        await user_service.update_user(
            user_id=user_id,
            name=context.get("name"),
            email=context.get("email"),
            role=context.get("role"),
            preferences=context.get("preferences"),
            custom_data=context.get("custom_data"),
        )

        # Store as a memory for mem0 context
        memory_client = self.get_memory_client(user_id)

        user_info = await self._get_user_info(user_id)

        context_text = f"User profile: {context}"

        result = memory_client.add(
            content=context_text,
            metadata={
                "type": "user_context",
                "context": context,
            },
            user_info=user_info,
        )

        return {"success": True, "data": result}

    async def get_user_context(self, user_id: str) -> Dict[str, Any]:
        """Get stored user context from MongoDB."""
        user = await user_service.get_user(user_id)

        if user:
            return {
                "user_id": user["id"],
                "name": user.get("name"),
                "email": user.get("email"),
                "role": user.get("role"),
                "preferences": user.get("preferences", {}),
                "custom_data": user.get("custom_data", {}),
            }

        return {}

    # ========================================================================
    # Statistics and Utilities
    # ========================================================================

    async def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        """Get statistics about a user's memories."""
        user_stats = await user_service.get_user_stats(user_id)

        if not user_stats:
            return {
                "user_id": user_id,
                "total_memories": 0,
                "collection_name": f"mem0_{user_id}",
            }

        all_memories = await self.get_all_memories(user_id)

        return {
            "user_id": user_id,
            "name": user_stats.get("name"),
            "email": user_stats.get("email"),
            "total_memories": len(all_memories),
            "collection_name": f"mem0_{user_id}",
            "created_at": user_stats.get("created_at"),
            "last_active": user_stats.get("last_active"),
        }

    def reset_user(self, user_id: str) -> bool:
        """Remove a user's memory client (useful for testing)."""
        if user_id in self._memories:
            del self._memories[user_id]
        return True


# Global instance
mem0_service = Mem0Service()

from fastapi import APIRouter, HTTPException, status
from typing import Optional

from app.models.schemas import (
    MemoryAdd,
    MemorySearch,
    MemoryUpdate,
    MemoryDelete,
    MemoryBatchAdd,
    ConversationMemory,
    UserContext,
    MemoryResponse,
    MemoryListResponse,
    BatchMemoryResponse,
)
from app.services.mem0_service import mem0_service

router = APIRouter(prefix="/memories", tags=["Memories"])


# ============================================================================
# Basic Memory Operations
# ============================================================================

@router.post("", response_model=MemoryResponse, status_code=status.HTTP_201_CREATED)
async def add_memory(request: MemoryAdd):
    """
    Add a new memory.

    The content will be processed, embedded, and stored in Qdrant.
    mem0 will intelligently extract facts and preferences from the content.
    """
    try:
        result = await mem0_service.add_memory(
            user_id=request.user_id,
            content=request.content,
            metadata=request.metadata,
            user_info=request.user_info,
        )
        return MemoryResponse(success=True, message="Memory added successfully", data=result)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add memory: {str(e)}",
        )


@router.post("/search", response_model=MemoryListResponse)
async def search_memories(request: MemorySearch):
    """
    Search memories using semantic similarity.

    Returns the most relevant memories based on the query.
    """
    try:
        results = await mem0_service.search_memories(
            user_id=request.user_id,
            query=request.query,
            limit=request.limit,
        )
        return MemoryListResponse(success=True, data=results, count=len(results))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}",
        )


@router.get("/{user_id}", response_model=MemoryListResponse)
async def get_all_memories(user_id: str, limit: Optional[int] = None):
    """
    Get all memories for a user.

    Args:
        user_id: User identifier
        limit: Optional limit on number of results
    """
    try:
        results = await mem0_service.get_all_memories(user_id=user_id, limit=limit)
        return MemoryListResponse(success=True, data=results, count=len(results))
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve memories: {str(e)}",
        )


@router.put("", response_model=MemoryResponse)
async def update_memory(request: MemoryUpdate):
    """
    Update an existing memory.

    At least one of content or metadata must be provided.
    """
    if request.content is None and request.metadata is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="At least one of content or metadata must be provided",
        )

    try:
        result = await mem0_service.update_memory(
            user_id=request.user_id,
            memory_id=request.memory_id,
            content=request.content,
            metadata=request.metadata,
        )
        return MemoryResponse(success=True, message="Memory updated successfully", data=result)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update memory: {str(e)}",
        )


@router.delete("", response_model=MemoryResponse)
async def delete_memory(request: MemoryDelete):
    """
    Delete a memory by ID.
    """
    try:
        await mem0_service.delete_memory(
            user_id=request.user_id,
            memory_id=request.memory_id,
        )
        return MemoryResponse(success=True, message="Memory deleted successfully")
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete memory: {str(e)}",
        )


# ============================================================================
# Batch Operations
# ============================================================================

@router.post("/batch", response_model=BatchMemoryResponse, status_code=status.HTTP_201_CREATED)
async def add_memories_batch(request: MemoryBatchAdd):
    """
    Add multiple memories in batch.

    Useful for bulk importing memories or processing multiple facts at once.
    """
    try:
        result = await mem0_service.add_memories_batch(
            user_id=request.user_id,
            memories=request.memories,
            metadata=request.metadata,
            user_info=request.user_info,
        )
        return BatchMemoryResponse(**result)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch operation failed: {str(e)}",
        )


# ============================================================================
# Conversation Memory
# ============================================================================

@router.post("/conversation", response_model=MemoryResponse, status_code=status.HTTP_201_CREATED)
async def add_conversation_memory(request: ConversationMemory):
    """
    Extract and store memories from a conversation.

    mem0 will analyze the conversation and automatically extract
    important facts, preferences, and context.
    """
    try:
        messages = [msg.model_dump() for msg in request.messages]
        result = await mem0_service.add_conversation_memory(
            user_id=request.user_id,
            messages=messages,
            user_info=request.user_info,
        )
        return MemoryResponse(success=True, message="Conversation memories added", data=result)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process conversation: {str(e)}",
        )


# ============================================================================
# User Context Management
# ============================================================================

@router.post("/context", response_model=MemoryResponse)
async def set_user_context(request: UserContext):
    """
    Set user profile and context information.

    This context helps mem0 provide more personalized memory extraction and retrieval.
    """
    try:
        result = await mem0_service.set_user_context(
            user_id=request.user_id,
            context=request.model_dump(),
        )
        return MemoryResponse(success=True, message="User context updated", data=result)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to set user context: {str(e)}",
        )


@router.get("/context/{user_id}", response_model=MemoryResponse)
async def get_user_context(user_id: str):
    """Get stored user context."""
    try:
        context = await mem0_service.get_user_context(user_id)
        return MemoryResponse(success=True, data=context)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get user context: {str(e)}",
        )


# ============================================================================
# Statistics
# ============================================================================

@router.get("/stats/{user_id}", response_model=MemoryResponse)
async def get_user_stats(user_id: str):
    """
    Get statistics about a user's memories.

    Returns total memory count and collection information.
    """
    try:
        stats = await mem0_service.get_user_stats(user_id)
        return MemoryResponse(success=True, data=stats)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get stats: {str(e)}",
        )

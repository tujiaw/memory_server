from typing import Annotated, Optional

from fastapi import APIRouter, Depends, HTTPException, status

from app.models.schemas import (
    MemoryAdd,
    ConversationMemory,
    BatchMemoryResponse,
    MemoryDelete,
    MemoryListResponse,
    MemoryResponse,
    MemorySearch,
    MemoryUpdate,
    MemoryBatchAdd,
    AuthContext,
)
from app.core.deps import authorize_namespace, get_auth_context
from app.services.mem0_service import mem0_service

router = APIRouter(prefix="/memories", tags=["Memories"])

@router.post("", response_model=MemoryResponse, status_code=status.HTTP_201_CREATED)
async def add_memory(
    request: MemoryAdd,
    auth_context: Annotated[AuthContext, Depends(get_auth_context)],
):
    """Add a new memory for a subject inside a namespace."""
    try:
        authorize_namespace(auth_context, request.namespace)
        result = await mem0_service.add_memory(
            namespace=request.namespace,
            subject_id=request.subject_id,
            content=request.content,
            metadata=request.metadata,
            run_id=request.run_id,
            infer=request.infer,
        )
        return MemoryResponse(success=True, message="Memory added successfully", data=result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to add memory: {str(e)}",
        )


@router.post("/search", response_model=MemoryListResponse)
async def search_memories(
    request: MemorySearch,
    auth_context: Annotated[AuthContext, Depends(get_auth_context)],
):
    """Search memories by namespace, subject, and optional run."""
    try:
        authorize_namespace(auth_context, request.namespace)
        result = await mem0_service.search_memories_scoped(
            namespace=request.namespace,
            subject_id=request.subject_id,
            query=request.query,
            limit=request.limit,
            mode=request.mode,
            run_id=request.run_id,
            filters=request.filters,
        )
        return MemoryListResponse(
            success=True,
            data=result["items"],
            count=len(result["items"]),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Search failed: {str(e)}",
        )


@router.get("/{namespace}/{subject_id}", response_model=MemoryListResponse)
async def get_all_memories(
    namespace: str,
    subject_id: str,
    auth_context: Annotated[AuthContext, Depends(get_auth_context)],
    limit: Optional[int] = None,
    run_id: Optional[str] = None,
):
    """Get all memories for a subject."""
    try:
        authorize_namespace(auth_context, namespace)
        result = await mem0_service.get_all_memories_scoped(
            namespace=namespace,
            subject_id=subject_id,
            limit=limit,
            run_id=run_id,
        )
        return MemoryListResponse(
            success=True,
            data=result["items"],
            count=len(result["items"]),
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to retrieve memories: {str(e)}",
        )


@router.put("", response_model=MemoryResponse)
async def update_memory(
    request: MemoryUpdate,
    auth_context: Annotated[AuthContext, Depends(get_auth_context)],
):
    """Update an existing memory's content."""
    try:
        authorize_namespace(auth_context, request.namespace)
        result = await mem0_service.update_memory(
            namespace=request.namespace,
            subject_id=request.subject_id,
            memory_id=request.memory_id,
            content=request.content,
        )
        return MemoryResponse(success=True, message="Memory updated successfully", data=result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update memory: {str(e)}",
        )


@router.delete("", response_model=MemoryResponse)
async def delete_memory(
    request: MemoryDelete,
    auth_context: Annotated[AuthContext, Depends(get_auth_context)],
):
    """Delete a memory by ID."""
    try:
        authorize_namespace(auth_context, request.namespace)
        await mem0_service.delete_memory(
            namespace=request.namespace,
            subject_id=request.subject_id,
            memory_id=request.memory_id,
        )
        return MemoryResponse(success=True, message="Memory deleted successfully")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete memory: {str(e)}",
        )

@router.post("/batch", response_model=BatchMemoryResponse, status_code=status.HTTP_201_CREATED)
async def add_memories_batch(
    request: MemoryBatchAdd,
    auth_context: Annotated[AuthContext, Depends(get_auth_context)],
):
    """Add multiple subject memories in a single request."""
    try:
        authorize_namespace(auth_context, request.namespace)
        result = await mem0_service.add_memories_batch(
            namespace=request.namespace,
            subject_id=request.subject_id,
            memories=[item.model_dump() for item in request.memories],
            metadata=request.metadata,
            run_id=request.run_id,
            infer=request.infer,
        )
        return BatchMemoryResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch operation failed: {str(e)}",
        )

@router.post("/conversation", response_model=MemoryResponse, status_code=status.HTTP_201_CREATED)
async def add_conversation_memory(
    request: ConversationMemory,
    auth_context: Annotated[AuthContext, Depends(get_auth_context)],
):
    """Extract memories from structured conversation messages."""
    try:
        authorize_namespace(auth_context, request.namespace)
        result = await mem0_service.add_conversation_memory(
            namespace=request.namespace,
            subject_id=request.subject_id,
            messages=[msg.model_dump() for msg in request.messages],
            metadata=request.metadata,
            run_id=request.run_id,
            infer=request.infer,
        )
        return MemoryResponse(success=True, message="Conversation memories added", data=result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process conversation: {str(e)}",
        )

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.core.deps import authorize_namespace, get_auth_context
from app.models.schemas import AuthContext, MemoryResponse, SubjectContextInput
from app.services.mem0_service import mem0_service

router = APIRouter(prefix="/subjects", tags=["Subjects"])


@router.put("/{subject_id}/context", response_model=MemoryResponse)
async def set_subject_context(
    subject_id: str,
    request: SubjectContextInput,
    auth_context: Annotated[AuthContext, Depends(get_auth_context)],
):
    """Upsert subject context used to improve memory quality."""
    try:
        authorize_namespace(auth_context, request.namespace)
        context = await mem0_service.set_subject_context(
            namespace=request.namespace,
            subject_id=subject_id,
            context=request.model_dump(exclude={"namespace"}),
        )
        return MemoryResponse(success=True, message="Subject context updated", data=context)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to set subject context: {str(exc)}",
        )


@router.get("/{subject_id}/context", response_model=MemoryResponse)
async def get_subject_context(
    subject_id: str,
    namespace: str,
    auth_context: Annotated[AuthContext, Depends(get_auth_context)],
):
    """Read subject context by namespace and subject ID."""
    try:
        authorize_namespace(auth_context, namespace)
        context = await mem0_service.get_subject_context(namespace=namespace, subject_id=subject_id)
        if not context:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Subject '{subject_id}' not found in namespace '{namespace}'",
            )
        return MemoryResponse(success=True, data=context)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get subject context: {str(exc)}",
        )


@router.get("/{subject_id}/stats", response_model=MemoryResponse)
async def get_subject_stats(
    subject_id: str,
    namespace: str,
    auth_context: Annotated[AuthContext, Depends(get_auth_context)],
):
    """Get memory statistics for a subject."""
    try:
        authorize_namespace(auth_context, namespace)
        stats = await mem0_service.get_subject_stats(namespace=namespace, subject_id=subject_id)
        return MemoryResponse(success=True, data=stats)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get subject stats: {str(exc)}",
        )

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Response, status

from app.core.deps import require_admin_token
from app.models.schemas import (
    AdminClientCreateRequest,
    AdminClientCreateResponse,
    AdminClientListResponse,
    AdminClientResetSecretResponse,
    AdminClientSummary,
    AdminClientUpdateRequest,
)
from app.services.auth_service import auth_service

router = APIRouter(
    prefix="/admin",
    tags=["Admin"],
    dependencies=[Depends(require_admin_token)],
)


@router.post("/clients", response_model=AdminClientCreateResponse, status_code=status.HTTP_201_CREATED)
async def create_service_client(request: AdminClientCreateRequest):
    """Create a managed service client and return its secret once."""
    try:
        client = await auth_service.create_service_client(
            client_id=request.client_id,
            scopes=request.scopes,
            namespaces=request.namespaces,
            description=request.description,
        )
        return AdminClientCreateResponse(**client)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail=str(exc),
        ) from exc


@router.get("/clients", response_model=AdminClientListResponse)
async def list_service_clients():
    """List active managed service clients."""
    clients = await auth_service.list_service_clients()
    return AdminClientListResponse(clients=clients)


@router.patch("/clients/{client_id}", response_model=AdminClientSummary)
async def update_service_client(client_id: str, request: AdminClientUpdateRequest):
    """Update a managed service client."""
    client = await auth_service.update_service_client(
        client_id=client_id,
        scopes=request.scopes,
        namespaces=request.namespaces,
        description=request.description,
        is_active=request.is_active,
    )
    if client is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Service client '{client_id}' not found",
        )
    return AdminClientSummary(**client)


@router.post("/clients/{client_id}/reset-secret", response_model=AdminClientResetSecretResponse)
async def reset_client_secret(client_id: str):
    """Reset client secret. Old secret is invalidated. New secret returned once."""
    result = await auth_service.reset_client_secret(client_id)
    if result is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Service client '{client_id}' not found",
        )
    return AdminClientResetSecretResponse(**result)


@router.delete("/clients/{client_id}", status_code=status.HTTP_204_NO_CONTENT)
async def delete_service_client(client_id: str) -> Response:
    """Delete a managed service client."""
    deleted = await auth_service.delete_service_client(client_id)
    if not deleted:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Service client '{client_id}' not found",
        )
    return Response(status_code=status.HTTP_204_NO_CONTENT)

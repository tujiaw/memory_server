from fastapi import APIRouter, HTTPException, status
from typing import Optional, List

from app.models.schemas import (
    UserContext,
    MemoryResponse,
)
from app.services.user_service import user_service

router = APIRouter(prefix="/users", tags=["Users"])


# ============================================================================
# User Management
# ============================================================================

@router.post("", response_model=MemoryResponse, status_code=status.HTTP_201_CREATED)
async def create_user(request: UserContext):
    """
    Create a new user.

    User information is stored in MongoDB for persistent profile management.
    """
    try:
        user = await user_service.create_user(
            user_id=request.user_id,
            name=request.name,
            email=request.email,
            role=request.role,
            preferences=request.preferences,
            custom_data=request.custom_data,
        )
        return MemoryResponse(success=True, message="User created successfully", data=user)
    except Exception as e:
        if "duplicate key" in str(e):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail=f"User with ID '{request.user_id}' already exists",
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create user: {str(e)}",
        )


@router.get("/{user_id}", response_model=MemoryResponse)
async def get_user(user_id: str):
    """Get user information by ID."""
    try:
        user = await user_service.get_user(user_id)

        if user is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User '{user_id}' not found",
            )

        return MemoryResponse(success=True, data=user)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get user: {str(e)}",
        )


@router.put("/{user_id}", response_model=MemoryResponse)
async def update_user(user_id: str, request: UserContext):
    """Update user information."""
    try:
        user = await user_service.update_user(
            user_id=user_id,
            name=request.name,
            email=request.email,
            role=request.role,
            preferences=request.preferences,
            custom_data=request.custom_data,
        )

        if user is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User '{user_id}' not found",
            )

        return MemoryResponse(success=True, message="User updated successfully", data=user)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to update user: {str(e)}",
        )


@router.delete("/{user_id}", response_model=MemoryResponse)
async def delete_user(user_id: str):
    """Delete a user."""
    try:
        success = await user_service.delete_user(user_id)

        if not success:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User '{user_id}' not found",
            )

        return MemoryResponse(success=True, message="User deleted successfully")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete user: {str(e)}",
        )


@router.get("", response_model=MemoryResponse)
async def list_users(limit: int = 100, skip: int = 0):
    """List all users with pagination."""
    try:
        users = await user_service.list_users(limit=limit, skip=skip)
        return MemoryResponse(
            success=True,
            data={"users": users, "count": len(users)}
        )
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list users: {str(e)}",
        )


@router.get("/{user_id}/stats", response_model=MemoryResponse)
async def get_user_stats(user_id: str):
    """Get user statistics."""
    try:
        stats = await user_service.get_user_stats(user_id)

        if stats is None:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"User '{user_id}' not found",
            )

        return MemoryResponse(success=True, data=stats)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get user stats: {str(e)}",
        )

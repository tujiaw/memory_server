import logging

from fastapi import APIRouter, HTTPException, status

from app.core.config import settings
from app.models.schemas import ServiceTokenRequest, ServiceTokenResponse
from app.services.auth_service import auth_service

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/auth", tags=["Authentication"])


@router.post("/token", response_model=ServiceTokenResponse)
async def issue_token(request: ServiceTokenRequest):
    """内部服务使用 client credentials 获取访问令牌。"""
    service_client = await auth_service.authenticate_service_client(
        client_id=request.client_id,
        client_secret=request.client_secret,
    )

    if service_client is None:
        logger.warning(
            "Token request rejected: invalid credentials for client_id=%s",
            request.client_id,
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid client credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    access_token = await auth_service.issue_service_token(
        service_id=service_client.client_id,
        namespaces=service_client.namespaces,
    )

    return ServiceTokenResponse(
        access_token=access_token,
        service_id=service_client.client_id,
        namespaces=service_client.namespaces,
        expires_in=settings.ACCESS_TOKEN_EXPIRE_MINUTES * 60,
    )

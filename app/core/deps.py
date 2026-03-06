import secrets
from fastapi import Depends, Header, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from typing import Annotated, Optional

from app.core.config import settings
from app.models.schemas import AuthContext
from app.services.auth_service import auth_service

security = HTTPBearer(auto_error=False)


async def get_auth_context(
    credentials: Annotated[Optional[HTTPAuthorizationCredentials], Depends(security)],
) -> AuthContext:
    """从 Bearer Token 中解析内部服务身份。"""
    if credentials is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    auth_context = auth_service.verify_token(credentials.credentials)

    if auth_context is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return auth_context


async def get_current_user(
    credentials: Annotated[Optional[HTTPAuthorizationCredentials], Depends(security)],
) -> str:
    """兼容旧依赖接口，返回服务标识。"""
    auth_context = await get_auth_context(credentials)
    return auth_context.service_id


async def require_admin_token(
    admin_token: Annotated[Optional[str], Header(alias="X-Admin-Token")] = None,
) -> None:
    """使用固定管理员令牌保护维护接口。"""
    if not settings.ADMIN_API_TOKEN or admin_token is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing admin token",
        )

    if not secrets.compare_digest(admin_token, settings.ADMIN_API_TOKEN):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid admin token",
        )


def authorize_namespace(auth_context: AuthContext, namespace: str, required_scope: str) -> None:
    """校验当前服务是否具备指定 namespace 的访问权限。"""
    if required_scope not in auth_context.scopes:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Missing required scope: {required_scope}",
        )

    if "*" in auth_context.namespaces:
        return

    if namespace not in auth_context.namespaces:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail=f"Service '{auth_context.service_id}' cannot access namespace '{namespace}'",
        )


def require_scope(required_scope: str):
    async def dependency(
        auth_context: Annotated[AuthContext, Depends(get_auth_context)],
    ) -> AuthContext:
        if required_scope not in auth_context.scopes:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Missing required scope: {required_scope}",
            )
        return auth_context

    return dependency


# 可选的认证（允许匿名访问）
async def get_optional_user(
    credentials: Annotated[Optional[HTTPAuthorizationCredentials], Depends(security)] = None,
) -> Optional[str]:
    """可选认证，不强制要求 token"""
    if credentials is None:
        return None
    return await get_current_user(credentials)

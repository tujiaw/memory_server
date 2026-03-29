from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import logging
import secrets
from typing import Any, Dict, List, Optional

from jose import JWTError, jwt
from passlib.context import CryptContext

from app.core.config import settings
from app.database import postgres as pg_database
from app.database.sql import (
    SERVICE_CLIENT_DELETE,
    SERVICE_CLIENT_INSERT,
    SERVICE_CLIENT_LIST,
    SERVICE_CLIENT_SELECT,
    SERVICE_TOKEN_INSERT,
    SERVICE_TOKEN_SELECT,
)
from app.models.schemas import AuthContext

logger = logging.getLogger(__name__)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


@dataclass(frozen=True)
class ServiceClient:
    client_id: str
    secret: str
    namespaces: List[str]


class AuthService:
    """认证服务：内部服务凭证校验和 JWT Token 生成/验证。"""

    def validate_service_credentials(
        self,
        client_id: str,
        client_secret: str,
    ) -> Optional[ServiceClient]:
        return self._validate_settings_service_client(client_id, client_secret)

    async def authenticate_service_client(
        self,
        client_id: str,
        client_secret: str,
    ) -> Optional[ServiceClient]:
        if pg_database.postgres_db.pool is not None:
            row = await pg_database.postgres_db.fetchrow(SERVICE_CLIENT_SELECT, client_id)
            if row is not None:
                if not row["is_active"]:
                    return None
                if not self.verify_password(client_secret, row["client_secret_hash"]):
                    return None
                return self._build_service_client(
                    client_id=client_id,
                    client_secret=client_secret,
                    namespaces=list(row["namespaces"] or []),
                )

        return self._validate_settings_service_client(client_id, client_secret)

    async def create_service_client(
        self,
        client_id: str,
        namespaces: List[str],
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        if pg_database.postgres_db.pool is None:
            raise RuntimeError("PostgreSQL is not initialized")
        existing = await pg_database.postgres_db.fetchrow(SERVICE_CLIENT_SELECT, client_id)
        if existing is not None:
            raise ValueError(f"Service client '{client_id}' already exists")

        client_secret = secrets.token_urlsafe(32)
        now = datetime.now(timezone.utc)
        await pg_database.postgres_db.execute(
            SERVICE_CLIENT_INSERT,
            client_id,
            self.hash_password(client_secret),
            list(namespaces),
            description,
            now,
        )
        document = {
            "client_id": client_id,
            "namespaces": list(namespaces),
            "description": description,
            "created_at": now,
            "updated_at": now,
            "is_active": True,
        }
        logger.info("Created service client client_id=%s namespaces=%s", client_id, namespaces)
        return {
            **self._serialize_service_client(document),
            "client_secret": client_secret,
        }

    async def list_service_clients(self) -> List[Dict[str, Any]]:
        if pg_database.postgres_db.pool is None:
            return []
        rows = await pg_database.postgres_db.fetch(SERVICE_CLIENT_LIST)
        return [self._serialize_service_client(dict(row)) for row in rows]

    async def update_service_client(
        self,
        client_id: str,
        namespaces: Optional[List[str]] = None,
        description: Optional[str] = None,
        is_active: Optional[bool] = None,
    ) -> Optional[Dict[str, Any]]:
        if pg_database.postgres_db.pool is None:
            return None
        existing = await pg_database.postgres_db.fetchrow(SERVICE_CLIENT_SELECT, client_id)
        if existing is None:
            return None

        now = datetime.now(timezone.utc)
        parts: List[str] = ["updated_at = $1"]
        args: List[Any] = [now]
        idx = 2
        if namespaces is not None:
            parts.append(f"namespaces = ${idx}")
            args.append(list(namespaces))
            idx += 1
        if description is not None:
            parts.append(f"description = ${idx}")
            args.append(description)
            idx += 1
        if is_active is not None:
            parts.append(f"is_active = ${idx}")
            args.append(is_active)
            idx += 1
        args.append(client_id)
        sql = f"UPDATE service_clients SET {', '.join(parts)} WHERE client_id = ${idx}"
        await pg_database.postgres_db.execute(sql, *args)

        updated = await pg_database.postgres_db.fetchrow(SERVICE_CLIENT_SELECT, client_id)
        if updated is None:
            return None
        logger.info("Updated service client client_id=%s", client_id)
        return self._serialize_service_client(
            {
                "client_id": updated["client_id"],
                "namespaces": list(updated["namespaces"] or []),
                "description": updated["description"],
                "created_at": updated["created_at"],
                "updated_at": updated["updated_at"],
                "is_active": updated["is_active"],
            }
        )

    async def reset_client_secret(self, client_id: str) -> Optional[Dict[str, Any]]:
        if pg_database.postgres_db.pool is None:
            return None
        existing = await pg_database.postgres_db.fetchrow(SERVICE_CLIENT_SELECT, client_id)
        if existing is None:
            return None

        new_secret = secrets.token_urlsafe(32)
        now = datetime.now(timezone.utc)
        await pg_database.postgres_db.execute(
            "UPDATE service_clients SET client_secret_hash = $1, updated_at = $2 WHERE client_id = $3",
            self.hash_password(new_secret),
            now,
            client_id,
        )
        logger.info("Reset secret for service client client_id=%s", client_id)
        return {"client_id": client_id, "client_secret": new_secret}

    async def delete_service_client(self, client_id: str) -> bool:
        if pg_database.postgres_db.pool is None:
            return False
        status = await pg_database.postgres_db.execute(SERVICE_CLIENT_DELETE, client_id)
        deleted = status == "DELETE 1"
        if deleted:
            logger.info("Deleted service client client_id=%s", client_id)
        return deleted

    def _validate_settings_service_client(
        self,
        client_id: str,
        client_secret: str,
    ) -> Optional[ServiceClient]:
        client_config = settings.service_clients.get(client_id)

        if not client_config:
            return None

        if client_config.get("secret") != client_secret:
            return None

        return self._build_service_client(
            client_id=client_id,
            client_secret=client_secret,
            namespaces=client_config.get("namespaces", []),
        )

    @staticmethod
    def _build_service_client(
        client_id: str,
        client_secret: str,
        namespaces: List[str],
    ) -> ServiceClient:
        return ServiceClient(
            client_id=client_id,
            secret=client_secret,
            namespaces=list(namespaces),
        )

    @staticmethod
    def _serialize_service_client(document: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "client_id": document["client_id"],
            "namespaces": list(document.get("namespaces", [])),
            "description": document.get("description"),
            "created_at": document.get("created_at").isoformat() if document.get("created_at") else None,
            "updated_at": document.get("updated_at").isoformat() if document.get("updated_at") else None,
            "is_active": bool(document.get("is_active", True)),
        }

    def create_service_token(
        self,
        service_id: str,
        namespaces: List[str],
        expires_delta: Optional[timedelta] = None,
    ) -> str:
        expires_at = self._resolve_token_expiry(expires_delta)
        payload = {
            "sub": service_id,
            "namespaces": namespaces,
            "exp": expires_at,
        }
        return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)

    async def issue_service_token(
        self,
        service_id: str,
        namespaces: List[str],
        expires_delta: Optional[timedelta] = None,
    ) -> str:
        expires_at = self._resolve_token_expiry(expires_delta)
        payload = {
            "sub": service_id,
            "namespaces": namespaces,
            "exp": expires_at,
        }
        token = jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
        await self._persist_service_token(
            token=token,
            service_id=service_id,
            namespaces=namespaces,
            expires_at=expires_at,
        )
        logger.info(
            "Issued service token service_id=%s namespaces=%s expires_at=%s",
            service_id,
            namespaces,
            expires_at.isoformat(),
        )
        return token

    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None) -> str:
        to_encode = data.copy()
        return self.create_service_token(
            service_id=to_encode.get("sub", ""),
            namespaces=list(to_encode.get("namespaces", [])),
            expires_delta=expires_delta,
        )

    async def verify_token(self, token: str) -> Optional[AuthContext]:
        try:
            payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
            service_id = payload.get("sub")
            if service_id is None:
                return None

            namespaces = payload.get("namespaces", [])
            if not isinstance(namespaces, list):
                return None

            if pg_database.postgres_db.pool is not None:
                persisted_token = await pg_database.postgres_db.fetchrow(
                    SERVICE_TOKEN_SELECT,
                    self._hash_token(token),
                )
                if persisted_token is None:
                    return None

                expires_at = self._normalize_utc_datetime(persisted_token.get("expires_at"))
                if persisted_token.get("is_active") is False:
                    return None
                if expires_at is not None and expires_at <= datetime.now(timezone.utc):
                    return None

            return AuthContext(
                service_id=service_id,
                namespaces=[str(namespace) for namespace in namespaces],
            )
        except JWTError as exc:
            logger.debug("JWT verify failed: %s", exc)
            return None

    def hash_password(self, password: str) -> str:
        if len(password.encode("utf-8")) > 72:
            password = password.encode("utf-8")[:72].decode("utf-8", errors="ignore")
        return pwd_context.hash(password)

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        return pwd_context.verify(plain_password, hashed_password)

    @staticmethod
    def _resolve_token_expiry(expires_delta: Optional[timedelta]) -> datetime:
        return datetime.now(timezone.utc) + (
            expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        )

    @staticmethod
    def _normalize_utc_datetime(value: Any) -> Optional[datetime]:
        if not isinstance(value, datetime):
            return None
        if value.tzinfo is None:
            return value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc)

    @staticmethod
    def _hash_token(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    async def _persist_service_token(
        self,
        token: str,
        service_id: str,
        namespaces: List[str],
        expires_at: datetime,
    ) -> None:
        if pg_database.postgres_db.pool is None:
            return

        now = datetime.now(timezone.utc)
        await pg_database.postgres_db.execute(
            SERVICE_TOKEN_INSERT,
            self._hash_token(token),
            service_id,
            list(namespaces),
            expires_at,
            now,
        )


auth_service = AuthService()

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import hashlib
import secrets
from typing import Any, Dict, List, Optional

from jose import JWTError, jwt
from passlib.context import CryptContext

from app.core.config import settings
from app.database.mongodb import get_service_client_collection, get_service_token_collection
from app.models.schemas import AuthContext

# 密码加密
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
        mongo_client = await self._get_service_client_document(client_id)
        if mongo_client is not None:
            if not mongo_client.get("is_active", True):
                return None
            if not self.verify_password(client_secret, mongo_client["client_secret_hash"]):
                return None
            return self._build_service_client(
                client_id=client_id,
                client_secret=client_secret,
                namespaces=mongo_client.get("namespaces", []),
            )

        return self._validate_settings_service_client(client_id, client_secret)

    async def create_service_client(
        self,
        client_id: str,
        namespaces: List[str],
        description: Optional[str] = None,
    ) -> Dict[str, Any]:
        collection = await get_service_client_collection()
        existing_client = await self._get_service_client_document(client_id)

        if existing_client is not None:
            raise ValueError(f"Service client '{client_id}' already exists")

        client_secret = secrets.token_urlsafe(32)
        now = datetime.now(timezone.utc)
        document = {
            "_id": client_id,
            "client_id": client_id,
            "client_secret_hash": self.hash_password(client_secret),
            "namespaces": list(namespaces),
            "description": description,
            "created_at": now,
            "updated_at": now,
            "is_active": True,
        }
        await collection.insert_one(document)
        return {
            **self._serialize_service_client(document),
            "client_secret": client_secret,
        }

    async def list_service_clients(self) -> List[Dict[str, Any]]:
        collection = await get_service_client_collection()
        documents = await collection.find({"is_active": True}).to_list(length=None)
        documents.sort(key=lambda document: document["client_id"])
        return [self._serialize_service_client(document) for document in documents]

    async def update_service_client(
        self,
        client_id: str,
        namespaces: Optional[List[str]] = None,
        description: Optional[str] = None,
        is_active: Optional[bool] = None,
    ) -> Optional[Dict[str, Any]]:
        collection = await get_service_client_collection()
        existing = await self._get_service_client_document(client_id)
        if existing is None:
            return None

        update_data: Dict[str, Any] = {"updated_at": datetime.now(timezone.utc)}
        if namespaces is not None:
            update_data["namespaces"] = list(namespaces)
        if description is not None:
            update_data["description"] = description
        if is_active is not None:
            update_data["is_active"] = is_active

        await collection.update_one({"_id": client_id}, {"$set": update_data})
        return self._serialize_service_client(
            {**existing, **update_data}
        )

    async def reset_client_secret(self, client_id: str) -> Optional[Dict[str, Any]]:
        collection = await get_service_client_collection()
        existing = await self._get_service_client_document(client_id)
        if existing is None:
            return None

        new_secret = secrets.token_urlsafe(32)
        now = datetime.now(timezone.utc)
        await collection.update_one(
            {"_id": client_id},
            {"$set": {"client_secret_hash": self.hash_password(new_secret), "updated_at": now}},
        )
        return {"client_id": client_id, "client_secret": new_secret}

    async def delete_service_client(self, client_id: str) -> bool:
        collection = await get_service_client_collection()
        result = await collection.delete_one({"_id": client_id})
        return result.deleted_count > 0

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
    async def _get_service_client_document(client_id: str) -> Optional[Dict[str, Any]]:
        try:
            collection = await get_service_client_collection()
        except TypeError:
            return None

        return await collection.find_one({"_id": client_id})

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
        return token

    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None) -> str:
        """兼容旧调用，内部转为服务 token。"""
        to_encode = data.copy()
        return self.create_service_token(
            service_id=to_encode.get("sub", ""),
            namespaces=list(to_encode.get("namespaces", [])),
            expires_delta=expires_delta,
        )

    async def verify_token(self, token: str) -> Optional[AuthContext]:
        """验证 token 并返回服务身份上下文。"""
        try:
            payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
            service_id = payload.get("sub")
            if service_id is None:
                return None

            namespaces = payload.get("namespaces", [])

            if not isinstance(namespaces, list):
                return None

            token_collection = await self._get_service_token_collection()
            if token_collection is not None:
                persisted_token = await token_collection.find_one({"_id": self._hash_token(token)})
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
        except JWTError:
            return None

    def hash_password(self, password: str) -> str:
        """兼容旧用户数据结构的密码哈希。"""
        # bcrypt 有 72 字节限制，截断超长密码
        if len(password.encode("utf-8")) > 72:
            password = password.encode("utf-8")[:72].decode("utf-8", errors="ignore")
        return pwd_context.hash(password)

    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """兼容旧用户数据结构的密码校验。"""
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
        collection = await self._get_service_token_collection()
        if collection is None:
            return

        now = datetime.now(timezone.utc)
        await collection.insert_one(
            {
                "_id": self._hash_token(token),
                "service_id": service_id,
                "namespaces": list(namespaces),
                "expires_at": expires_at,
                "created_at": now,
                "updated_at": now,
                "is_active": True,
            }
        )

    @staticmethod
    async def _get_service_token_collection():
        try:
            return await get_service_token_collection()
        except TypeError:
            return None


# 全局实例
auth_service = AuthService()

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import secrets
from typing import Any, Dict, List, Optional

from jose import JWTError, jwt
from passlib.context import CryptContext

from app.core.config import settings
from app.database.mongodb import get_service_client_collection
from app.models.schemas import AuthContext

# 密码加密
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


@dataclass(frozen=True)
class ServiceClient:
    client_id: str
    secret: str
    scopes: List[str]
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
                scopes=mongo_client.get("scopes", []),
                namespaces=mongo_client.get("namespaces", []),
            )

        return self._validate_settings_service_client(client_id, client_secret)

    async def create_service_client(
        self,
        client_id: str,
        scopes: List[str],
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
            "scopes": list(scopes),
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
        scopes: Optional[List[str]] = None,
        namespaces: Optional[List[str]] = None,
        description: Optional[str] = None,
        is_active: Optional[bool] = None,
    ) -> Optional[Dict[str, Any]]:
        collection = await get_service_client_collection()
        existing = await self._get_service_client_document(client_id)
        if existing is None:
            return None

        update_data: Dict[str, Any] = {"updated_at": datetime.now(timezone.utc)}
        if scopes is not None:
            update_data["scopes"] = list(scopes)
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
            scopes=client_config.get("scopes", []),
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
        scopes: List[str],
        namespaces: List[str],
    ) -> ServiceClient:
        return ServiceClient(
            client_id=client_id,
            secret=client_secret,
            scopes=list(scopes),
            namespaces=list(namespaces),
        )

    @staticmethod
    def _serialize_service_client(document: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "client_id": document["client_id"],
            "scopes": list(document.get("scopes", [])),
            "namespaces": list(document.get("namespaces", [])),
            "description": document.get("description"),
            "created_at": document.get("created_at").isoformat() if document.get("created_at") else None,
            "updated_at": document.get("updated_at").isoformat() if document.get("updated_at") else None,
            "is_active": bool(document.get("is_active", True)),
        }

    def create_service_token(
        self,
        service_id: str,
        scopes: List[str],
        namespaces: List[str],
        expires_delta: Optional[timedelta] = None,
    ) -> str:
        expires_at = datetime.now(timezone.utc) + (
            expires_delta or timedelta(minutes=settings.ACCESS_TOKEN_EXPIRE_MINUTES)
        )
        payload = {
            "sub": service_id,
            "scopes": scopes,
            "namespaces": namespaces,
            "exp": expires_at,
        }
        return jwt.encode(payload, settings.SECRET_KEY, algorithm=settings.ALGORITHM)

    def create_access_token(self, data: dict, expires_delta: Optional[timedelta] = None) -> str:
        """兼容旧调用，内部转为服务 token。"""
        to_encode = data.copy()
        return self.create_service_token(
            service_id=to_encode.get("sub", ""),
            scopes=list(to_encode.get("scopes", [])),
            namespaces=list(to_encode.get("namespaces", [])),
            expires_delta=expires_delta,
        )

    def verify_token(self, token: str) -> Optional[AuthContext]:
        """验证 token 并返回服务身份上下文。"""
        try:
            payload = jwt.decode(token, settings.SECRET_KEY, algorithms=[settings.ALGORITHM])
            service_id = payload.get("sub")
            if service_id is None:
                return None

            scopes = payload.get("scopes", [])
            namespaces = payload.get("namespaces", [])

            if not isinstance(scopes, list) or not isinstance(namespaces, list):
                return None

            return AuthContext(
                service_id=service_id,
                scopes=[str(scope) for scope in scopes],
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


# 全局实例
auth_service = AuthService()

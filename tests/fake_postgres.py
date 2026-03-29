"""In-memory PostgreSQL stub for admin/auth tests (no real Postgres required)."""

from __future__ import annotations

from copy import deepcopy
from datetime import datetime
from typing import Any, Dict, List, Optional


class FakePostgresDB:
    """Minimal stub matching call patterns used by auth_service."""

    def __init__(self) -> None:
        self.pool = object()
        self.clients: Dict[str, Dict[str, Any]] = {}
        self.tokens: Dict[str, Dict[str, Any]] = {}

    async def connect(self) -> None:
        return None

    async def disconnect(self) -> None:
        return None

    async def init_schema(self) -> None:
        return None

    async def fetchrow(self, query: str, *args: Any) -> Optional[Dict[str, Any]]:
        q = " ".join(query.split())
        if "FROM service_clients" in q and "WHERE client_id" in q:
            cid = str(args[0])
            row = self.clients.get(cid)
            return deepcopy(row) if row else None
        if "FROM service_tokens" in q and "WHERE token_hash" in q:
            th = args[0]
            row = self.tokens.get(th)
            return deepcopy(row) if row else None
        return None

    async def fetch(self, query: str, *args: Any) -> List[Dict[str, Any]]:
        q = " ".join(query.split())
        if "FROM service_clients" in q and "is_active" in q and "ORDER BY" in q:
            return [deepcopy(v) for v in self.clients.values() if v.get("is_active", True)]
        return []

    async def execute(self, query: str, *args: Any) -> str:
        q = " ".join(query.split())
        if "INSERT INTO service_clients" in q:
            client_id, secret_hash, namespaces, description, now = args[0], args[1], args[2], args[3], args[4]
            self.clients[str(client_id)] = {
                "client_id": str(client_id),
                "client_secret_hash": secret_hash,
                "namespaces": list(namespaces),
                "description": description,
                "created_at": now,
                "updated_at": now,
                "is_active": True,
            }
            return "INSERT 0 1"
        if "INSERT INTO service_tokens" in q:
            token_hash, service_id, namespaces, expires_at, now = args
            self.tokens[str(token_hash)] = {
                "token_hash": str(token_hash),
                "service_id": service_id,
                "namespaces": list(namespaces),
                "expires_at": expires_at,
                "is_active": True,
                "created_at": now,
                "updated_at": now,
            }
            return "INSERT 0 1"
        if "UPDATE service_clients SET" in q and "WHERE client_id" in q:
            cid = str(args[-1])
            if cid not in self.clients:
                return "UPDATE 0"
            doc = self.clients[cid]
            if "client_secret_hash = $1" in q and len(args) >= 3:
                doc["client_secret_hash"] = args[0]
                doc["updated_at"] = args[1]
                return "UPDATE 1"
            doc["updated_at"] = args[0]
            idx = 1
            if "namespaces = $" in q:
                doc["namespaces"] = list(args[idx])
                idx += 1
            if "description = $" in q:
                doc["description"] = args[idx]
                idx += 1
            if "is_active = $" in q:
                doc["is_active"] = args[idx]
            return "UPDATE 1"
        if "DELETE FROM service_clients" in q:
            cid = str(args[0])
            if cid in self.clients:
                del self.clients[cid]
                return "DELETE 1"
            return "DELETE 0"
        return "OK 0"

import json
from datetime import datetime, timezone
from typing import Any, Dict, Optional

from app.database import postgres as pg_database
from app.database.sql import (
    SUBJECT_INCREMENT_MEMORY,
    SUBJECT_SELECT,
    SUBJECT_UPSERT_CONTEXT,
    SUBJECT_UPSERT_TOUCH,
)


class UserService:
    """Subject context and activity in PostgreSQL."""

    async def upsert_subject_context(
        self,
        namespace: str,
        subject_id: str,
        name: Optional[str] = None,
        email: Optional[str] = None,
        role: Optional[str] = None,
        preferences: Optional[Dict[str, Any]] = None,
        custom_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        if pg_database.postgres_db.pool is None:
            raise RuntimeError("PostgreSQL is not initialized")
        now = datetime.now(timezone.utc)
        await pg_database.postgres_db.execute(
            SUBJECT_UPSERT_CONTEXT,
            namespace,
            subject_id,
            name,
            email,
            role,
            json.dumps(preferences or {}),
            json.dumps(custom_data or {}),
            now,
        )
        return await self.get_subject_context(namespace, subject_id) or {}

    async def get_subject_context(self, namespace: str, subject_id: str) -> Optional[Dict[str, Any]]:
        if pg_database.postgres_db.pool is None:
            return None
        row = await pg_database.postgres_db.fetchrow(SUBJECT_SELECT, namespace, subject_id)
        if row is None:
            return None
        return self._row_to_subject(row)

    async def touch_subject(self, namespace: str, subject_id: str) -> None:
        if pg_database.postgres_db.pool is None:
            return
        now = datetime.now(timezone.utc)
        await pg_database.postgres_db.execute(SUBJECT_UPSERT_TOUCH, namespace, subject_id, now)

    async def increment_memory_count(self, namespace: str, subject_id: str, amount: int = 1) -> None:
        if pg_database.postgres_db.pool is None:
            return
        now = datetime.now(timezone.utc)
        await pg_database.postgres_db.execute(SUBJECT_INCREMENT_MEMORY, namespace, subject_id, now, amount)

    async def get_subject_stats(self, namespace: str, subject_id: str) -> Optional[Dict[str, Any]]:
        context = await self.get_subject_context(namespace, subject_id)
        if context is None:
            return None
        return {
            "namespace": namespace,
            "subject_id": subject_id,
            "name": context.get("name"),
            "email": context.get("email"),
            "memory_count": context.get("memory_count", 0),
            "created_at": context.get("created_at"),
            "last_active": context.get("last_active"),
        }

    @staticmethod
    def _row_to_subject(row: Any) -> Dict[str, Any]:
        prefs = row["preferences"]
        if isinstance(prefs, str):
            prefs = json.loads(prefs)
        custom = row["custom_data"]
        if isinstance(custom, str):
            custom = json.loads(custom)

        def iso(v: Any) -> Optional[str]:
            if v is None:
                return None
            if isinstance(v, datetime):
                return v.isoformat()
            return str(v)

        return {
            "namespace": row["namespace"],
            "subject_id": row["subject_id"],
            "name": row["name"],
            "email": row["email"],
            "role": row["role"],
            "preferences": prefs or {},
            "custom_data": custom or {},
            "created_at": iso(row["created_at"]),
            "updated_at": iso(row["updated_at"]),
            "last_active": iso(row["last_active"]),
            "memory_count": row["memory_count"] or 0,
        }


user_service = UserService()

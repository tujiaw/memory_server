from datetime import datetime, timezone
from typing import Any, Dict, Optional

from app.database.mongodb import get_subject_collection


class UserService:
    """Service for managing subject context in MongoDB."""

    @staticmethod
    def _subject_key(namespace: str, subject_id: str) -> str:
        return f"{namespace}:{subject_id}"

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
        collection = await get_subject_collection()
        now = datetime.now(timezone.utc)

        update_data = {
            "namespace": namespace,
            "subject_id": subject_id,
            "updated_at": now,
        }
        if name is not None:
            update_data["name"] = name
        if email is not None:
            update_data["email"] = email
        if role is not None:
            update_data["role"] = role
        if preferences is not None:
            update_data["preferences"] = preferences
        if custom_data is not None:
            update_data["custom_data"] = custom_data

        await collection.update_one(
            {"_id": self._subject_key(namespace, subject_id)},
            {
                "$set": update_data,
                "$setOnInsert": {
                    "created_at": now,
                    "last_active": now,
                    "memory_count": 0,
                },
            },
            upsert=True,
        )

        return await self.get_subject_context(namespace, subject_id)

    async def get_subject_context(self, namespace: str, subject_id: str) -> Optional[Dict[str, Any]]:
        collection = await get_subject_collection()
        document = await collection.find_one({"_id": self._subject_key(namespace, subject_id)})

        if document is None:
            return None

        return self._serialize_subject(document)

    async def touch_subject(self, namespace: str, subject_id: str) -> None:
        collection = await get_subject_collection()
        await collection.update_one(
            {"_id": self._subject_key(namespace, subject_id)},
            {
                "$set": {
                    "namespace": namespace,
                    "subject_id": subject_id,
                    "last_active": datetime.now(timezone.utc),
                },
                "$setOnInsert": {
                    "created_at": datetime.now(timezone.utc),
                    "updated_at": datetime.now(timezone.utc),
                    "memory_count": 0,
                    "preferences": {},
                    "custom_data": {},
                },
            },
            upsert=True,
        )

    async def increment_memory_count(self, namespace: str, subject_id: str, amount: int = 1) -> None:
        collection = await get_subject_collection()
        now = datetime.now(timezone.utc)
        await collection.update_one(
            {"_id": self._subject_key(namespace, subject_id)},
            {
                "$set": {
                    "namespace": namespace,
                    "subject_id": subject_id,
                    "last_active": now,
                    "updated_at": now,
                },
                "$setOnInsert": {
                    "created_at": now,
                    "preferences": {},
                    "custom_data": {},
                },
                "$inc": {"memory_count": amount},
            },
            upsert=True,
        )

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
    def _serialize_subject(document: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "namespace": document["namespace"],
            "subject_id": document["subject_id"],
            "name": document.get("name"),
            "email": document.get("email"),
            "role": document.get("role"),
            "preferences": document.get("preferences", {}),
            "custom_data": document.get("custom_data", {}),
            "created_at": document.get("created_at").isoformat() if document.get("created_at") else None,
            "updated_at": document.get("updated_at").isoformat() if document.get("updated_at") else None,
            "last_active": document.get("last_active").isoformat() if document.get("last_active") else None,
            "memory_count": document.get("memory_count", 0),
        }


# Global instance
user_service = UserService()

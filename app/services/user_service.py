from typing import Optional, Dict, Any, List
from datetime import datetime
from bson import ObjectId
from app.database.mongodb import get_user_collection


class UserService:
    """Service for managing user profiles in MongoDB."""

    async def create_user(
        self,
        user_id: str,
        name: Optional[str] = None,
        email: Optional[str] = None,
        role: Optional[str] = None,
        preferences: Optional[Dict[str, Any]] = None,
        custom_data: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Create a new user.

        Returns the created user document.
        """
        collection = await get_user_collection()

        user_doc = {
            "_id": user_id,
            "name": name,
            "email": email,
            "role": role,
            "preferences": preferences or {},
            "custom_data": custom_data or {},
            "created_at": datetime.now(),
            "updated_at": datetime.now(),
            "last_active": datetime.now(),
            "memory_count": 0,
        }

        await collection.insert_one(user_doc)

        return self._serialize_user(user_doc)

    async def get_user(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get a user by ID."""
        collection = await get_user_collection()

        user_doc = await collection.find_one({"_id": user_id})

        if user_doc:
            return self._serialize_user(user_doc)

        return None

    async def update_user(
        self,
        user_id: str,
        name: Optional[str] = None,
        email: Optional[str] = None,
        role: Optional[str] = None,
        preferences: Optional[Dict[str, Any]] = None,
        custom_data: Optional[Dict[str, Any]] = None,
    ) -> Optional[Dict[str, Any]]:
        """Update user information."""
        collection = await get_user_collection()

        update_data = {"updated_at": datetime.now()}

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

        result = await collection.update_one(
            {"_id": user_id},
            {"$set": update_data}
        )

        if result.modified_count > 0:
            return await self.get_user(user_id)

        return None

    async def delete_user(self, user_id: str) -> bool:
        """Delete a user."""
        collection = await get_user_collection()

        result = await collection.delete_one({"_id": user_id})

        return result.deleted_count > 0

    async def update_last_active(self, user_id: str) -> None:
        """Update user's last active timestamp."""
        collection = await get_user_collection()

        await collection.update_one(
            {"_id": user_id},
            {"$set": {"last_active": datetime.now()}}
        )

    async def increment_memory_count(self, user_id: str) -> None:
        """Increment user's memory count."""
        collection = await get_user_collection()

        await collection.update_one(
            {"_id": user_id},
            {"$inc": {"memory_count": 1}}
        )

    async def get_user_stats(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user statistics."""
        user = await self.get_user(user_id)

        if user:
            return {
                "user_id": user["id"],
                "name": user.get("name"),
                "email": user.get("email"),
                "memory_count": user.get("memory_count", 0),
                "created_at": user.get("created_at"),
                "last_active": user.get("last_active"),
            }

        return None

    async def list_users(
        self,
        limit: int = 100,
        skip: int = 0,
    ) -> List[Dict[str, Any]]:
        """List all users with pagination."""
        collection = await get_user_collection()

        cursor = collection.find().skip(skip).limit(limit)
        users = await cursor.to_list(length=limit)

        return [self._serialize_user(user) for user in users]

    def _serialize_user(self, user_doc: Dict[str, Any]) -> Dict[str, Any]:
        """Convert MongoDB document to API response format."""
        return {
            "id": user_doc["_id"],
            "name": user_doc.get("name"),
            "email": user_doc.get("email"),
            "role": user_doc.get("role"),
            "preferences": user_doc.get("preferences", {}),
            "custom_data": user_doc.get("custom_data", {}),
            "created_at": user_doc.get("created_at"),
            "updated_at": user_doc.get("updated_at"),
            "last_active": user_doc.get("last_active"),
            "memory_count": user_doc.get("memory_count", 0),
        }


# Global instance
user_service = UserService()

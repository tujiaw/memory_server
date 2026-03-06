from motor.motor_asyncio import AsyncIOMotorClient
from typing import Optional
from app.core.config import settings


class MongoDB:
    """MongoDB database client."""

    def __init__(self):
        self.client: Optional[AsyncIOMotorClient] = None
        self.db = None

    async def connect(self):
        """Initialize MongoDB connection."""
        self.client = AsyncIOMotorClient(settings.MONGO_URI)
        self.db = self.client[settings.MONGO_DB]
        print(f"Connected to MongoDB: {settings.MONGO_DB}")

    async def disconnect(self):
        """Close MongoDB connection."""
        if self.client:
            self.client.close()
            print("MongoDB connection closed")

    def get_collection(self, collection_name: str):
        """Get a collection by name."""
        return self.db[collection_name]


# Global instance
mongodb = MongoDB()


async def get_user_collection():
    """Backward-compatible alias for the subjects collection."""
    return mongodb.get_collection(settings.MONGO_COLLECTION_USERS)


async def get_subject_collection():
    """Get the subject context collection."""
    collection_name = getattr(settings, "MONGO_COLLECTION_SUBJECTS", settings.MONGO_COLLECTION_USERS)
    return mongodb.get_collection(collection_name)


async def get_service_client_collection():
    """Get the service client collection."""
    return mongodb.get_collection(settings.MONGO_COLLECTION_SERVICE_CLIENTS)

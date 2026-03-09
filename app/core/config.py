import json
from typing import Any, Dict

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore"
    )

    # Application
    APP_NAME: str = "Memory Server"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    HOST: str = "0.0.0.0"
    PORT: int = 8899

    # Qdrant
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_GRPC_PORT: int = 6334
    QDRANT_API_KEY: str = ""
    VECTOR_SIZE: int = 1536

    # MongoDB
    MONGO_URI: str = "mongodb://localhost:27017"
    MONGO_DB: str = "memory_server"
    MONGO_COLLECTION_USERS: str = "users"
    MONGO_COLLECTION_SUBJECTS: str = "subjects"
    MONGO_COLLECTION_SERVICE_CLIENTS: str = "service_clients"
    MONGO_COLLECTION_SERVICE_TOKENS: str = "service_tokens"

    # OpenAI Configuration
    OPENAI_API_KEY: str
    OPENAI_API_BASE: str
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"

    # Mem0 Configuration
    MEM0_API_KEY: str = ""
    MEM0_SEARCH_MSG_LIMIT: int = 5
    NEO4J_URL: str = ""
    NEO4J_USERNAME: str = ""
    NEO4J_PASSWORD: str = ""
    MEM0_GRAPH_CUSTOM_PROMPT: str = ""

    # Security
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 24 * 60 # 24 hours
    ADMIN_API_TOKEN: str = ""
    SERVICE_CLIENTS_JSON: str = "{}"

    @property
    def qdrant_url(self) -> str:
        if self.QDRANT_API_KEY:
            return f"https://{self.QDRANT_HOST}:{self.QDRANT_PORT}"
        return f"http://{self.QDRANT_HOST}:{self.QDRANT_PORT}"

    @property
    def service_clients(self) -> Dict[str, Dict[str, Any]]:
        try:
            raw_value = json.loads(self.SERVICE_CLIENTS_JSON)
        except json.JSONDecodeError as exc:
            raise ValueError("SERVICE_CLIENTS_JSON must be valid JSON") from exc

        if not isinstance(raw_value, dict):
            raise ValueError("SERVICE_CLIENTS_JSON must be a JSON object")

        return raw_value


settings = Settings()

# Set OpenAI environment variables for mem0
import os

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY
os.environ["OPENAI_API_BASE"] = settings.OPENAI_API_BASE

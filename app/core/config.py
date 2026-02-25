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

    # OpenAI Configuration
    OPENAI_API_KEY: str
    OPENAI_API_BASE: str
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"

    # Mem0 Configuration
    MEM0_API_KEY: str = ""
    MEM0_SEARCH_MSG_LIMIT: int = 5

    @property
    def qdrant_url(self) -> str:
        if self.QDRANT_API_KEY:
            return f"https://{self.QDRANT_HOST}:{self.QDRANT_PORT}"
        return f"http://{self.QDRANT_HOST}:{settings.QDRANT_PORT}"


settings = Settings()

# Set OpenAI environment variables for mem0
import os
os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY
os.environ["OPENAI_API_BASE"] = settings.OPENAI_API_BASE

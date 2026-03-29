import json
import os
from typing import Any, Dict, Optional

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

# 先于 Settings：把 .env 合并进 os.environ，并修正代理（httpx 仅识别 socks5:// / socks4://，不认 socks://）
load_dotenv()


def _normalize_socks_proxy_urls_in_environ() -> None:
    for key in (
        "ALL_PROXY",
        "all_proxy",
        "HTTP_PROXY",
        "http_proxy",
        "HTTPS_PROXY",
        "https_proxy",
    ):
        val = os.environ.get(key)
        if val and val.startswith("socks://"):
            os.environ[key] = "socks5://" + val[len("socks://") :]


_normalize_socks_proxy_urls_in_environ()


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

    DATABASE_URL: str = "postgresql://memory:memory@localhost:5433/memory_server"
    # asyncpg 默认会尝试 SSL；本地 Docker/ParadeDB 多为明文，需显式关闭，否则握手阶段可能被 RST
    DATABASE_SSL: bool = False

    # Qdrant
    QDRANT_HOST: str = "localhost"
    QDRANT_PORT: int = 6333
    QDRANT_GRPC_PORT: int = 6334
    QDRANT_API_KEY: str = ""
    QDRANT_MEM0_COLLECTION: str = "mem0_global_memory"
    VECTOR_SIZE: int = 1536

    # OpenAI Configuration
    OPENAI_API_KEY: str
    OPENAI_BASE_URL: str
    OPENAI_MODEL: str = "gpt-4o-mini"
    OPENAI_EMBEDDING_MODEL: str = "text-embedding-3-small"

    # Mem0 Configuration
    MEM0_API_KEY: str = ""
    MEM0_SEARCH_MSG_LIMIT: int = 5
    MEM0_HYBRID_VECTOR_WEIGHT: float = 0.7
    MEM0_HYBRID_BM25_WEIGHT: float = 0.3
    MEM0_SEARCH_VECTOR_MIN_SCORE: float = 0.0
    MEM0_SEARCH_BM25_MIN_SCORE: Optional[float] = None
    MEM0_SEARCH_MIN_HYBRID_SCORE: Optional[float] = None

    # Security
    SECRET_KEY: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 24 * 60 # 24 hours
    ADMIN_API_TOKEN: str = "123456"
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

os.environ["OPENAI_API_KEY"] = settings.OPENAI_API_KEY
os.environ["OPENAI_BASE_URL"] = settings.OPENAI_BASE_URL

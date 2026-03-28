import asyncio
from typing import Any, Awaitable, Callable, List, Optional, TypeVar

import asyncpg

from app.core.config import settings

T = TypeVar("T")


def _is_transient_pool_error(exc: BaseException) -> bool:
    """连接在执行中途被对端关闭、池回收或网络闪断时，可换连接重试。"""
    if isinstance(
        exc,
        (
            asyncpg.exceptions.ConnectionDoesNotExistError,
            asyncpg.exceptions.InterfaceError,
            asyncpg.exceptions.PostgresConnectionError,
            ConnectionResetError,
            BrokenPipeError,
        ),
    ):
        return True
    if isinstance(exc, OSError):
        # Linux: 104=ECONNRESET, 32=EPIPE, 110=ETIMEDOUT, 103=ECONNABORTED
        errno = getattr(exc, "errno", None)
        if errno in (104, 32, 110, 103):
            return True
    return False


def _is_transient_connect_error(exc: BaseException) -> bool:
    """建连 / create_pool 阶段：容器未就绪、恢复模式、握手被掐断等可重试。"""
    if isinstance(exc, (TimeoutError, asyncio.TimeoutError)):
        return True
    if isinstance(exc, ConnectionRefusedError):
        return True
    if isinstance(
        exc,
        (
            asyncpg.exceptions.CannotConnectNowError,
            asyncpg.exceptions.ConnectionFailureError,
            asyncpg.exceptions.ConnectionRejectionError,
        ),
    ):
        return True
    return _is_transient_pool_error(exc)

SCHEMA_STATEMENTS: List[str] = [
    """
    CREATE TABLE IF NOT EXISTS subjects (
        namespace TEXT NOT NULL,
        subject_id TEXT NOT NULL,
        name TEXT,
        email TEXT,
        role TEXT,
        preferences JSONB NOT NULL DEFAULT '{}',
        custom_data JSONB NOT NULL DEFAULT '{}',
        memory_count INT NOT NULL DEFAULT 0,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        last_active TIMESTAMPTZ,
        PRIMARY KEY (namespace, subject_id)
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS service_clients (
        client_id TEXT PRIMARY KEY,
        client_secret_hash TEXT NOT NULL,
        namespaces TEXT[] NOT NULL DEFAULT ARRAY[]::TEXT[],
        description TEXT,
        is_active BOOLEAN NOT NULL DEFAULT true,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS service_tokens (
        token_hash TEXT PRIMARY KEY,
        service_id TEXT NOT NULL,
        namespaces TEXT[] NOT NULL,
        expires_at TIMESTAMPTZ NOT NULL,
        is_active BOOLEAN NOT NULL DEFAULT true,
        created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
        updated_at TIMESTAMPTZ NOT NULL DEFAULT now()
    )
    """,
    """
    CREATE TABLE IF NOT EXISTS memory_lexical (
        memory_id TEXT NOT NULL PRIMARY KEY,
        namespace TEXT NOT NULL,
        subject_id TEXT NOT NULL,
        run_id TEXT,
        content TEXT NOT NULL,
        metadata JSONB NOT NULL DEFAULT '{}',
        created_at TIMESTAMPTZ,
        updated_at TIMESTAMPTZ
    )
    """,
    """
    CREATE INDEX IF NOT EXISTS memory_lexical_ns_subj_run
    ON memory_lexical (namespace, subject_id, run_id)
    """,
    """
    CREATE INDEX IF NOT EXISTS memory_lexical_metadata_gin
    ON memory_lexical USING gin (metadata jsonb_path_ops)
    """,
    "DROP INDEX IF EXISTS memory_lexical_bm25_idx",
    """
    CREATE INDEX memory_lexical_bm25_idx ON memory_lexical
    USING bm25 (
        memory_id,
        (content::pdb.lindera('chinese')),
        namespace,
        subject_id,
        run_id
    )
    WITH (key_field='memory_id')
    """,
]


class PostgresDB:
    def __init__(self):
        self.pool: Optional[asyncpg.Pool] = None

    async def _run_conn(self, fn: Callable[[asyncpg.Connection], Awaitable[T]]) -> T:
        for attempt in range(4):
            try:
                if self.pool is None:
                    raise RuntimeError("PostgreSQL pool is not initialized")
                async with self.pool.acquire() as conn:
                    return await fn(conn)
            except Exception as exc:
                if not _is_transient_pool_error(exc) or attempt == 3:
                    raise
                await asyncio.sleep(0.06 * (2**attempt))

    async def connect(self) -> None:
        ssl = None if settings.DATABASE_SSL else False
        for attempt in range(15):
            try:
                self.pool = await asyncpg.create_pool(
                    settings.DATABASE_URL,
                    min_size=1,
                    max_size=10,
                    max_inactive_connection_lifetime=60.0,
                    ssl=ssl,
                    timeout=90,
                )
                await self.init_schema()
                return
            except Exception as exc:
                if self.pool is not None:
                    try:
                        await self.pool.close()
                    finally:
                        self.pool = None
                if not _is_transient_connect_error(exc) or attempt == 14:
                    raise
                await asyncio.sleep(min(0.5 * (1.55**attempt), 10.0))

    async def disconnect(self) -> None:
        if self.pool is not None:
            await self.pool.close()
            self.pool = None

    async def init_schema(self) -> None:
        if self.pool is None:
            return
        for attempt in range(4):
            try:
                async with self.pool.acquire() as conn:
                    for stmt in SCHEMA_STATEMENTS:
                        s = stmt.strip()
                        if s:
                            await conn.execute(s)
                return
            except Exception as exc:
                if not _is_transient_pool_error(exc) or attempt == 3:
                    raise
                await asyncio.sleep(0.06 * (2**attempt))

    async def execute(self, query: str, *args: Any) -> str:
        return await self._run_conn(lambda conn: conn.execute(query, *args))

    async def fetch(self, query: str, *args: Any) -> List[asyncpg.Record]:
        return await self._run_conn(lambda conn: conn.fetch(query, *args))

    async def fetchrow(self, query: str, *args: Any) -> Optional[asyncpg.Record]:
        return await self._run_conn(lambda conn: conn.fetchrow(query, *args))

    async def fetchval(self, query: str, *args: Any) -> Any:
        return await self._run_conn(lambda conn: conn.fetchval(query, *args))


postgres_db = PostgresDB()

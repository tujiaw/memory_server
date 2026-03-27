from typing import Any, List, Optional

import asyncpg

from app.core.config import settings

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
    USING bm25 (memory_id, content, namespace, subject_id, run_id)
    WITH (key_field='memory_id')
    """,
]


class PostgresDB:
    def __init__(self):
        self.pool: Optional[asyncpg.Pool] = None

    async def connect(self) -> None:
        self.pool = await asyncpg.create_pool(
            settings.DATABASE_URL,
            min_size=1,
            max_size=10,
        )
        await self.init_schema()

    async def disconnect(self) -> None:
        if self.pool is not None:
            await self.pool.close()
            self.pool = None

    async def init_schema(self) -> None:
        if self.pool is None:
            return
        async with self.pool.acquire() as conn:
            for stmt in SCHEMA_STATEMENTS:
                s = stmt.strip()
                if s:
                    await conn.execute(s)

    async def execute(self, query: str, *args: Any) -> str:
        if self.pool is None:
            raise RuntimeError("PostgreSQL pool is not initialized")
        async with self.pool.acquire() as conn:
            return await conn.execute(query, *args)

    async def fetch(self, query: str, *args: Any) -> List[asyncpg.Record]:
        if self.pool is None:
            raise RuntimeError("PostgreSQL pool is not initialized")
        async with self.pool.acquire() as conn:
            return await conn.fetch(query, *args)

    async def fetchrow(self, query: str, *args: Any) -> Optional[asyncpg.Record]:
        if self.pool is None:
            raise RuntimeError("PostgreSQL pool is not initialized")
        async with self.pool.acquire() as conn:
            return await conn.fetchrow(query, *args)

    async def fetchval(self, query: str, *args: Any) -> Any:
        if self.pool is None:
            raise RuntimeError("PostgreSQL pool is not initialized")
        async with self.pool.acquire() as conn:
            return await conn.fetchval(query, *args)


postgres_db = PostgresDB()

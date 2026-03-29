from contextlib import asynccontextmanager
import logging
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.admin_routes import router as admin_router
from app.core.config import settings
from app.api.auth_routes import router as auth_router
from app.api.mem0_routes import router as memories_router
from app.api.user_routes import router as users_router
from app.database import postgres as pg_database
from app.database.es_memory import cluster_health_snippet, ensure_mem0_index, make_elasticsearch_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
)


def build_error_response(
    code: str,
    message: str,
    detail: Any = None,
    status_code: int = status.HTTP_500_INTERNAL_SERVER_ERROR,
) -> JSONResponse:
    payload: Dict[str, Any] = {
        "success": False,
        "code": code,
        "message": message,
    }
    if detail is not None and settings.DEBUG:
        payload["detail"] = detail
    return JSONResponse(status_code=status_code, content=payload)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events."""
    print(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    print(f"Elasticsearch: {settings.ELASTICSEARCH_HOST}:{settings.ELASTICSEARCH_PORT} index={settings.ELASTICSEARCH_INDEX}")
    print(f"PostgreSQL: {settings.DATABASE_URL.split('@')[-1] if '@' in settings.DATABASE_URL else settings.DATABASE_URL}")
    print(f"OpenAI: {settings.OPENAI_MODEL} + {settings.OPENAI_EMBEDDING_MODEL}")
    await pg_database.postgres_db.connect()
    es = make_elasticsearch_client()
    ensure_mem0_index(es, settings.ELASTICSEARCH_INDEX, settings.VECTOR_SIZE)
    yield
    print("Shutting down...")
    await pg_database.postgres_db.disconnect()


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Internal Memory MVP API: FastAPI + mem0 + Elasticsearch (vector + BM25) + PostgreSQL",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(HTTPException)
async def http_exception_handler(_: Request, exc: HTTPException) -> JSONResponse:
    detail = exc.detail if isinstance(exc.detail, str) else str(exc.detail)
    return build_error_response(
        code=f"HTTP_{exc.status_code}",
        message=detail,
        detail=detail,
        status_code=exc.status_code,
    )


@app.exception_handler(Exception)
async def unhandled_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    logging.exception("Unhandled server error")
    return build_error_response(
        code="INTERNAL_ERROR",
        message="Internal server error",
        detail=str(exc),
    )


@app.get("/")
async def root():
    return {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "endpoints": {
            "token": "/api/v1/auth/token",
            "subjects": "/api/v1/subjects",
        },
    }


app.include_router(auth_router, prefix="/api/v1")
app.include_router(memories_router, prefix="/api/v1")
app.include_router(users_router, prefix="/api/v1")
app.include_router(admin_router, prefix="/api/v1")


async def collect_health_status() -> Dict[str, Any]:
    services: Dict[str, str] = {
        "postgresql": "healthy",
        "elasticsearch": "healthy",
        "openai_config": "healthy",
    }

    try:
        if pg_database.postgres_db.pool is None:
            raise RuntimeError("PostgreSQL pool is not initialized")
        await pg_database.postgres_db.fetchval("SELECT 1")
    except Exception:
        services["postgresql"] = "unhealthy"

    try:
        es = make_elasticsearch_client()
        status = cluster_health_snippet(es)
        if status not in ("green", "yellow"):
            raise RuntimeError(f"Elasticsearch cluster status: {status}")
    except Exception:
        services["elasticsearch"] = "unhealthy"

    if not settings.OPENAI_API_KEY or not settings.OPENAI_BASE_URL:
        services["openai_config"] = "unhealthy"

    overall_status = "healthy" if all(state == "healthy" for state in services.values()) else "degraded"
    return {"status": overall_status, "services": services}


@app.get("/health")
async def health(response: Response):
    """Health check endpoint with real dependency probing."""
    payload = await collect_health_status()
    if payload["status"] != "healthy":
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return payload


if __name__ == "__main__":
    import os
    import uvicorn

    # Uvicorn 会监视 cwd；Docker 挂载的 data/ 子目录可能不可读，导致 watchfiles 抛 PermissionError
    if settings.DEBUG:
        os.environ.setdefault("WATCHFILES_IGNORE_PERMISSION_DENIED", "1")

    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        reload_dirs=["app"] if settings.DEBUG else None,
    )

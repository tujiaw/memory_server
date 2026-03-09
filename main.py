from contextlib import asynccontextmanager
import logging
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from qdrant_client import QdrantClient

from app.api.admin_routes import router as admin_router
from app.core.config import settings
from app.api.auth_routes import router as auth_router
from app.api.mem0_routes import router as memories_router
from app.api.user_routes import router as users_router
from app.database.mongodb import mongodb

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
    print(f"Qdrant: {settings.QDRANT_HOST}:{settings.QDRANT_PORT}")
    print(f"MongoDB: {settings.MONGO_URI}")
    print(f"OpenAI: {settings.OPENAI_MODEL} + {settings.OPENAI_EMBEDDING_MODEL}")
    await mongodb.connect()
    yield
    print("Shutting down...")
    await mongodb.disconnect()


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Internal Memory MVP API powered by FastAPI + mem0 + Qdrant + MongoDB",
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
async def unexpected_exception_handler(_: Request, exc: Exception) -> JSONResponse:
    return build_error_response(
        code="INTERNAL_SERVER_ERROR",
        message="Internal server error",
        detail=str(exc),
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
    )


app.include_router(auth_router, prefix="/api/v1")
app.include_router(admin_router, prefix="/api/v1")
app.include_router(memories_router, prefix="/api/v1")
app.include_router(users_router, prefix="/api/v1")


@app.get("/")
async def root():
    """API information and available endpoints."""
    return {
        "name": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "running",
        "docs": "/docs",
        "endpoints": {
            "token": "/api/v1/auth/token",
            "admin_clients": "/api/v1/admin/clients",
            "memories": "/api/v1/memories",
            "subjects": "/api/v1/subjects",
        },
        "features": [
            "Service-to-service JWT authentication",
            "Subject-scoped semantic memory storage",
            "Subject context management",
            "Conversation memory extraction",
            "Batch memory writes",
        ],
    }


async def collect_health_status() -> Dict[str, Any]:
    services: Dict[str, str] = {
        "mongodb": "healthy",
        "qdrant": "healthy",
        "openai_config": "healthy",
    }

    try:
        if mongodb.client is None:
            raise RuntimeError("MongoDB client is not initialized")
        await mongodb.client.admin.command("ping")
    except Exception:
        services["mongodb"] = "unhealthy"

    try:
        qdrant_client = QdrantClient(
            host=settings.QDRANT_HOST,
            port=settings.QDRANT_PORT,
            api_key=settings.QDRANT_API_KEY or None,
        )
        qdrant_client.get_collections()
    except Exception:
        services["qdrant"] = "unhealthy"

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
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        log_level="info",
    )

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.core.config import settings
from app.api.mem0_routes import router as memories_router
from app.api.user_routes import router as users_router
from app.database.mongodb import mongodb


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan events."""
    # Startup
    print(f"Starting {settings.APP_NAME} v{settings.APP_VERSION}")
    print(f"Qdrant: {settings.QDRANT_HOST}:{settings.QDRANT_PORT}")
    print(f"MongoDB: {settings.MONGO_URI}")
    print(f"OpenAI: {settings.OPENAI_MODEL} + {settings.OPENAI_EMBEDDING_MODEL}")

    # Connect to MongoDB
    await mongodb.connect()

    yield

    # Shutdown
    print("Shutting down...")
    await mongodb.disconnect()


# Create FastAPI app
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="A user memory management API powered by mem0 + Qdrant + OpenAI + MongoDB",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
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
            "users": "/api/v1/users",
            "memories": "/api/v1/memories",
        },
        "features": [
            "User profile management (MongoDB)",
            "Semantic memory storage (Qdrant + mem0)",
            "Smart memory search",
            "Batch operations",
            "Conversation memory extraction",
        ]
    }


@app.get("/health")
async def health():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "services": {
            "mem0": "running",
            "vector_store": "qdrant",
            "database": "mongodb",
            "llm": "openai",
        }
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )

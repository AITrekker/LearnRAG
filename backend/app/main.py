from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

from app.core.database import init_db
from app.routers import auth, tenants, embeddings, rag
from app.services.tenant_service import TenantService


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await init_db()
    # Auto-discover tenants from demo_data
    tenant_service = TenantService()
    await tenant_service.auto_discover_tenants()
    yield
    # Shutdown
    pass


app = FastAPI(
    title="LearnRAG API",
    description="Educational RAG platform for learning embeddings and retrieval techniques",
    version="1.0.0",
    lifespan=lifespan
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(auth.router, prefix="/api/auth", tags=["auth"])
app.include_router(tenants.router, prefix="/api/tenants", tags=["tenants"])
app.include_router(embeddings.router, prefix="/api/embeddings", tags=["embeddings"])
app.include_router(rag.router, prefix="/api/rag", tags=["rag"])


@app.get("/")
async def root():
    return {"message": "LearnRAG API is running!"}


@app.get("/health")
async def health():
    return {"status": "healthy"}
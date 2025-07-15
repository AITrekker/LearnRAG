from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
import logging
import time

from database import init_db
from api import auth, tenants, embeddings, rag
from services.tenant_service import TenantService


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await init_db()
    # Auto-discover tenants from setup
    tenant_service = TenantService()
    await tenant_service.auto_discover_tenants()
    
    print("🚀 LearnRAG API startup complete!")
    
    yield
    # Shutdown
    pass


app = FastAPI(
    title="LearnRAG API",
    description="Educational RAG platform for learning embeddings and retrieval techniques",
    version="1.0.0",
    lifespan=lifespan
)

@app.middleware("http")
async def emoji_logging_middleware(request: Request, call_next):
    start_time = time.time()
    
    method_emoji = {
        "GET": "🔍",
        "POST": "📝", 
        "PUT": "✏️",
        "DELETE": "🗑️",
        "PATCH": "🔧"
    }.get(request.method, "📡")
    
    response = await call_next(request)
    duration = time.time() - start_time
    
    status_emoji = "✅" if response.status_code < 300 else "🔄" if response.status_code < 400 else "❌" if response.status_code < 500 else "💥"
    
    print(f"{method_emoji} {request.method} {request.url.path} {status_emoji} {response.status_code} ({duration:.3f}s)")
    
    return response


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
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
import time
import traceback

from database import init_db
from api import auth, tenants, embeddings, rag
from services.tenant_service import TenantService
from exceptions import LearnRAGException, ErrorResponse, ErrorCode, ErrorType


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

# Global exception handlers
@app.exception_handler(LearnRAGException)
async def learnrag_exception_handler(request: Request, exc: LearnRAGException):
    """Handle custom LearnRAG exceptions with standardized responses"""
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.error_response.dict()
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Convert FastAPI HTTPExceptions to standardized format"""
    error_response = ErrorResponse(
        error_code="HTTP_ERROR",
        error_type=ErrorType.SYSTEM if exc.status_code >= 500 else ErrorType.VALIDATION,
        message=exc.detail if isinstance(exc.detail, str) else "Request failed",
        details={"original_detail": exc.detail} if not isinstance(exc.detail, str) else None
    )
    return JSONResponse(
        status_code=exc.status_code,
        content=error_response.dict()
    )

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Handle unexpected exceptions with safe error messages"""
    # Log the full error for debugging
    logging.error(f"Unhandled exception: {exc}", exc_info=True)
    
    error_response = ErrorResponse(
        error_code=ErrorCode.DATABASE_ERROR if "database" in str(exc).lower() else "INTERNAL_ERROR",
        error_type=ErrorType.SYSTEM,
        message="An unexpected error occurred. Please try again.",
        details={"error_type": type(exc).__name__} if logging.getLogger().level <= logging.DEBUG else None
    )
    return JSONResponse(
        status_code=500,
        content=error_response.dict()
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
from config import CORS_ORIGINS

app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
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
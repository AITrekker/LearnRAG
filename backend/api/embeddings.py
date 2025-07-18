"""
Embeddings API - Vector Generation and Management Endpoints

Teaching Purpose: This API demonstrates embedding lifecycle management:

1. EMBEDDING GENERATION: Convert text documents into searchable vectors
2. STATUS MONITORING: Track embedding progress and completion
3. MODEL MANAGEMENT: Select and configure different embedding models
4. BACKGROUND PROCESSING: Handle long-running embedding tasks
5. METRICS COLLECTION: Track performance and usage patterns

Core API Concepts Illustrated:
- RESTful endpoints for embedding operations
- Background task processing for long operations
- Status endpoints for real-time progress tracking
- Model configuration and selection APIs
- Tenant-specific embedding management
"""

from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete, func
from typing import List

from database import get_db, AsyncSessionLocal
from models import Tenant, File, Embedding
from models import (
    GenerateEmbeddingsRequest, 
    GenerateEmbeddingsResponse,
    EmbeddingStatus,
    GeneralEmbeddingStatus
)
from api.auth import get_current_tenant
from services.embedding_service import EmbeddingService
from services.metrics_service import MetricsService
from config import AVAILABLE_EMBEDDING_MODELS, CHUNKING_STRATEGIES

router = APIRouter()


async def _generate_embeddings_background(
    files: List[File],
    embedding_model: str,
    chunking_strategy: str,
    chunk_size: int,
    chunk_overlap: int,
    tenant_name: str
):
    """
    Background task for embedding generation with proper database session management.
    
    Teaching Purpose: Demonstrates proper async database session handling in background tasks.
    - Create new session within background task scope
    - Ensure session is properly closed after completion
    - Avoid session sharing between request and background task
    """
    embedding_service = EmbeddingService()
    
    # Create a new database session for the background task
    async with AsyncSessionLocal() as db:
        try:
            await embedding_service.generate_embeddings_for_files(
                files, embedding_model, chunking_strategy, db,
                chunk_size, chunk_overlap, tenant_name
            )
            await db.commit()
        except ValueError as e:
            # Handle validation errors (like invalid model names) gracefully
            await db.rollback()
            print(f"Validation error in background embedding generation: {e}")
            print(f"Skipping embedding generation for tenant {tenant_name} due to invalid parameters")
        except Exception as e:
            # Handle any other errors
            await db.rollback()
            print(f"Unexpected error in background embedding generation: {e}")
            print(f"Failed to generate embeddings for tenant {tenant_name}")
        finally:
            await db.close()


@router.post("/generate", response_model=GenerateEmbeddingsResponse)
async def generate_embeddings(
    request: GenerateEmbeddingsRequest,
    background_tasks: BackgroundTasks,
    tenant: Tenant = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db)
):
    """Generate embeddings for tenant files using current settings or request params"""
    from models import TenantEmbeddingSettings
    
    # Get tenant settings if no specific params provided in request
    settings_result = await db.execute(
        select(TenantEmbeddingSettings).where(TenantEmbeddingSettings.tenant_id == tenant.id)
    )
    settings = settings_result.scalar_one_or_none()
    
    # Use request params or fall back to tenant settings, then to config defaults
    from config import DEFAULT_EMBEDDING_MODEL, DEFAULT_CHUNKING_STRATEGY, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
    
    embedding_model = request.embedding_model or (settings.embedding_model if settings else None) or DEFAULT_EMBEDDING_MODEL
    chunking_strategy = request.chunking_strategy or (settings.chunking_strategy if settings else None) or DEFAULT_CHUNKING_STRATEGY
    chunk_size = request.chunk_size or (settings.chunk_size if settings else None) or DEFAULT_CHUNK_SIZE
    chunk_overlap = request.chunk_overlap or (settings.chunk_overlap if settings else None) or DEFAULT_CHUNK_OVERLAP
    
    embedding_service = EmbeddingService()
    
    # Get all files for the tenant
    result = await db.execute(
        select(File).where(File.tenant_id == tenant.id)
    )
    files = result.scalars().all()
    
    if not files:
        raise HTTPException(status_code=404, detail="No files found")
    
    # Start background task for embedding generation
    # Note: Create new DB session in background task to avoid connection issues
    background_tasks.add_task(
        _generate_embeddings_background,
        files, embedding_model, chunking_strategy,
        chunk_size, chunk_overlap, tenant.name
    )
    
    return GenerateEmbeddingsResponse(
        message=f"Started embedding generation for {len(files)} files",
        files_processed=len(files),
        total_chunks=0  # Will be updated during processing
    )


@router.get("/models")
async def get_available_models():
    """Get list of available embedding models"""
    return {
        "models": AVAILABLE_EMBEDDING_MODELS
    }


@router.get("/status-summary", response_model=GeneralEmbeddingStatus)
async def get_general_embedding_status(
    tenant: Tenant = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db)
):
    """Get general embedding status summary for the tenant"""
    # Get total file count
    total_files_result = await db.execute(
        select(func.count(File.id)).where(File.tenant_id == tenant.id)
    )
    total_files = total_files_result.scalar() or 0
    
    # Get files with embeddings count
    files_with_embeddings_result = await db.execute(
        select(func.count(func.distinct(Embedding.file_id)))
        .join(File)
        .where(File.tenant_id == tenant.id)
    )
    files_with_embeddings = files_with_embeddings_result.scalar() or 0
    
    # Get total chunks count
    total_chunks_result = await db.execute(
        select(func.count(Embedding.id))
        .join(File)
        .where(File.tenant_id == tenant.id)
    )
    total_chunks = total_chunks_result.scalar() or 0
    
    # Get available models and strategies
    models_result = await db.execute(
        select(func.distinct(Embedding.embedding_model))
        .join(File)
        .where(File.tenant_id == tenant.id)
    )
    available_models = [row[0] for row in models_result.fetchall()]
    
    strategies_result = await db.execute(
        select(func.distinct(Embedding.chunking_strategy))
        .join(File)
        .where(File.tenant_id == tenant.id)
    )
    available_strategies = [row[0] for row in strategies_result.fetchall()]
    
    # Get last updated timestamp
    last_updated_result = await db.execute(
        select(func.max(Embedding.created_at))
        .join(File)
        .where(File.tenant_id == tenant.id)
    )
    last_updated = last_updated_result.scalar()
    
    return GeneralEmbeddingStatus(
        total_files=total_files,
        files_with_embeddings=files_with_embeddings,
        files_without_embeddings=total_files - files_with_embeddings,
        total_chunks=total_chunks,
        available_models=available_models,
        available_strategies=available_strategies,
        last_updated=last_updated
    )


@router.get("/status/{file_id}", response_model=EmbeddingStatus)
async def get_embedding_status(
    file_id: int,
    tenant: Tenant = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db)
):
    """Get embedding status for a specific file"""
    # Check file belongs to tenant
    file_result = await db.execute(
        select(File).where(File.id == file_id, File.tenant_id == tenant.id)
    )
    file = file_result.scalar_one_or_none()
    
    if not file:
        raise HTTPException(status_code=404, detail="File not found")
    
    # Get embedding info
    embeddings_result = await db.execute(
        select(Embedding).where(Embedding.file_id == file_id)
    )
    embeddings = embeddings_result.scalars().all()
    
    if not embeddings:
        return EmbeddingStatus(
            file_id=file_id,
            filename=file.filename,
            has_embeddings=False,
            chunk_count=0,
            embedding_model=None,
            chunking_strategy=None,
            last_updated=None
        )
    
    models = list(set(e.embedding_model for e in embeddings))
    strategies = list(set(e.chunking_strategy for e in embeddings))
    last_updated = max(e.created_at for e in embeddings)
    
    return EmbeddingStatus(
        file_id=file_id,
        filename=file.filename,
        has_embeddings=True,
        chunk_count=len(embeddings),
        embedding_model=models[0] if models else None,
        chunking_strategy=strategies[0] if strategies else None,
        last_updated=last_updated
    )


@router.delete("/{file_id}")
async def delete_embeddings(
    file_id: int,
    tenant: Tenant = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db)
):
    """Delete all embeddings for a specific file"""
    # Check file belongs to tenant
    file_result = await db.execute(
        select(File).where(File.id == file_id, File.tenant_id == tenant.id)
    )
    file = file_result.scalar_one_or_none()
    
    if not file:
        raise HTTPException(status_code=404, detail="File not found")
    
    # Delete embeddings
    await db.execute(delete(Embedding).where(Embedding.file_id == file_id))
    await db.commit()
    
    return {"message": f"Deleted embeddings for file {file.filename}"}


@router.get("/metrics/current")
async def get_current_metrics(
    tenant: Tenant = Depends(get_current_tenant)
):
    """Get current embedding generation metrics if active"""
    metrics_service = MetricsService()
    
    # Check if there's an active session
    if hasattr(metrics_service, 'session_data') and metrics_service.session_data:
        summary = metrics_service.get_session_summary()
        return {
            "active": True,
            "progress": summary
        }
    else:
        return {
            "active": False,
            "progress": None
        }


@router.get("/chunking-strategies")
async def get_chunking_strategies():
    """Get list of available chunking strategies"""
    return {
        "strategies": CHUNKING_STRATEGIES
    }
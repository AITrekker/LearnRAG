from fastapi import APIRouter, Depends, HTTPException, BackgroundTasks
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete
from typing import List

from app.core.database import get_db
from app.models.database import Tenant, File, Embedding
from app.models.schemas import (
    GenerateEmbeddingsRequest, 
    GenerateEmbeddingsResponse,
    EmbeddingStatus
)
from app.routers.auth import get_current_tenant
from app.services.embedding_service import EmbeddingService

router = APIRouter()


@router.post("/generate", response_model=GenerateEmbeddingsResponse)
async def generate_embeddings(
    request: GenerateEmbeddingsRequest,
    background_tasks: BackgroundTasks,
    tenant: Tenant = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db)
):
    """Generate embeddings for tenant files"""
    embedding_service = EmbeddingService()
    
    # Get files to process
    if request.file_ids:
        result = await db.execute(
            select(File).where(
                File.tenant_id == tenant.id,
                File.id.in_(request.file_ids)
            )
        )
    else:
        result = await db.execute(
            select(File).where(File.tenant_id == tenant.id)
        )
    
    files = result.scalars().all()
    
    if not files:
        raise HTTPException(status_code=404, detail="No files found")
    
    # Start background task for embedding generation
    background_tasks.add_task(
        embedding_service.generate_embeddings_for_files,
        files, request.embedding_model, request.chunking_strategy, db
    )
    
    return GenerateEmbeddingsResponse(
        message=f"Started embedding generation for {len(files)} files",
        processed_files=len(files),
        total_chunks=0,  # Will be updated during processing
        embedding_model=request.embedding_model,
        chunking_strategy=request.chunking_strategy
    )


@router.get("/models")
async def get_available_models():
    """Get list of available embedding models"""
    return {
        "models": [
            {
                "name": "sentence-transformers/all-MiniLM-L6-v2",
                "description": "Lightweight, fast model (384 dimensions)",
                "dimension": 384,
                "default": True
            }
        ]
    }


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
            embedding_models=[],
            chunking_strategies=[],
            total_chunks=0,
            last_updated=None
        )
    
    models = list(set(e.embedding_model for e in embeddings))
    strategies = list(set(e.chunking_strategy for e in embeddings))
    last_updated = max(e.created_at for e in embeddings)
    
    return EmbeddingStatus(
        file_id=file_id,
        filename=file.filename,
        has_embeddings=True,
        embedding_models=models,
        chunking_strategies=strategies,
        total_chunks=len(embeddings),
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
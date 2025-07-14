from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from typing import List

from app.core.database import get_db
from app.models.database import Tenant, File, Embedding
from app.models.schemas import TenantInfo, File as FileSchema, GeneralEmbeddingStatus
from app.routers.auth import get_current_tenant
from app.services.tenant_service import TenantService

router = APIRouter()


@router.get("/info", response_model=TenantInfo)
async def get_tenant_info(
    tenant: Tenant = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db)
):
    """Get current tenant information with statistics"""
    # Get file count
    file_count_result = await db.execute(
        select(func.count(File.id)).where(File.tenant_id == tenant.id)
    )
    file_count = file_count_result.scalar() or 0
    
    # Get embedding count
    embedding_count_result = await db.execute(
        select(func.count(Embedding.id))
        .join(File)
        .where(File.tenant_id == tenant.id)
    )
    embedding_count = embedding_count_result.scalar() or 0
    
    return TenantInfo(
        id=tenant.id,
        slug=tenant.slug,
        name=tenant.name,
        file_count=file_count,
        embedding_count=embedding_count,
        created_at=tenant.created_at
    )


@router.post("/sync-files")
async def sync_files(
    force: bool = False,
    tenant: Tenant = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db)
):
    """Sync files from demo_data for current tenant"""
    tenant_service = TenantService()
    if force:
        # Force re-sync all files
        result = await tenant_service.sync_tenant_files(tenant.slug, db)
    else:
        # Only sync new/changed files (delta sync)
        result = await tenant_service.sync_tenant_files(tenant.slug, db)
    return result


@router.get("/files", response_model=List[FileSchema])
async def get_tenant_files(
    tenant: Tenant = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db)
):
    """Get all files for current tenant"""
    result = await db.execute(
        select(File).where(File.tenant_id == tenant.id).order_by(File.filename)
    )
    files = result.scalars().all()
    return files


@router.get("/stats")
async def get_tenant_stats(
    tenant: Tenant = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db)
):
    """Get detailed statistics for current tenant"""
    # File statistics
    file_stats_result = await db.execute(
        select(
            func.count(File.id).label("total_files"),
            func.sum(File.file_size).label("total_size"),
            func.count(func.distinct(File.content_type)).label("content_types")
        ).where(File.tenant_id == tenant.id)
    )
    file_stats = file_stats_result.one()
    
    # Embedding statistics
    embedding_stats_result = await db.execute(
        select(
            func.count(Embedding.id).label("total_chunks"),
            func.count(func.distinct(Embedding.embedding_model)).label("models_used"),
            func.count(func.distinct(Embedding.chunking_strategy)).label("strategies_used")
        )
        .join(File)
        .where(File.tenant_id == tenant.id)
    )
    embedding_stats = embedding_stats_result.one()
    
    return {
        "files": {
            "total_files": file_stats.total_files or 0,
            "total_size_bytes": file_stats.total_size or 0,
            "content_types": file_stats.content_types or 0
        },
        "embeddings": {
            "total_chunks": embedding_stats.total_chunks or 0,
            "models_used": embedding_stats.models_used or 0,
            "strategies_used": embedding_stats.strategies_used or 0
        }
    }


@router.get("/embedding-summary", response_model=GeneralEmbeddingStatus)
async def get_embedding_summary(
    tenant: Tenant = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db)
):
    """Get embedding summary for the tenant"""
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
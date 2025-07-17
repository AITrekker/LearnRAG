"""
Tenants API - Multi-Tenant Management and Statistics

This module demonstrates multi-tenant architecture and data management:

1. TENANT INFORMATION: Get tenant metadata and statistics
2. FILE MANAGEMENT: Sync and track files per tenant
3. EMBEDDING SETTINGS: Configure RAG parameters per tenant
4. STATISTICS TRACKING: Monitor usage and performance metrics
5. DATA ISOLATION: Ensure tenant data separation and security

Core Multi-Tenant Concepts Illustrated:
- Tenant-scoped database queries using foreign key relationships
- File synchronization and tracking per tenant
- Configurable embedding settings with change detection
- Statistics aggregation using SQL functions
- Cascade deletion patterns for data consistency
"""

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func
from typing import List

from database import get_db
from models import Tenant, File, Embedding, TenantEmbeddingSettings
from models import TenantInfo, FileInfo, GeneralEmbeddingStatus, EmbeddingSettingsResponse, EmbeddingSettingsRequest
from api.auth import get_current_tenant
from services.tenant_service import TenantService
from config import DEFAULT_EMBEDDING_MODEL, DEFAULT_CHUNKING_STRATEGY, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP

router = APIRouter()


@router.get("/info", response_model=TenantInfo)
async def get_tenant_info(
    tenant: Tenant = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db)
):
    """
    Get current tenant information with statistics - Tenant Overview
    
    WHY TENANT INFO ENDPOINT?
    - Provides dashboard overview of tenant data
    - Shows file count and embedding status at a glance
    - Enables UI to display tenant context and activity
    - Supports administrative monitoring and debugging
    
    STATISTICS COMPUTED:
    - File count: Total files uploaded for this tenant
    - Embedding count: Total vector chunks generated
    - Creation timestamp: When tenant was first created
    - Tenant metadata: Name, slug, and identifiers
    """
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
    source: str = "data",
    force: bool = False,
    tenant: Tenant = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db)
):
    """
    Sync files for current tenant - File Management
    
    WHY FILE SYNCHRONIZATION?
    - Keeps database in sync with filesystem changes
    - Supports both initial seeding and runtime updates
    - Handles file additions, modifications, and deletions
    - Enables hot-reload of content without container restart
    
    SYNC SOURCES:
    - 'data': Runtime sync from internal files directory
    - 'setup': Initial seeding from setup directory
    
    FORCE PARAMETER:
    - false: Only sync changed files (delta sync)
    - true: Re-sync all files regardless of changes
    """
    tenant_service = TenantService()
    result = await tenant_service.sync_tenant_files(tenant.slug, db, source)
    return result


@router.get("/files", response_model=List[FileInfo])
async def get_tenant_files(
    tenant: Tenant = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db)
):
    """
    Get all files for current tenant - File Listing
    
    WHY FILE LISTING?
    - Provides UI with complete file inventory
    - Shows file metadata (size, type, timestamps)
    - Enables file-specific operations and debugging
    - Supports file management and organization features
    
    FILE METADATA INCLUDED:
    - Name and path for identification
    - Size and content type for display
    - Creation and modification timestamps
    - Internal ID for API operations
    """
    result = await db.execute(
        select(File).where(File.tenant_id == tenant.id).order_by(File.filename)
    )
    files = result.scalars().all()
    return [
        FileInfo(
            id=file.id,
            name=file.filename,
            path=file.file_path,
            size=file.file_size,
            type=file.content_type,
            created_at=file.created_at,
            updated_at=file.last_modified
        )
        for file in files
    ]


@router.get("/stats")
async def get_tenant_stats(
    tenant: Tenant = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db)
):
    """
    Get detailed statistics for current tenant - Analytics Dashboard
    
    WHY DETAILED STATISTICS?
    - Provides insights into tenant usage patterns
    - Helps identify performance bottlenecks
    - Supports capacity planning and optimization
    - Enables monitoring and alerting on usage
    
    STATISTICS CATEGORIES:
    - Files: Count, total size, content type diversity
    - Embeddings: Chunk count, model usage, strategy diversity
    - Performance: Processing metrics and efficiency
    """
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
    """
    Get embedding summary for the tenant - Embedding Status Overview
    
    WHY EMBEDDING SUMMARY?
    - Shows embedding completion status across all files
    - Identifies which files need embedding generation
    - Displays model and strategy usage patterns
    - Provides timestamp of last embedding activity
    
    SUMMARY METRICS:
    - File coverage: Files with vs without embeddings
    - Chunk statistics: Total vectors generated
    - Model diversity: Which embedding models are in use
    - Strategy diversity: Which chunking strategies are active
    """
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


@router.get("/embedding-settings", response_model=EmbeddingSettingsResponse)
async def get_embedding_settings(
    tenant: Tenant = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db)
):
    """
    Get current embedding settings for the tenant - Configuration Management
    
    WHY EMBEDDING SETTINGS?
    - Allows per-tenant customization of RAG parameters
    - Enables A/B testing of different embedding models
    - Supports tenant-specific optimization requirements
    - Provides defaults for new tenants
    
    SETTINGS INCLUDED:
    - embedding_model: Which neural network to use
    - chunking_strategy: How to split documents
    - chunk_size: Size of text chunks in tokens
    - chunk_overlap: Overlap between chunks for context
    """
    # Get existing settings or create default
    settings_result = await db.execute(
        select(TenantEmbeddingSettings).where(TenantEmbeddingSettings.tenant_id == tenant.id)
    )
    settings = settings_result.scalar_one_or_none()
    
    if not settings:
        # Create default settings if none exist
        settings = TenantEmbeddingSettings(
            tenant_id=tenant.id,
            embedding_model=DEFAULT_EMBEDDING_MODEL,
            chunking_strategy=DEFAULT_CHUNKING_STRATEGY,
            chunk_size=DEFAULT_CHUNK_SIZE,
            chunk_overlap=DEFAULT_CHUNK_OVERLAP
        )
        db.add(settings)
        await db.commit()
        await db.refresh(settings)
    
    return settings


@router.post("/embedding-settings", response_model=EmbeddingSettingsResponse)
async def update_embedding_settings(
    settings_update: EmbeddingSettingsRequest,
    tenant: Tenant = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db)
):
    """
    Update embedding settings for the tenant - Configuration Update
    
    WHY SETTINGS UPDATE?
    - Allows dynamic reconfiguration of RAG parameters
    - Enables experimentation with different models/strategies
    - Supports optimization based on tenant-specific needs
    - Maintains data consistency through cascade operations
    
    CHANGE DETECTION:
    - Compares new settings against existing configuration
    - Triggers embedding regeneration when parameters change
    - Preserves embeddings when settings are unchanged
    - Ensures data consistency across setting changes
    
    CASCADE OPERATIONS:
    - Deletes existing embeddings when settings change
    - Forces regeneration with new parameters
    - Maintains referential integrity
    - Prevents stale embedding data
    """
    from sqlalchemy import delete
    
    # Get existing settings or create new
    settings_result = await db.execute(
        select(TenantEmbeddingSettings).where(TenantEmbeddingSettings.tenant_id == tenant.id)
    )
    settings = settings_result.scalar_one_or_none()
    
    # Check if settings are actually changing
    settings_changed = False
    if settings:
        settings_changed = (
            settings.embedding_model != settings_update.embedding_model or
            settings.chunking_strategy != settings_update.chunking_strategy or
            settings.chunk_size != settings_update.chunk_size or
            settings.chunk_overlap != settings_update.chunk_overlap
        )
        # Check if we need to regenerate embeddings due to model/strategy changes
    
    # If settings changed, delete all existing embeddings for this tenant
    if settings_changed:
        print(f"🗑️  Settings changed for tenant {tenant.name}, deleting existing embeddings...")
        await db.execute(
            delete(Embedding).where(
                Embedding.file_id.in_(
                    select(File.id).where(File.tenant_id == tenant.id)
                )
            )
        )
        print(f"✅ Deleted existing embeddings for tenant {tenant.name}")
    
    if settings:
        # Update existing settings
        settings.embedding_model = settings_update.embedding_model
        settings.chunking_strategy = settings_update.chunking_strategy
        settings.chunk_size = settings_update.chunk_size
        settings.chunk_overlap = settings_update.chunk_overlap
    else:
        # Create new settings
        settings = TenantEmbeddingSettings(
            tenant_id=tenant.id,
            embedding_model=settings_update.embedding_model,
            chunking_strategy=settings_update.chunking_strategy,
            chunk_size=settings_update.chunk_size,
            chunk_overlap=settings_update.chunk_overlap
        )
        db.add(settings)
    
    await db.commit()
    await db.refresh(settings)
    
    return settings
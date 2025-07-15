"""
Shared utilities to eliminate code duplication across services
"""
from pathlib import Path
from typing import Optional
import hashlib
import mimetypes
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select
from models import Tenant
from config import INTERNAL_FILES_DIR, DEMO_DATA_DIR, MODELS_CACHE_DIR, OUTPUT_DIR

class PathConfig:
    """Centralized path configuration using environment variables"""
    INTERNAL_FILES = INTERNAL_FILES_DIR
    DEMO_DATA = DEMO_DATA_DIR
    MODELS_CACHE = MODELS_CACHE_DIR
    OUTPUT = OUTPUT_DIR

class DatabaseUtils:
    """Common database query patterns"""
    
    @staticmethod
    async def get_tenant_by_id(db: AsyncSession, tenant_id: int) -> Optional[Tenant]:
        result = await db.execute(select(Tenant).where(Tenant.id == tenant_id))
        return result.scalar_one_or_none()
    
    @staticmethod
    async def get_tenant_by_slug(db: AsyncSession, slug: str) -> Optional[Tenant]:
        result = await db.execute(select(Tenant).where(Tenant.slug == slug))
        return result.scalar_one_or_none()

class FileUtils:
    """File operation utilities"""
    
    @staticmethod
    def calculate_file_hash(file_path: Path) -> str:
        """Calculate SHA-256 hash of file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    @staticmethod
    def get_content_type(file_path: Path) -> str:
        """Get content type from file extension"""
        content_type, _ = mimetypes.guess_type(str(file_path))
        return content_type or "application/octet-stream"
    
    @staticmethod
    def ensure_tenant_directory(tenant_slug: str) -> Path:
        """Ensure tenant directory exists and return path"""
        tenant_dir = PathConfig.INTERNAL_FILES / tenant_slug
        tenant_dir.mkdir(parents=True, exist_ok=True)
        return tenant_dir
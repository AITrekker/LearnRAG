import os
import shutil
import json
import hashlib
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import secrets

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func, delete

from database import AsyncSessionLocal
from models import Tenant, File, TenantEmbeddingSettings
from utils import PathConfig, DatabaseUtils, FileUtils


class TenantService:
    def __init__(self):
        self.demo_data_dir = PathConfig.DEMO_DATA
        self.internal_files_dir = PathConfig.INTERNAL_FILES
        self.internal_files_dir.mkdir(exist_ok=True)

    async def auto_discover_tenants(self):
        """Auto-discover tenants from setup directory - only if database is empty"""
        async with AsyncSessionLocal() as db:
            # Check if database already has tenants
            tenant_count = await self._get_tenant_count(db)
            
            if tenant_count > 0:
                print(f"Database already has {tenant_count} tenants. Skipping auto-discovery.")
                await self._write_api_keys_to_file(db)
                return

            print("Database is empty. Starting tenant auto-discovery...")
            await self._discover_and_create_tenants(db)

    async def _get_tenant_count(self, db: AsyncSession) -> int:
        """Get count of existing tenants"""
        result = await db.execute(select(func.count(Tenant.id)))
        return result.scalar() or 0

    async def _discover_and_create_tenants(self, db: AsyncSession):
        """Discover tenant folders and create tenants"""
        if not self.demo_data_dir.exists():
            print(f"Setup directory {self.demo_data_dir} does not exist")
            return

        tenant_folders = [
            folder for folder in self.demo_data_dir.iterdir()
            if folder.is_dir() and not folder.name.startswith('.')
        ]

        if not tenant_folders:
            print("No tenant folders found in setup directory")
            return

        for folder in tenant_folders:
            await self._create_or_update_tenant(folder.name, db, force_sync=True)

        await db.commit()
        await self._write_api_keys_to_file(db)
        print("Auto-discovery complete.")

    async def _write_api_keys_to_file(self, db: AsyncSession):
        """Write tenant API keys to JSON files"""
        result = await db.execute(select(Tenant))
        tenants = result.scalars().all()
        
        api_keys_data = {
            "tenants": [
                {
                    "slug": tenant.slug,
                    "name": tenant.name,
                    "api_key": tenant.api_key
                }
                for tenant in tenants
            ]
        }
        
        # Write to both locations for backend and frontend access
        for path in [
            PathConfig.OUTPUT / "api_keys.json",
            PathConfig.OUTPUT / "frontend/public/api_keys.json"
        ]:
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                json.dump(api_keys_data, f, indent=2)
        

    async def _create_or_update_tenant(self, slug: str, db: AsyncSession, force_sync: bool = False):
        """Create or update a tenant"""
        # Check if tenant already exists
        result = await db.execute(select(Tenant).where(Tenant.slug == slug))
        tenant = result.scalar_one_or_none()

        if not tenant:
            # Create new tenant
            api_key = self._generate_api_key()
            tenant = Tenant(
                slug=slug,
                name=slug.replace('_', ' ').replace('-', ' ').title(),
                api_key=api_key
            )
            db.add(tenant)
            await db.flush()  # Get the tenant ID
            
            # Create default embedding settings for new tenant
            default_settings = TenantEmbeddingSettings(
                tenant_id=tenant.id,
                embedding_model="sentence-transformers/all-MiniLM-L6-v2",
                chunking_strategy="fixed_size",
                chunk_size=512,
                chunk_overlap=50
            )
            db.add(default_settings)
            print(f"Created tenant: {slug} with API key: {api_key}")
            
            # Always seed files from setup folder for new tenants
            await self.sync_tenant_files(slug, db, "setup")
        elif force_sync:
            # Only sync files for existing tenants if explicitly requested
            print(f"Force syncing files for existing tenant: {slug}")
            await self.sync_tenant_files(slug, db)
        else:
            print(f"Skipping file sync for existing tenant: {slug} (use force_sync=True to override)")

        return tenant

    def _generate_api_key(self) -> str:
        """Generate a secure API key"""
        return f"lr_{secrets.token_urlsafe(32)}"


    async def sync_tenant_files(self, tenant_slug: str, db: AsyncSession, source_folder: str = "data") -> Dict[str, Any]:
        """Sync files from specified folder (setup for seeding, data for runtime sync)"""
        if source_folder == "setup":
            source_dir = self.demo_data_dir / tenant_slug
            target_dir = self.internal_files_dir / tenant_slug
            target_dir.mkdir(exist_ok=True)
            copy_files = True
        else:
            source_dir = self.internal_files_dir / tenant_slug
            target_dir = source_dir
            copy_files = False
            
        if not source_dir.exists():
            return {"error": f"{source_folder.title()} folder for tenant {tenant_slug} not found"}

        # Get tenant
        result = await db.execute(select(Tenant).where(Tenant.slug == tenant_slug))
        tenant = result.scalar_one_or_none()
        if not tenant:
            return {"error": f"Tenant {tenant_slug} not found in database"}

        processed_files = 0
        new_files = 0
        updated_files = 0
        deleted_files = 0

        # Get current files in source folder
        source_files = {}
        for file_path in source_dir.rglob("*"):
            if file_path.is_file() and not file_path.name.startswith('.'):
                relative_path = file_path.relative_to(source_dir)
                source_files[str(relative_path)] = file_path

        # Get current files in database
        db_files_result = await db.execute(
            select(File).where(File.tenant_id == tenant.id)
        )
        db_files = {file.file_path: file for file in db_files_result.scalars().all()}

        # Process files in source folder
        for relative_path_str, file_path in source_files.items():
            if relative_path_str in db_files:
                # File exists in both - check for updates
                result = await self._process_existing_file(file_path, db_files[relative_path_str], db)
                if result["action"] == "updated":
                    updated_files += 1
            else:
                # New file - add to database (no embeddings)
                await self._process_new_file(file_path, tenant, target_dir, db, copy_files)
                new_files += 1
            processed_files += 1

        # Handle deleted files (in DB but not in source folder) - only for data folder sync
        if not copy_files:
            for file_path, db_file in db_files.items():
                if file_path not in source_files:
                    await self._process_deleted_file(db_file, db)
                    deleted_files += 1

        await db.commit()

        return {
            "tenant": tenant_slug,
            "processed_files": processed_files,
            "new_files": new_files,
            "updated_files": updated_files,
            "deleted_files": deleted_files
        }


    async def _process_existing_file(self, file_path: Path, db_file: File, db: AsyncSession) -> Dict[str, str]:
        """Process a file that exists in both data folder and database"""
        # Calculate current file hash
        file_hash = self._calculate_file_hash(file_path)
        file_size = file_path.stat().st_size
        last_modified = datetime.fromtimestamp(file_path.stat().st_mtime)
        
        # Check if file has changed
        if db_file.file_hash != file_hash or db_file.last_modified != last_modified:
            # File has been updated - clean up old embeddings
            await self._cleanup_embeddings_for_file(db_file.id, db)
            
            # Update file record
            db_file.file_hash = file_hash
            db_file.file_size = file_size
            db_file.last_modified = last_modified
            db_file.content_type = self._get_content_type(file_path)
            
            return {"action": "updated"}
        else:
            return {"action": "unchanged"}
    
    async def _process_new_file(self, file_path: Path, tenant: Tenant, target_dir: Path, db: AsyncSession, copy_file: bool = False) -> Dict[str, str]:
        """Process a new file found in source folder"""
        # Calculate file metadata
        file_hash = self._calculate_file_hash(file_path)
        file_size = file_path.stat().st_size
        last_modified = datetime.fromtimestamp(file_path.stat().st_mtime)
        if copy_file:
            # For setup -> data copy, get relative path from setup folder
            relative_path = file_path.relative_to(self.demo_data_dir / tenant.slug)
        else:
            # For data folder sync, get relative path from data folder
            relative_path = file_path.relative_to(target_dir)
        
        # Copy file if needed (setup -> data)
        if copy_file:
            target_file_path = target_dir / relative_path
            target_file_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file_path, target_file_path)
        content_type = self._get_content_type(file_path)
        
        # Create new file record (no embeddings generated)
        new_file = File(
            tenant_id=tenant.id,
            filename=file_path.name,
            file_path=str(relative_path),
            file_hash=file_hash,
            file_size=file_size,
            content_type=content_type,
            last_modified=last_modified
        )
        db.add(new_file)
        
        return {"action": "new"}
    
    async def _process_deleted_file(self, db_file: File, db: AsyncSession) -> Dict[str, str]:
        """Process a file that was deleted from data folder"""
        # Clean up embeddings
        await self._cleanup_embeddings_for_file(db_file.id, db)
        
        # Remove file record
        db.delete(db_file)
        
        return {"action": "deleted"}
    
    async def _cleanup_embeddings_for_file(self, file_id: int, db: AsyncSession) -> int:
        """Delete all embeddings for a specific file"""
        from models import Embedding
        from sqlalchemy import delete
        
        # Count embeddings before deletion for logging
        count_result = await db.execute(
            select(func.count(Embedding.id)).where(Embedding.file_id == file_id)
        )
        embedding_count = count_result.scalar() or 0
        
        # Delete all embeddings for this file
        if embedding_count > 0:
            await db.execute(
                delete(Embedding).where(Embedding.file_id == file_id)
            )
            print(f"Deleted {embedding_count} embeddings for file_id {file_id}")
        
        return embedding_count

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file"""
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

    def _get_content_type(self, file_path: Path) -> str:
        """Determine content type based on file extension"""
        extension = file_path.suffix.lower()
        content_types = {
            '.txt': 'text/plain',
            '.pdf': 'application/pdf',
            '.doc': 'application/msword',
            '.docx': 'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
            '.xls': 'application/vnd.ms-excel',
            '.xlsx': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet',
            '.ppt': 'application/vnd.ms-powerpoint',
            '.pptx': 'application/vnd.openxmlformats-officedocument.presentationml.presentation',
        }
        return content_types.get(extension, 'application/octet-stream')
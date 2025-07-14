import os
import hashlib
import shutil
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any
import secrets

from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from app.core.database import AsyncSessionLocal
from app.models.database import Tenant, File, TenantEmbeddingSettings


class TenantService:
    def __init__(self):
        self.demo_data_dir = Path("/app/demo_data")
        self.internal_files_dir = Path("/app/internal_files")
        self.internal_files_dir.mkdir(exist_ok=True)

    async def auto_discover_tenants(self):
        """Auto-discover tenants from demo_data directory - only if database is empty"""
        async with AsyncSessionLocal() as db:
            # Check if database already has tenants
            tenant_count_result = await db.execute(select(func.count(Tenant.id)))
            tenant_count = tenant_count_result.scalar()
            
            if tenant_count > 0:
                print(f"Database already has {tenant_count} tenants. Skipping auto-discovery.")
                # Still write existing API keys to file for frontend
                await self._write_existing_api_keys_to_file(db)
                return

            print("Database is empty. Starting tenant auto-discovery...")
            
            if not self.demo_data_dir.exists():
                print(f"Demo data directory {self.demo_data_dir} does not exist")
                return

            api_keys_data = {"tenants": []}
            
            tenant_folders = [
                folder for folder in self.demo_data_dir.iterdir()
                if folder.is_dir() and not folder.name.startswith('.')
            ]

            if not tenant_folders:
                print("No tenant folders found in demo_data directory")
                return

            for folder in tenant_folders:
                tenant = await self._create_or_update_tenant(folder.name, db, force_sync=True)
                if tenant:
                    api_keys_data["tenants"].append({
                        "slug": tenant.slug,
                        "name": tenant.name,
                        "api_key": tenant.api_key
                    })

            await db.commit()
            
            # Write API keys to JSON file accessible by frontend
            api_keys_file = Path("/app/output") / "api_keys.json"
            with open(api_keys_file, "w") as f:
                json.dump(api_keys_data, f, indent=2)
            print(f"Auto-discovery complete. API keys written to {api_keys_file}")

    async def _write_existing_api_keys_to_file(self, db: AsyncSession):
        """Write existing tenant API keys to JSON file"""
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
        
        api_keys_file = Path("/app/output") / "api_keys.json"
        with open(api_keys_file, "w") as f:
            json.dump(api_keys_data, f, indent=2)
        print(f"Existing API keys written to {api_keys_file}")

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
            
            # Always sync files for new tenants
            await self.sync_tenant_files(slug, db)
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

    async def sync_tenant_files(self, tenant_slug: str, db: AsyncSession) -> Dict[str, Any]:
        """Sync files for a specific tenant"""
        tenant_folder = self.demo_data_dir / tenant_slug
        if not tenant_folder.exists():
            return {"error": f"Tenant folder {tenant_slug} not found"}

        # Get tenant
        result = await db.execute(select(Tenant).where(Tenant.slug == tenant_slug))
        tenant = result.scalar_one_or_none()
        if not tenant:
            return {"error": f"Tenant {tenant_slug} not found in database"}

        # Create tenant internal directory
        tenant_internal_dir = self.internal_files_dir / tenant_slug
        tenant_internal_dir.mkdir(exist_ok=True)

        processed_files = 0
        new_files = 0
        updated_files = 0

        # Process all files in tenant folder
        for file_path in tenant_folder.rglob("*"):
            if file_path.is_file() and not file_path.name.startswith('.'):
                result = await self._process_file(file_path, tenant, tenant_internal_dir, db)
                if result["action"] == "new":
                    new_files += 1
                elif result["action"] == "updated":
                    updated_files += 1
                processed_files += 1

        await db.commit()

        return {
            "tenant": tenant_slug,
            "processed_files": processed_files,
            "new_files": new_files,
            "updated_files": updated_files
        }

    async def _process_file(
        self, 
        file_path: Path, 
        tenant: Tenant, 
        tenant_internal_dir: Path, 
        db: AsyncSession
    ) -> Dict[str, str]:
        """Process a single file"""
        # Calculate file hash
        file_hash = self._calculate_file_hash(file_path)
        file_size = file_path.stat().st_size
        last_modified = datetime.fromtimestamp(file_path.stat().st_mtime)
        
        # Relative path within tenant folder
        relative_path = file_path.relative_to(self.demo_data_dir / tenant.slug)
        
        # Check if file exists in database
        result = await db.execute(
            select(File).where(
                File.tenant_id == tenant.id,
                File.file_path == str(relative_path)
            )
        )
        existing_file = result.scalar_one_or_none()

        # Determine content type
        content_type = self._get_content_type(file_path)

        # Copy file to internal storage
        internal_file_path = tenant_internal_dir / relative_path
        internal_file_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, internal_file_path)

        if existing_file:
            # Check if file has changed
            if existing_file.file_hash != file_hash or existing_file.last_modified != last_modified:
                # Update existing file
                existing_file.file_hash = file_hash
                existing_file.file_size = file_size
                existing_file.last_modified = last_modified
                return {"action": "updated"}
            else:
                return {"action": "unchanged"}
        else:
            # Create new file record
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
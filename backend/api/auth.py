from fastapi import APIRouter, Depends, HTTPException, Header
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from database import get_db
from models import Tenant

router = APIRouter()


async def get_current_tenant(
    x_api_key: str = Header(None, alias="X-API-Key"),
    db: AsyncSession = Depends(get_db)
) -> Tenant:
    """Dependency to get current tenant from API key"""
    if not x_api_key:
        raise HTTPException(status_code=401, detail="API key required")
        
    result = await db.execute(select(Tenant).where(Tenant.api_key == x_api_key))
    tenant = result.scalar_one_or_none()
    
    if not tenant:
        raise HTTPException(status_code=401, detail="Invalid API key")
    
    return tenant


@router.get("/validate")
async def validate_api_key(tenant: Tenant = Depends(get_current_tenant)):
    """Validate API key and return tenant info"""
    return {
        "valid": True,
        "tenant": {
            "id": tenant.id,
            "slug": tenant.slug,
            "name": tenant.name
        }
    }
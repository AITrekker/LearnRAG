from fastapi import APIRouter, Depends, Header
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from database import get_db
from models import Tenant
from exceptions import AuthenticationError, SystemError

router = APIRouter()


async def get_current_tenant(
    x_api_key: str = Header(None, alias="X-API-Key"),
    db: AsyncSession = Depends(get_db)
) -> Tenant:
    """
    Dependency to get current tenant from API key
    
    Teaching Purpose: Demonstrates proper authentication error handling
    - Clear error messages for missing vs invalid API keys
    - Standardized error response format
    - Proper HTTP status codes for different auth failures
    """
    if not x_api_key:
        raise AuthenticationError(
            message="API key is required. Please provide X-API-Key header.",
            details={"header_name": "X-API-Key"}
        )
    
    try:
        result = await db.execute(select(Tenant).where(Tenant.api_key == x_api_key))
        tenant = result.scalar_one_or_none()
        
        if not tenant:
            raise AuthenticationError(
                message="Invalid API key. Please check your credentials.",
                details={"api_key_prefix": x_api_key[:8] + "..." if len(x_api_key) > 8 else "***"}
            )
        
        return tenant
        
    except AuthenticationError:
        raise  # Re-raise authentication errors as-is
    except Exception as e:
        raise SystemError(
            message="Database connection error during authentication",
            error_code="DATABASE_AUTH_ERROR",
            details={"operation": "tenant_lookup"}
        )


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
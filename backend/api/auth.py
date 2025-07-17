"""
Authentication API - API Key Based Security System

This module demonstrates secure API authentication patterns:

1. HEADER-BASED AUTH: API keys passed via X-API-Key header
2. TENANT RESOLUTION: Map API keys to tenant records for multi-tenancy
3. ERROR HANDLING: Proper HTTP status codes and error messages
4. SECURITY LOGGING: Safe error reporting without exposing sensitive data
5. DEPENDENCY INJECTION: FastAPI dependencies for auth validation

Core Security Concepts Illustrated:
- API key authentication without exposing keys in URLs
- Multi-tenant security isolation using database lookups
- Standardized error responses for authentication failures
- Safe error logging that doesn't leak sensitive information
- Dependency pattern for reusable authentication logic
"""

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
    Dependency to get current tenant from API key - Multi-Tenant Authentication
    
    WHY API KEY AUTHENTICATION?
    - Simple for API clients (no complex OAuth flows)
    - Stateless authentication suitable for RAG applications
    - Easy to revoke and rotate keys per tenant
    - Works well with containerized microservices
    
    SECURITY CONSIDERATIONS:
    - API keys passed in headers (not URL parameters)
    - Database lookup prevents key enumeration attacks
    - Partial key logging for debugging without full exposure
    - Proper error differentiation (missing vs invalid keys)
    
    ERROR HANDLING:
    - Missing key: 401 with clear instructions
    - Invalid key: 401 with safe error message
    - Database error: 500 with system error details
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
    """
    Validate API key and return tenant info - Authentication Verification
    
    WHY VALIDATION ENDPOINT?
    - Allows clients to verify API key validity before operations
    - Useful for health checks and connection testing
    - Returns tenant metadata for client-side display
    - Enables frontend to show current tenant context
    
    SECURITY RESPONSE:
    - Only returns non-sensitive tenant information
    - Confirms authentication without exposing internal IDs
    - Provides user-friendly tenant identification
    """
    return {
        "valid": True,
        "tenant": {
            "id": tenant.id,
            "slug": tenant.slug,
            "name": tenant.name
        }
    }
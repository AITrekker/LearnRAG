"""
Standardized exceptions for LearnRAG API

Teaching Purpose: Demonstrates proper error handling patterns in RAG systems:
- Clear error categorization (validation, auth, system, external)
- Consistent error response formats
- User-friendly error messages
- Debugging-friendly error details
"""
from fastapi import HTTPException
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any
from datetime import datetime
import uuid

class ErrorType:
    """Standard error type categories for RAG applications"""
    VALIDATION = "validation"     # User input errors
    AUTHENTICATION = "auth"       # API key, permission errors  
    SYSTEM = "system"            # Internal server errors
    EXTERNAL = "external"        # Third-party service errors
    NOT_FOUND = "not_found"      # Resource not found

class ErrorCode:
    """Standard error codes for programmatic handling"""
    # Authentication errors
    API_KEY_REQUIRED = "API_KEY_REQUIRED"
    INVALID_API_KEY = "INVALID_API_KEY"
    
    # Validation errors
    INVALID_INPUT = "INVALID_INPUT"
    FILE_NOT_FOUND = "FILE_NOT_FOUND"
    MODEL_NOT_SUPPORTED = "MODEL_NOT_SUPPORTED"
    
    # System errors
    DATABASE_ERROR = "DATABASE_ERROR"
    MODEL_LOADING_ERROR = "MODEL_LOADING_ERROR"
    EMBEDDING_GENERATION_ERROR = "EMBEDDING_GENERATION_ERROR"
    
    # External service errors
    LLM_SERVICE_ERROR = "LLM_SERVICE_ERROR"
    VECTOR_SEARCH_ERROR = "VECTOR_SEARCH_ERROR"

class ErrorResponse(BaseModel):
    """Standardized error response model"""
    error_code: str
    error_type: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    request_id: str = Field(default_factory=lambda: uuid.uuid4().hex)

class LearnRAGException(HTTPException):
    """Base exception for LearnRAG with standardized error handling"""
    
    def __init__(
        self, 
        status_code: int,
        error_code: str,
        error_type: str,
        message: str,
        details: Optional[Dict[str, Any]] = None
    ):
        self.error_response = ErrorResponse(
            error_code=error_code,
            error_type=error_type,
            message=message,
            details=details
        )
        super().__init__(status_code=status_code, detail=self.error_response.dict())

# Specific exception classes for common RAG scenarios
class AuthenticationError(LearnRAGException):
    """Authentication and authorization errors"""
    def __init__(self, message: str = "Authentication failed", details: Optional[Dict] = None):
        super().__init__(
            status_code=401,
            error_code=ErrorCode.INVALID_API_KEY,
            error_type=ErrorType.AUTHENTICATION,
            message=message,
            details=details
        )

class ValidationError(LearnRAGException):
    """Input validation errors"""
    def __init__(self, message: str, error_code: str = ErrorCode.INVALID_INPUT, details: Optional[Dict] = None):
        super().__init__(
            status_code=400,
            error_code=error_code,
            error_type=ErrorType.VALIDATION,
            message=message,
            details=details
        )

class ResourceNotFoundError(LearnRAGException):
    """Resource not found errors"""
    def __init__(self, message: str, resource_type: str = "resource", details: Optional[Dict] = None):
        super().__init__(
            status_code=404,
            error_code=ErrorCode.FILE_NOT_FOUND,
            error_type=ErrorType.NOT_FOUND,
            message=message,
            details={**(details or {}), "resource_type": resource_type}
        )

class SystemError(LearnRAGException):
    """Internal system errors"""
    def __init__(self, message: str = "Internal system error", error_code: str = "SYSTEM_ERROR", details: Optional[Dict] = None):
        super().__init__(
            status_code=500,
            error_code=error_code,
            error_type=ErrorType.SYSTEM,
            message=message,
            details=details
        )

class ExternalServiceError(LearnRAGException):
    """External service errors (LLM, embeddings, etc.)"""
    def __init__(self, message: str, service_name: str, error_code: str = "EXTERNAL_SERVICE_ERROR", details: Optional[Dict] = None):
        super().__init__(
            status_code=503,
            error_code=error_code,
            error_type=ErrorType.EXTERNAL,
            message=message,
            details={**(details or {}), "service": service_name}
        )
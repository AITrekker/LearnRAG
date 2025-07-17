"""
Standardized Exception System - Production-Ready Error Handling

This module demonstrates comprehensive error handling patterns for RAG applications:

1. ERROR CATEGORIZATION: Clear taxonomy of error types and codes
2. CONSISTENT RESPONSES: Standardized error format across all endpoints
3. USER-FRIENDLY MESSAGES: Clear, actionable error messages
4. DEBUGGING SUPPORT: Detailed error information for troubleshooting
5. TRACEABILITY: Request IDs and timestamps for error tracking

Core Error Handling Concepts Illustrated:
- Exception hierarchy with specialized error types
- HTTP status code mapping to error categories
- Structured error responses with metadata
- Request correlation for distributed system debugging
- Client-friendly error messages with technical details
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
    """
    Standardized error response model - Consistent API Error Format
    
    WHY STANDARDIZED ERRORS?
    - Consistent error handling across all API endpoints
    - Structured format enables programmatic error handling
    - Debugging information without exposing sensitive data
    - Request correlation for distributed system troubleshooting
    
    RESPONSE STRUCTURE:
    - error_code: Machine-readable error identifier
    - error_type: Category for client-side error handling
    - message: Human-readable error description
    - details: Additional context for debugging
    - timestamp: When the error occurred
    - request_id: Unique identifier for error tracking
    """
    error_code: str
    error_type: str
    message: str
    details: Optional[Dict[str, Any]] = None
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())
    request_id: str = Field(default_factory=lambda: uuid.uuid4().hex)

class LearnRAGException(HTTPException):
    """
    Base exception for LearnRAG with standardized error handling - Exception Hierarchy
    
    WHY CUSTOM EXCEPTION BASE CLASS?
    - Ensures consistent error response format across all exceptions
    - Integrates with FastAPI's exception handling system
    - Provides structured error information for debugging
    - Enables centralized error logging and monitoring
    
    INHERITANCE PATTERN:
    - All LearnRAG exceptions inherit from this base class
    - Automatically formats errors according to ErrorResponse model
    - Maintains HTTP status codes and error categorization
    - Preserves FastAPI's exception handling flow
    """
    
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
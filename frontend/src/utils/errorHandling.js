/**
 * Frontend Error Handling Utilities
 * 
 * Teaching Purpose: Demonstrates proper error handling in RAG frontend applications:
 * - Parsing backend error responses
 * - User-friendly error messages
 * - Error categorization for different UI treatments
 * - Debugging support with error details
 */

export const ErrorType = {
  VALIDATION: 'validation',
  AUTHENTICATION: 'auth',
  SYSTEM: 'system',
  EXTERNAL: 'external',
  NOT_FOUND: 'not_found',
  NETWORK: 'network'
};

export const ErrorCode = {
  // Authentication
  API_KEY_REQUIRED: 'API_KEY_REQUIRED',
  INVALID_API_KEY: 'INVALID_API_KEY',
  
  // Validation
  INVALID_INPUT: 'INVALID_INPUT',
  FILE_NOT_FOUND: 'FILE_NOT_FOUND',
  MODEL_NOT_SUPPORTED: 'MODEL_NOT_SUPPORTED',
  
  // System
  DATABASE_ERROR: 'DATABASE_ERROR',
  MODEL_LOADING_ERROR: 'MODEL_LOADING_ERROR',
  EMBEDDING_GENERATION_ERROR: 'EMBEDDING_GENERATION_ERROR',
  
  // External services
  LLM_SERVICE_ERROR: 'LLM_SERVICE_ERROR',
  VECTOR_SEARCH_ERROR: 'VECTOR_SEARCH_ERROR',
  
  // Network
  NETWORK_ERROR: 'NETWORK_ERROR',
  TIMEOUT_ERROR: 'TIMEOUT_ERROR'
};

/**
 * Parse API error response into standardized format
 */
export const parseApiError = (error) => {
  // Network errors (no response received)
  if (!error.response) {
    return {
      error_code: ErrorCode.NETWORK_ERROR,
      error_type: ErrorType.NETWORK,
      message: 'Network connection failed. Please check your internet connection.',
      user_message: 'Connection failed - please try again',
      details: { originalError: error.message },
      severity: 'high'
    };
  }

  const { status, data } = error.response;

  // Handle standardized backend error responses
  if (data && data.error_code) {
    return {
      ...data,
      user_message: getUserFriendlyMessage(data),
      severity: getSeverityLevel(data.error_type, status)
    };
  }

  // Handle legacy/non-standardized responses
  const message = data?.detail || data?.message || 'An unexpected error occurred';
  
  return {
    error_code: getErrorCodeFromStatus(status),
    error_type: getErrorTypeFromStatus(status),
    message,
    user_message: getUserFriendlyMessage({ message, error_type: getErrorTypeFromStatus(status) }),
    details: { legacy_response: data },
    severity: getSeverityLevel(getErrorTypeFromStatus(status), status)
  };
};

/**
 * Get user-friendly error message based on error details
 */
const getUserFriendlyMessage = (errorData) => {
  const { error_code, error_type, message } = errorData;

  // Authentication errors
  if (error_type === ErrorType.AUTHENTICATION) {
    if (error_code === ErrorCode.API_KEY_REQUIRED) {
      return 'Please provide your API key to continue';
    }
    if (error_code === ErrorCode.INVALID_API_KEY) {
      return 'Invalid API key - please check your credentials';
    }
    return 'Authentication failed - please check your credentials';
  }

  // Validation errors - often safe to show directly
  if (error_type === ErrorType.VALIDATION) {
    return message || 'Please check your input and try again';
  }

  // System errors - generic message for security
  if (error_type === ErrorType.SYSTEM) {
    if (error_code === ErrorCode.DATABASE_ERROR) {
      return 'Database is temporarily unavailable - please try again';
    }
    if (error_code === ErrorCode.MODEL_LOADING_ERROR) {
      return 'AI model is loading - please wait a moment and try again';
    }
    return 'Service temporarily unavailable - please try again';
  }

  // External service errors
  if (error_type === ErrorType.EXTERNAL) {
    if (error_code === ErrorCode.LLM_SERVICE_ERROR) {
      return 'AI service is temporarily unavailable - please try again';
    }
    return 'External service error - please try again later';
  }

  // Network errors
  if (error_type === ErrorType.NETWORK) {
    return 'Network connection failed - please check your connection';
  }

  // Default fallback
  return 'Something went wrong - please try again';
};

/**
 * Determine error type from HTTP status code
 */
const getErrorTypeFromStatus = (status) => {
  if (status === 401 || status === 403) return ErrorType.AUTHENTICATION;
  if (status >= 400 && status < 500) return ErrorType.VALIDATION;
  if (status >= 500) return ErrorType.SYSTEM;
  return ErrorType.SYSTEM;
};

/**
 * Determine error code from HTTP status
 */
const getErrorCodeFromStatus = (status) => {
  switch (status) {
    case 401: return ErrorCode.INVALID_API_KEY;
    case 404: return ErrorCode.FILE_NOT_FOUND;
    case 422: return ErrorCode.INVALID_INPUT;
    case 500: return ErrorCode.DATABASE_ERROR;
    case 503: return ErrorCode.LLM_SERVICE_ERROR;
    default: return 'HTTP_ERROR';
  }
};

/**
 * Determine severity level for UI treatment
 */
const getSeverityLevel = (errorType, status) => {
  if (errorType === ErrorType.NETWORK) return 'high';
  if (errorType === ErrorType.AUTHENTICATION) return 'medium';
  if (errorType === ErrorType.SYSTEM) return 'high';
  if (status >= 500) return 'high';
  return 'medium';
};

/**
 * Check if error should trigger a retry
 */
export const shouldRetry = (errorData, retryCount = 0) => {
  const { error_type, error_code } = errorData;
  
  // Don't retry authentication errors
  if (error_type === ErrorType.AUTHENTICATION) return false;
  
  // Don't retry validation errors
  if (error_type === ErrorType.VALIDATION) return false;
  
  // Retry network and system errors up to 3 times
  if (error_type === ErrorType.NETWORK || error_type === ErrorType.SYSTEM) {
    return retryCount < 3;
  }
  
  // Retry specific external service errors
  if (error_code === ErrorCode.LLM_SERVICE_ERROR && retryCount < 2) {
    return true;
  }
  
  return false;
};

/**
 * Get retry delay in milliseconds with exponential backoff
 */
export const getRetryDelay = (retryCount) => {
  return Math.min(1000 * Math.pow(2, retryCount), 10000); // Max 10 seconds
};

/**
 * Log error for debugging (in development) or error reporting (in production)
 */
export const logError = (error, context = {}) => {
  const errorData = parseApiError(error);
  
  console.group(`🚨 Error: ${errorData.error_code}`);
  console.error('Error Type:', errorData.error_type);
  console.error('User Message:', errorData.user_message);
  console.error('Technical Message:', errorData.message);
  if (errorData.details) {
    console.error('Details:', errorData.details);
  }
  if (context && Object.keys(context).length > 0) {
    console.error('Context:', context);
  }
  console.groupEnd();

  // In production, you might send this to an error reporting service
  if (process.env.NODE_ENV === 'production') {
    // Example: Send to error reporting service
    // errorReportingService.report(errorData, context);
  }
};
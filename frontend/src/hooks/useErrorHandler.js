import { useState, useCallback } from 'react';
import { parseApiError, shouldRetry, getRetryDelay, logError } from '../utils/errorHandling';

/**
 * useErrorHandler Hook
 * 
 * Teaching Purpose: Demonstrates centralized error handling in RAG applications:
 * - Consistent error parsing and logging
 * - Automatic retry logic for recoverable errors
 * - Error state management
 * - User notification integration
 */
export const useErrorHandler = () => {
  const [error, setError] = useState(null);
  const [retryCount, setRetryCount] = useState(0);

  const handleError = useCallback((rawError, context = {}) => {
    const parsedError = parseApiError(rawError);
    
    // Log error for debugging
    logError(rawError, {
      ...context,
      retryCount,
      timestamp: new Date().toISOString()
    });

    setError(parsedError);
    return parsedError;
  }, [retryCount]);

  const retry = useCallback(async (retryFunction) => {
    if (!error || !shouldRetry(error, retryCount)) {
      return false;
    }

    const delay = getRetryDelay(retryCount);
    setRetryCount(prev => prev + 1);
    
    // Clear current error while retrying
    setError(null);
    
    try {
      // Wait for exponential backoff delay
      await new Promise(resolve => setTimeout(resolve, delay));
      
      // Execute the retry function
      await retryFunction();
      
      // Reset retry count on success
      setRetryCount(0);
      return true;
    } catch (retryError) {
      // Handle the retry error
      handleError(retryError, { isRetry: true, retryAttempt: retryCount });
      return false;
    }
  }, [error, retryCount, handleError]);

  const clearError = useCallback(() => {
    setError(null);
    setRetryCount(0);
  }, []);

  const canRetry = error ? shouldRetry(error, retryCount) : false;

  return {
    error,
    handleError,
    retry,
    clearError,
    canRetry,
    retryCount
  };
};

/**
 * useApiCall Hook - Enhanced with error handling
 * 
 * Teaching Purpose: Shows how to integrate error handling with API calls
 */
export const useApiCall = () => {
  const [loading, setLoading] = useState(false);
  const { error, handleError, retry, clearError, canRetry } = useErrorHandler();

  const execute = useCallback(async (apiFunction, context = {}) => {
    setLoading(true);
    clearError();

    try {
      const result = await apiFunction();
      setLoading(false);
      return result;
    } catch (err) {
      setLoading(false);
      handleError(err, context);
      throw err; // Re-throw for caller to handle if needed
    }
  }, [handleError, clearError]);

  const retryLastCall = useCallback(async (apiFunction) => {
    if (!canRetry) return false;
    
    setLoading(true);
    const success = await retry(async () => {
      const result = await apiFunction();
      setLoading(false);
      return result;
    });
    
    if (!success) {
      setLoading(false);
    }
    
    return success;
  }, [retry, canRetry]);

  return {
    loading,
    error,
    execute,
    retry: retryLastCall,
    clearError,
    canRetry
  };
};
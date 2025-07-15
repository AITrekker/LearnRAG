import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { AlertCircle, X, RefreshCw } from 'lucide-react';

/**
 * ErrorNotification Component
 * 
 * Teaching Purpose: Demonstrates user-friendly error display in RAG applications:
 * - Visual error categorization (color coding)
 * - Clear, actionable error messages
 * - Retry functionality for recoverable errors
 * - Dismissible notifications
 */
const ErrorNotification = ({ 
  error, 
  onDismiss, 
  onRetry, 
  showRetry = false,
  className = "" 
}) => {
  if (!error) return null;

  const getSeverityStyles = (severity) => {
    switch (severity) {
      case 'high':
        return 'bg-red-50 border-red-200 text-red-800';
      case 'medium':
        return 'bg-yellow-50 border-yellow-200 text-yellow-800';
      default:
        return 'bg-blue-50 border-blue-200 text-blue-800';
    }
  };

  const getIconColor = (severity) => {
    switch (severity) {
      case 'high':
        return 'text-red-500';
      case 'medium':
        return 'text-yellow-500';
      default:
        return 'text-blue-500';
    }
  };

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0, y: -10 }}
        animate={{ opacity: 1, y: 0 }}
        exit={{ opacity: 0, y: -10 }}
        className={`p-4 rounded-lg border ${getSeverityStyles(error.severity)} ${className}`}
      >
        <div className="flex items-start">
          <AlertCircle 
            className={`w-5 h-5 mt-0.5 mr-3 ${getIconColor(error.severity)}`} 
          />
          
          <div className="flex-1 min-w-0">
            <div className="flex items-start justify-between">
              <div>
                <p className="font-medium">
                  {error.user_message || error.message}
                </p>
                
                {error.details && process.env.NODE_ENV === 'development' && (
                  <details className="mt-2">
                    <summary className="text-sm cursor-pointer opacity-75">
                      Technical Details
                    </summary>
                    <pre className="mt-1 text-xs opacity-75 whitespace-pre-wrap">
                      {JSON.stringify(error.details, null, 2)}
                    </pre>
                  </details>
                )}
              </div>
              
              <div className="flex items-center ml-3">
                {showRetry && onRetry && (
                  <motion.button
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={onRetry}
                    className="mr-2 p-1 rounded hover:bg-opacity-20 hover:bg-gray-600 transition-colors"
                    title="Retry"
                  >
                    <RefreshCw className="w-4 h-4" />
                  </motion.button>
                )}
                
                {onDismiss && (
                  <motion.button
                    whileHover={{ scale: 1.05 }}
                    whileTap={{ scale: 0.95 }}
                    onClick={onDismiss}
                    className="p-1 rounded hover:bg-opacity-20 hover:bg-gray-600 transition-colors"
                    title="Dismiss"
                  >
                    <X className="w-4 h-4" />
                  </motion.button>
                )}
              </div>
            </div>
          </div>
        </div>
      </motion.div>
    </AnimatePresence>
  );
};

export default ErrorNotification;
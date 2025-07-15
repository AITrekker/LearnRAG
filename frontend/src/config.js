/**
 * Simple Configuration for LearnRAG Frontend
 * Teaching Purpose: Shows how to handle frontend configuration
 * - Environment variables for build-time configuration
 * - Constants for runtime configuration
 * - Simple and maintainable approach
 */

// API Configuration
export const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';
export const API_TIMEOUT = parseInt(process.env.REACT_APP_API_TIMEOUT || '30000');

// Polling Configuration
export const POLLING_INTERVALS = {
  FAST: parseInt(process.env.REACT_APP_POLLING_FAST || '1000'),
  MEDIUM: parseInt(process.env.REACT_APP_POLLING_MEDIUM || '2000'),
  SLOW: parseInt(process.env.REACT_APP_POLLING_SLOW || '3000')
};

export const POLLING_THRESHOLDS = {
  FAST_TO_MEDIUM: parseInt(process.env.REACT_APP_POLLING_THRESHOLD_1 || '30'),
  MEDIUM_TO_SLOW: parseInt(process.env.REACT_APP_POLLING_THRESHOLD_2 || '90')
};

// Default Values
export const DEFAULTS = {
  EMBEDDING_MODEL: process.env.REACT_APP_DEFAULT_EMBEDDING_MODEL || 'sentence-transformers/all-MiniLM-L6-v2',
  CHUNKING_STRATEGY: process.env.REACT_APP_DEFAULT_CHUNKING_STRATEGY || 'fixed_size',
  CHUNK_SIZE: parseInt(process.env.REACT_APP_DEFAULT_CHUNK_SIZE || '512'),
  CHUNK_OVERLAP: parseInt(process.env.REACT_APP_DEFAULT_CHUNK_OVERLAP || '50'),
  RAG_TECHNIQUE: process.env.REACT_APP_DEFAULT_RAG_TECHNIQUE || 'similarity_search',
  TOP_K: parseInt(process.env.REACT_APP_DEFAULT_TOP_K || '5'),
  MAX_ANSWER_LENGTH: parseInt(process.env.REACT_APP_DEFAULT_MAX_ANSWER_LENGTH || '200'),
  TEMPERATURE: parseFloat(process.env.REACT_APP_DEFAULT_TEMPERATURE || '0.7')
};

// Form Validation Limits
export const VALIDATION_LIMITS = {
  CHUNK_SIZE: {
    MIN: parseInt(process.env.REACT_APP_CHUNK_SIZE_MIN || '50'),
    MAX: parseInt(process.env.REACT_APP_CHUNK_SIZE_MAX || '2048')
  },
  CHUNK_OVERLAP: {
    MIN: parseInt(process.env.REACT_APP_CHUNK_OVERLAP_MIN || '0'),
    MAX: parseInt(process.env.REACT_APP_CHUNK_OVERLAP_MAX || '200')
  },
  TOP_K: {
    MIN: parseInt(process.env.REACT_APP_TOP_K_MIN || '1'),
    MAX: parseInt(process.env.REACT_APP_TOP_K_MAX || '20')
  }
};

// UI Configuration
export const UI_CONFIG = {
  ANIMATIONS: {
    TRANSITION_DURATION: parseInt(process.env.REACT_APP_TRANSITION_DURATION || '300'),
    BOUNCE_SCALE: parseFloat(process.env.REACT_APP_BOUNCE_SCALE || '1.05'),
    FADE_DURATION: parseInt(process.env.REACT_APP_FADE_DURATION || '200')
  },
  NOTIFICATIONS: {
    SUCCESS_DURATION: parseInt(process.env.REACT_APP_SUCCESS_DURATION || '3000'),
    ERROR_DURATION: parseInt(process.env.REACT_APP_ERROR_DURATION || '5000'),
    INFO_DURATION: parseInt(process.env.REACT_APP_INFO_DURATION || '4000')
  }
};

// Feature Flags
export const FEATURES = {
  AUTO_REFRESH: process.env.REACT_APP_AUTO_REFRESH !== 'false',
  REAL_TIME_PROGRESS: process.env.REACT_APP_REAL_TIME_PROGRESS !== 'false',
  SMART_POLLING: process.env.REACT_APP_SMART_POLLING !== 'false',
  ERROR_RECOVERY: process.env.REACT_APP_ERROR_RECOVERY !== 'false'
};

// Error Handling
export const ERROR_CONFIG = {
  MAX_RETRIES: parseInt(process.env.REACT_APP_MAX_RETRIES || '3'),
  RETRY_DELAY: parseInt(process.env.REACT_APP_RETRY_DELAY || '1000'),
  MAX_RETRY_DELAY: parseInt(process.env.REACT_APP_MAX_RETRY_DELAY || '10000')
};

// Cache Configuration
export const CACHE_CONFIG = {
  STALE_TIME: parseInt(process.env.REACT_APP_STALE_TIME || '30000'),
  CACHE_TIME: parseInt(process.env.REACT_APP_CACHE_TIME || '300000')
};
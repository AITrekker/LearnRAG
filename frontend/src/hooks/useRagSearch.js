import { useState, useCallback, useEffect } from 'react';
import { useMutation, useQuery } from 'react-query';
import apiService from '../services/api';
import { useErrorHandler } from './useErrorHandler';
import { DEFAULTS } from '../config';

/**
 * useRagSearch Hook - Centralized RAG Search State Management
 * 
 * WHY CUSTOM HOOKS?
 * - Encapsulates complex search state logic in reusable component
 * - Manages multiple related async operations (search + answer)
 * - Provides clean, predictable interface for UI components
 * - Integrates error handling for robust user experience
 * 
 * RAG STATE MANAGEMENT:
 * - Search parameters affect retrieval quality and performance
 * - State coordination between search and answer generation
 * - Real-time feedback during async RAG operations
 * - Configuration persistence across component re-renders
 */
export const useRagSearch = () => {
  // Search configuration state
  const [searchConfig, setSearchConfig] = useState({
    query: '',
    embedding_model: DEFAULTS.EMBEDDING_MODEL,
    chunking_strategy: DEFAULTS.CHUNKING_STRATEGY,
    rag_technique: DEFAULTS.RAG_TECHNIQUE,
    top_k: DEFAULTS.TOP_K,
    // Generation parameters
    answer_model: DEFAULTS.LLM_MODEL,
    prompt_template: 'factual',
    temperature: DEFAULTS.GENERATION_TEMPERATURE,
    max_length: DEFAULTS.MAX_ANSWER_LENGTH,
    context_chunks: DEFAULTS.CONTEXT_CHUNKS,
    min_similarity: DEFAULTS.MIN_SIMILARITY,
    repetition_penalty: DEFAULTS.REPETITION_PENALTY,
    top_p: DEFAULTS.TOP_P
  });

  // Results state
  const [searchResults, setSearchResults] = useState(null);
  const [answerResult, setAnswerResult] = useState(null);
  
  // Error handling
  const { error, handleError, clearError } = useErrorHandler();

  // Fetch tenant's actual embedding settings to use as defaults
  const { data: tenantSettings } = useQuery(
    'tenantEmbeddingSettings',
    () => apiService.getEmbeddingSettings(),
    {
      onError: (error) => {
        console.warn('Could not fetch tenant embedding settings, using defaults:', error);
      },
      retry: 1
    }
  );

  // Update search config when tenant settings are loaded
  useEffect(() => {
    if (tenantSettings) {
      setSearchConfig(prev => ({
        ...prev,
        embedding_model: tenantSettings.embedding_model || DEFAULTS.EMBEDDING_MODEL,
        chunking_strategy: tenantSettings.chunking_strategy || DEFAULTS.CHUNKING_STRATEGY,
        // Keep other settings as configured by user or defaults
      }));
    }
  }, [tenantSettings]);

  // Search mutation for semantic retrieval
  const searchMutation = useMutation(
    (params) => apiService.search(params),
    {
      onSuccess: (data) => {
        setSearchResults(data);
        clearError();
      },
      onError: (error) => {
        handleError(error, { 
          operation: 'semantic_search',
          query: searchConfig.query,
          model: searchConfig.embedding_model
        });
      }
    }
  );

  // Answer generation mutation for LLM synthesis
  const answerMutation = useMutation(
    (params) => apiService.generateAnswer(params),
    {
      onSuccess: (data) => {
        setAnswerResult(data);
        clearError();
      },
      onError: (error) => {
        handleError(error, {
          operation: 'answer_generation', 
          query: searchConfig.query,
          context_chunks: searchResults?.results?.length || 0
        });
      }
    }
  );

  // Update search configuration
  const updateConfig = useCallback((updates) => {
    setSearchConfig(prev => ({ ...prev, ...updates }));
  }, []);

  // Perform semantic search
  const performSearch = useCallback(async () => {
    if (!searchConfig.query.trim()) return;
    
    // Clear previous results
    setSearchResults(null);
    setAnswerResult(null);
    
    // Execute search
    searchMutation.mutate(searchConfig);
  }, [searchConfig, searchMutation]);

  // Generate answer from search results
  const generateAnswer = useCallback(async () => {
    if (!searchConfig.query.trim()) return;
    
    const answerParams = {
      query: searchConfig.query,
      // Retrieval parameters
      embedding_model: searchConfig.embedding_model,
      chunking_strategy: searchConfig.chunking_strategy,
      top_k: Math.min(searchConfig.top_k * 2, 10), // More context for answers
      min_similarity: searchConfig.min_similarity,
      // Generation parameters
      answer_model: searchConfig.answer_model,
      temperature: searchConfig.temperature,
      max_length: searchConfig.max_length,
      context_chunks: searchConfig.context_chunks,
      repetition_penalty: searchConfig.repetition_penalty,
      top_p: searchConfig.top_p
    };
    
    answerMutation.mutate(answerParams);
  }, [searchConfig, answerMutation]);

  // Combined search and answer workflow
  const searchAndAnswer = useCallback(async () => {
    await performSearch();
    // Answer generation will be triggered after search completes
    if (searchResults?.results?.length > 0) {
      generateAnswer();
    }
  }, [performSearch, generateAnswer, searchResults]);

  // Clear all results
  const clearResults = useCallback(() => {
    setSearchResults(null);
    setAnswerResult(null);
    clearError();
  }, [clearError]);

  return {
    // Configuration
    searchConfig,
    updateConfig,
    
    // Results
    searchResults,
    answerResult,
    
    // Actions
    performSearch,
    generateAnswer, 
    searchAndAnswer,
    clearResults,
    
    // Status
    isSearching: searchMutation.isLoading,
    isGeneratingAnswer: answerMutation.isLoading,
    isLoading: searchMutation.isLoading || answerMutation.isLoading,
    
    // Error handling
    error,
    clearError
  };
};
import { useState, useCallback } from 'react';
import { useMutation } from 'react-query';
import apiService from '../services/api';
import { useErrorHandler } from './useErrorHandler';

/**
 * useRagSearch Hook - Centralized RAG Search State Management
 * 
 * Teaching Purpose: Demonstrates advanced React patterns for RAG applications:
 * - Encapsulating complex search state logic
 * - Managing multiple related async operations (search + answer)
 * - Providing clean, reusable interface for components
 * - Error handling integration for robust UX
 * 
 * RAG Concepts:
 * - Search parameters affect retrieval quality and performance
 * - State coordination between search and answer generation
 * - Real-time feedback during async RAG operations
 */
export const useRagSearch = () => {
  // Search configuration state
  const [searchConfig, setSearchConfig] = useState({
    query: '',
    embedding_model: 'sentence-transformers/all-MiniLM-L6-v2',
    chunking_strategy: 'fixed_size',
    rag_technique: 'similarity_search',
    top_k: 5
  });

  // Results state
  const [searchResults, setSearchResults] = useState(null);
  const [answerResult, setAnswerResult] = useState(null);
  
  // Error handling
  const { error, handleError, clearError } = useErrorHandler();

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
      embedding_model: searchConfig.embedding_model,
      chunking_strategy: searchConfig.chunking_strategy,
      top_k: Math.min(searchConfig.top_k * 2, 10), // More context for answers
      min_similarity: 0.3,
      max_length: 200
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
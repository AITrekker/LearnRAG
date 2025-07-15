import { useState, useCallback, useEffect } from 'react';
import { useMutation, useQuery, useQueryClient } from 'react-query';
import apiService from '../services/api';
import { useErrorHandler } from './useErrorHandler';
import { DEFAULTS, POLLING_INTERVALS, POLLING_THRESHOLDS } from '../config';

/**
 * useEmbeddingGeneration Hook - Advanced State Management for RAG Embedding Pipeline
 * 
 * Teaching Purpose: Demonstrates complex async state management patterns:
 * - Real-time progress tracking for long-running operations
 * - Polling with smart intervals and automatic cleanup
 * - State coordination between configuration, generation, and monitoring
 * - Production-ready error handling and recovery
 * 
 * RAG Concepts:
 * - Embedding generation is compute-intensive and time-consuming
 * - Real-time feedback improves user experience during processing
 * - Configuration changes require regeneration for consistency
 * - Delta sync prevents redundant work in production systems
 */
export const useEmbeddingGeneration = (apiKey) => {
  // Configuration state
  const [config, setConfig] = useState({
    embedding_model: DEFAULTS.EMBEDDING_MODEL,
    chunking_strategy: DEFAULTS.CHUNKING_STRATEGY,
    chunk_size: DEFAULTS.CHUNK_SIZE,
    chunk_overlap: DEFAULTS.CHUNK_OVERLAP
  });

  // Generation state
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [isGenerating, setIsGenerating] = useState(false);
  const [pollCount, setPollCount] = useState(0);
  
  // Error handling
  const { error, handleError, clearError } = useErrorHandler();
  const queryClient = useQueryClient();

  // Optimized metrics polling with smart intervals
  const { data: metrics } = useQuery(
    'currentMetrics',
    apiService.getCurrentMetrics,
    {
      enabled: !!apiKey && isGenerating,
      // Smart polling: faster at start, slower for long operations
      refetchInterval: isGenerating ? (
        pollCount < POLLING_THRESHOLDS.FAST_TO_MEDIUM ? POLLING_INTERVALS.FAST :
        pollCount < POLLING_THRESHOLDS.MEDIUM_TO_SLOW ? POLLING_INTERVALS.MEDIUM :
        POLLING_INTERVALS.SLOW
      ) : false,
      refetchIntervalInBackground: false, // Save battery/CPU when tab not active
      onSuccess: (data) => {
        // Increment poll count for smart interval adjustment
        if (isGenerating) {
          setPollCount(prev => prev + 1);
        }
        
        // Auto-stop polling when generation completes
        if (data && !data.active && isGenerating) {
          setIsGenerating(false);
          setPollCount(0); // Reset poll count
          // Refresh file list to show new embeddings
          queryClient.invalidateQueries('tenantFiles');
        }
      }
    }
  );

  // Available models query
  const { data: models } = useQuery(
    'embeddingModels',
    apiService.getAvailableModels,
    { enabled: !!apiKey, staleTime: 60000 }
  );

  // Available chunking strategies
  const { data: strategies } = useQuery(
    'chunkingStrategies', 
    apiService.getChunkingStrategies,
    { enabled: !!apiKey, staleTime: 60000 }
  );

  // Current embedding settings
  const { data: settings } = useQuery(
    'embeddingSettings',
    apiService.getEmbeddingSettings,
    { enabled: !!apiKey }
  );

  // Sync local config with saved settings
  useEffect(() => {
    if (settings) {
      setConfig({
        embedding_model: settings.embedding_model,
        chunking_strategy: settings.chunking_strategy,
        chunk_size: settings.chunk_size || 512,
        chunk_overlap: settings.chunk_overlap || 50
      });
    }
  }, [settings]);

  // Settings update mutation
  const updateSettingsMutation = useMutation(
    (newSettings) => apiService.updateEmbeddingSettings(newSettings),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('embeddingSettings');
        queryClient.invalidateQueries('tenantFiles');
        clearError();
      },
      onError: (error) => {
        handleError(error, { 
          operation: 'update_settings',
          settings: config
        });
      }
    }
  );

  // Embedding generation mutation
  const generateMutation = useMutation(
    (params) => apiService.generateEmbeddings(params),
    {
      onSuccess: () => {
        setIsGenerating(true); // Start polling for progress
        setPollCount(0); // Reset poll count for new generation
        clearError();
        setSelectedFiles([]); // Clear selection
      },
      onError: (error) => {
        handleError(error, {
          operation: 'generate_embeddings',
          file_count: selectedFiles.length,
          model: config.embedding_model
        });
      }
    }
  );

  // Update configuration
  const updateConfig = useCallback((updates) => {
    setConfig(prev => ({ ...prev, ...updates }));
  }, []);

  // Save settings to backend
  const saveSettings = useCallback(() => {
    updateSettingsMutation.mutate(config);
  }, [config, updateSettingsMutation]);

  // Generate embeddings
  const generateEmbeddings = useCallback(() => {
    const params = {
      ...config,
      file_ids: selectedFiles.length > 0 ? selectedFiles : undefined
    };
    generateMutation.mutate(params);
  }, [config, selectedFiles, generateMutation]);

  // File selection helpers
  const toggleFileSelection = useCallback((fileId) => {
    setSelectedFiles(prev => 
      prev.includes(fileId)
        ? prev.filter(id => id !== fileId)
        : [...prev, fileId]
    );
  }, []);

  const selectAllFiles = useCallback((fileIds) => {
    setSelectedFiles(fileIds);
  }, []);

  const clearFileSelection = useCallback(() => {
    setSelectedFiles([]);
  }, []);

  // Progress information
  const progress = metrics?.progress || null;
  const isActive = metrics?.active || false;

  return {
    // Configuration
    config,
    updateConfig,
    saveSettings,
    isUpdatingSettings: updateSettingsMutation.isLoading,
    
    // Available options
    models,
    strategies,
    settings,
    
    // File selection
    selectedFiles,
    toggleFileSelection,
    selectAllFiles,
    clearFileSelection,
    
    // Generation
    generateEmbeddings,
    isGenerating: generateMutation.isLoading || isGenerating,
    isActive,
    progress,
    
    // Error handling
    error,
    clearError
  };
};
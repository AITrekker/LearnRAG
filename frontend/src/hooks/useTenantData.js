import { useQuery, useMutation, useQueryClient } from 'react-query';
import apiService from '../services/api';
import { useErrorHandler } from './useErrorHandler';

/**
 * useTenantData Hook - Centralized Tenant Data Management
 * 
 * Teaching Purpose: Demonstrates data management patterns for multi-tenant RAG systems:
 * - Coordinated queries for related data (info, files, stats)
 * - Smart caching and invalidation strategies
 * - Error handling for data operations
 * - File synchronization for dynamic document collections
 * 
 * Multi-Tenant RAG Concepts:
 * - Data isolation between tenants for security
 * - File sync operations for keeping embeddings current
 * - Statistics tracking for understanding system usage
 * - Efficient data fetching patterns for responsive UIs
 */
export const useTenantData = (apiKey) => {
  const queryClient = useQueryClient();
  const { error, handleError, clearError } = useErrorHandler();

  // Tenant information query
  const {
    data: tenantInfo,
    isLoading: infoLoading,
    error: infoError
  } = useQuery(
    'tenantInfo',
    apiService.getTenantInfo,
    {
      enabled: !!apiKey,
      retry: 1,
      staleTime: 30000, // Cache for 30 seconds
      onError: (error) => handleError(error, { operation: 'fetch_tenant_info' })
    }
  );

  // Tenant files query  
  const {
    data: files,
    isLoading: filesLoading,
    error: filesError
  } = useQuery(
    'tenantFiles',
    apiService.getTenantFiles,
    {
      enabled: !!apiKey,
      retry: 1,
      staleTime: 30000,
      onError: (error) => handleError(error, { operation: 'fetch_tenant_files' })
    }
  );

  // Tenant statistics query
  const {
    data: stats,
    isLoading: statsLoading,
    error: statsError  
  } = useQuery(
    'tenantStats',
    apiService.getTenantStats,
    {
      enabled: !!apiKey,
      retry: 1,
      staleTime: 60000, // Cache stats longer as they change less frequently
      onError: (error) => handleError(error, { operation: 'fetch_tenant_stats' })
    }
  );

  // File sync mutation
  const syncMutation = useMutation(
    apiService.syncFiles,
    {
      onSuccess: () => {
        // Invalidate related queries to refresh data
        queryClient.invalidateQueries('tenantFiles');
        queryClient.invalidateQueries('tenantStats');
        queryClient.invalidateQueries('tenantInfo');
        clearError();
      },
      onError: (error) => {
        handleError(error, { 
          operation: 'sync_files',
          tenant_id: tenantInfo?.id
        });
      }
    }
  );

  // Derived state for convenience
  const isLoading = infoLoading || filesLoading || statsLoading;
  const hasError = infoError || filesError || statsError || error;
  const isSyncing = syncMutation.isLoading;

  // File statistics derived from files data
  const fileStats = files ? {
    total: files.length,
    withEmbeddings: files.filter(f => f.has_embeddings).length,
    withoutEmbeddings: files.filter(f => !f.has_embeddings).length,
    totalSize: files.reduce((sum, f) => sum + f.size, 0)
  } : null;

  // Actions
  const syncFiles = () => {
    syncMutation.mutate();
  };

  const refreshAll = () => {
    queryClient.invalidateQueries('tenantInfo');
    queryClient.invalidateQueries('tenantFiles');
    queryClient.invalidateQueries('tenantStats');
  };

  return {
    // Data
    tenantInfo,
    files,
    stats,
    fileStats,
    
    // Loading states
    isLoading,
    infoLoading,
    filesLoading,
    statsLoading,
    isSyncing,
    
    // Error states
    hasError,
    error: hasError ? (error || infoError || filesError || statsError) : null,
    clearError,
    
    // Actions
    syncFiles,
    refreshAll
  };
};
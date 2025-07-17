/**
 * Dashboard Component - RAG System Overview and Management
 * 
 * This component demonstrates modern React patterns for data dashboards:
 * 
 * 1. REACT QUERY INTEGRATION: Efficient data fetching and caching
 * 2. REAL-TIME UPDATES: Live statistics and file synchronization
 * 3. ANIMATION EFFECTS: Smooth transitions using Framer Motion
 * 4. ERROR HANDLING: Graceful error states and user feedback
 * 5. MUTATION MANAGEMENT: Optimistic updates and cache invalidation
 * 
 * Core Dashboard Concepts Illustrated:
 * - Multiple data sources aggregated into unified view
 * - Real-time synchronization with backend state
 * - Progressive loading states and error boundaries
 * - User-initiated actions with immediate feedback
 * - Responsive design with animated statistics cards
 */

import React from 'react';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import { motion } from 'framer-motion';
import { Files, Database, Cpu, RefreshCw } from 'lucide-react';
import apiService from '../services/api';
import { formatFileSize } from '../utils';

const Dashboard = ({ apiKey }) => {
  /**
   * Dashboard data management using React Query
   * 
   * WHY REACT QUERY?
   * - Intelligent caching reduces unnecessary API calls
   * - Background refetching keeps data fresh
   * - Automatic error handling and retry logic
   * - Optimistic updates for better user experience
   * 
   * QUERY CONFIGURATION:
   * - enabled: Only fetch when API key is available
   * - retry: Limited retries to prevent endless loops
   * - staleTime: Balance between freshness and performance
   */
  const queryClient = useQueryClient();

  const { data: tenantInfo, isLoading: infoLoading, error: infoError } = useQuery(
    'tenantInfo',
    apiService.getTenantInfo,
    {
      enabled: !!apiKey,
      retry: 1,
      staleTime: 30000,
    }
  );

  const { data: tenantStats, isLoading: statsLoading, error: statsError } = useQuery(
    'tenantStats',
    apiService.getTenantStats,
    {
      enabled: !!apiKey,
      retry: 1,
      staleTime: 30000,
    }
  );

  const { data: files, isLoading: filesLoading, error: filesError } = useQuery(
    'tenantFiles',
    apiService.getTenantFiles,
    {
      enabled: !!apiKey,
      retry: 1,
      staleTime: 30000,
    }
  );

  const { mutate: syncFiles, isLoading: syncing } = useMutation(
    apiService.syncFiles,
    {
      onSuccess: () => {
        queryClient.invalidateQueries('tenantFiles');
        queryClient.invalidateQueries('tenantStats');
      }
    }
  );

  // Show errors if any API calls fail
  if (infoError || statsError || filesError) {
    console.error('Dashboard API errors:', { infoError, statsError, filesError });
  }

  if (infoLoading || statsLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin w-8 h-8 border-4 border-primary-600 border-t-transparent rounded-full"></div>
      </div>
    );
  }

  const stats = [
    {
      name: 'Total Files',
      value: tenantStats?.files?.total_files || 0,
      icon: Files,
      color: 'text-blue-600',
      bgColor: 'bg-blue-50'
    },
    {
      name: 'Total Chunks',
      value: tenantStats?.embeddings?.total_chunks || 0,
      icon: Database,
      color: 'text-green-600',
      bgColor: 'bg-green-50'
    },
    {
      name: 'Models Used',
      value: tenantStats?.embeddings?.models_used || 0,
      icon: Cpu,
      color: 'text-purple-600',
      bgColor: 'bg-purple-50'
    }
  ];

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Dashboard</h1>
          <p className="text-gray-600 mt-1">
            Welcome to {tenantInfo?.name} - {tenantInfo?.slug}
          </p>
        </div>
        
        <motion.button
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          onClick={() => syncFiles()}
          disabled={syncing}
          className="btn-secondary flex items-center"
        >
          <RefreshCw className={`w-4 h-4 mr-2 ${syncing ? 'animate-spin' : ''}`} />
          {syncing ? 'Syncing...' : 'Sync Files'}
        </motion.button>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        {stats.map((stat, index) => (
          <motion.div
            key={stat.name}
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            transition={{ delay: index * 0.1 }}
            className="card"
          >
            <div className="flex items-center">
              <div className={`p-3 rounded-lg ${stat.bgColor}`}>
                <stat.icon className={`w-6 h-6 ${stat.color}`} />
              </div>
              <div className="ml-4">
                <p className="text-sm font-medium text-gray-600">{stat.name}</p>
                <p className="text-2xl font-bold text-gray-900">{stat.value}</p>
              </div>
            </div>
          </motion.div>
        ))}
      </div>

      {/* Recent Files */}
      <div className="card">
        <h3 className="text-lg font-semibold text-gray-900 mb-4">Recent Files</h3>
        {filesLoading ? (
          <div className="animate-pulse space-y-3">
            {[...Array(3)].map((_, i) => (
              <div key={i} className="h-4 bg-gray-200 rounded"></div>
            ))}
          </div>
        ) : files && files.length > 0 ? (
          <div className="space-y-3">
            {files.slice(0, 5).map((file, index) => (
              <motion.div
                key={file.id}
                initial={{ opacity: 0, x: -20 }}
                animate={{ opacity: 1, x: 0 }}
                transition={{ delay: index * 0.1 }}
                className="flex items-center justify-between p-3 bg-gray-50 rounded-lg"
              >
                <div>
                  <p className="font-medium text-gray-900">{file.name}</p>
                  <p className="text-sm text-gray-500">
                    {formatFileSize(file.size)} • {file.type}
                  </p>
                </div>
                <div className="text-sm text-gray-400">
                  {new Date(file.created_at).toLocaleDateString()}
                </div>
              </motion.div>
            ))}
          </div>
        ) : (
          <p className="text-gray-500 text-center py-8">
            No files found. Sync your demo data to get started.
          </p>
        )}
      </div>
    </div>
  );
};

export default Dashboard;
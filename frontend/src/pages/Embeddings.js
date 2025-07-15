import React, { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Zap, Play, Settings, Loader
} from 'lucide-react';
import apiService from '../services/api';
import EmbeddingSettings from '../components/EmbeddingSettings';
import EmbeddingProgress from '../components/EmbeddingProgress';
import FileSelection from '../components/FileSelection';

const Embeddings = ({ apiKey }) => {
  const queryClient = useQueryClient();
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [showSettings, setShowSettings] = useState(false);

  // Fetch embedding settings
  const { data: settings } = useQuery(
    'embeddingSettings',
    apiService.getEmbeddingSettings,
    { enabled: !!apiKey, retry: 1 }
  );


  // Fetch available models
  const { data: models, isLoading: modelsLoading } = useQuery(
    'embeddingModels',
    apiService.getAvailableModels,
    { enabled: !!apiKey, retry: 1, staleTime: 60000 }
  );

  // Fetch chunking strategies
  const { data: strategies, isLoading: strategiesLoading } = useQuery(
    'chunkingStrategies',
    apiService.getChunkingStrategies,
    { enabled: !!apiKey, retry: 1, staleTime: 60000 }
  );

  // Fetch tenant files
  const { data: files, isLoading: filesLoading } = useQuery(
    'tenantFiles',
    apiService.getTenantFiles,
    { enabled: !!apiKey, retry: 1, staleTime: 30000 }
  );

  // Fetch current metrics
  const { data: metrics } = useQuery(
    'currentMetrics',
    apiService.getCurrentMetrics,
    { 
      enabled: !!apiKey, 
      retry: 1, 
      refetchInterval: 2000, // Poll every 2 seconds
      refetchIntervalInBackground: true
    }
  );


  // Generate embeddings mutation
  const generateEmbeddingsMutation = useMutation(
    apiService.generateEmbeddings,
    {
      onSuccess: () => {
        queryClient.invalidateQueries('tenantFiles');
        queryClient.invalidateQueries('currentMetrics');
        setSelectedFiles([]);
      }
    }
  );


  const handleGenerate = () => {
    generateEmbeddingsMutation.mutate({
      embedding_model: settings?.embedding_model,
      chunking_strategy: settings?.chunking_strategy,
      chunk_size: settings?.chunk_size,
      chunk_overlap: settings?.chunk_overlap,
      file_ids: selectedFiles.length > 0 ? selectedFiles : undefined
    });
  };

  const toggleFileSelection = (fileId) => {
    setSelectedFiles(prev => 
      prev.includes(fileId) 
        ? prev.filter(id => id !== fileId)
        : [...prev, fileId]
    );
  };


  if (modelsLoading || strategiesLoading || filesLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin w-8 h-8 border-4 border-primary-600 border-t-transparent rounded-full"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6 max-w-full overflow-hidden">
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-3xl font-bold text-gray-900">Embeddings</h1>
          <p className="text-gray-600 mt-1">Generate vector embeddings for your documents</p>
          {settings && (
            <div className="mt-2 p-3 bg-blue-50 rounded-lg border border-blue-200">
              <p className="text-sm text-blue-800">
                <strong>Current Settings:</strong> {settings.embedding_model} • {settings.chunking_strategy}
                {settings.chunking_strategy === 'fixed_size' && settings.chunk_size && ` • ${settings.chunk_size} tokens`} • {settings.chunk_overlap} overlap
              </p>
            </div>
          )}
        </div>
        <motion.button
          whileHover={{ scale: 1.05 }}
          whileTap={{ scale: 0.95 }}
          onClick={() => setShowSettings(!showSettings)}
          className={`flex items-center px-4 py-2 rounded-lg font-medium transition-colors ${
            showSettings
              ? 'bg-primary-100 text-primary-700 border border-primary-300'
              : 'bg-white text-gray-700 border border-gray-300 hover:bg-gray-50'
          }`}
        >
          <Settings className="w-4 h-4 mr-2" />
          Settings
        </motion.button>
      </div>

      {/* Current Progress */}
      <EmbeddingProgress metrics={metrics} />

      {/* Settings Modal */}
      <EmbeddingSettings
        apiKey={apiKey}
        settings={settings}
        models={models}
        strategies={strategies}
        showSettings={showSettings}
        setShowSettings={setShowSettings}
      />

      {/* File Selection */}
      <FileSelection
        files={files}
        selectedFiles={selectedFiles}
        onToggleFile={toggleFileSelection}
        onClearSelection={() => setSelectedFiles([])}
        onSelectAll={setSelectedFiles}
      />

      {/* Generate Button */}
      <div className="card">
        <div className="flex items-center justify-between">
          <div>
            <h3 className="text-lg font-semibold text-gray-900">Generate Embeddings</h3>
            <p className="text-sm text-gray-500">
              Create vector embeddings using {settings?.embedding_model || 'current model'} with {settings?.chunking_strategy || 'current'} chunking
            </p>
          </div>
          
          <motion.button
            whileHover={{ scale: generateEmbeddingsMutation.isLoading ? 1 : 1.05 }}
            whileTap={{ scale: generateEmbeddingsMutation.isLoading ? 1 : 0.95 }}
            onClick={handleGenerate}
            disabled={generateEmbeddingsMutation.isLoading || !files?.length || metrics?.active}
            className={`flex items-center px-6 py-3 rounded-lg font-medium transition-colors ${
              generateEmbeddingsMutation.isLoading || !files?.length || metrics?.active
                ? 'bg-gray-300 cursor-not-allowed text-gray-500'
                : 'bg-primary-600 hover:bg-primary-700 text-white'
            }`}
          >
            {generateEmbeddingsMutation.isLoading || metrics?.active ? (
              <>
                <Loader className="w-5 h-5 mr-2 animate-spin" />
                {metrics?.active ? 'Processing...' : 'Starting...'}
              </>
            ) : (
              <>
                <Play className="w-5 h-5 mr-2" />
                Generate
              </>
            )}
          </motion.button>
        </div>

        <AnimatePresence>
          {(generateEmbeddingsMutation.isLoading || metrics?.active) && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="mt-4 p-4 bg-primary-50 rounded-lg"
            >
              <div className="flex items-center">
                <Zap className="w-5 h-5 text-primary-600 mr-2" />
                <span className="text-primary-700">
                  {metrics?.active ? 'Processing embeddings...' : 'Starting embedding generation...'}
                </span>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
};

export default Embeddings;
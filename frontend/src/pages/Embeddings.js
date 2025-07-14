import React, { useState, useEffect } from 'react';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import { motion, AnimatePresence } from 'framer-motion';
import { 
  Zap, Play, CheckCircle, FileText, Settings, Loader, 
  Save, RotateCcw, AlertCircle, TrendingUp, Clock
} from 'lucide-react';
import apiService from '../services/api';

const Embeddings = ({ apiKey }) => {
  const queryClient = useQueryClient();
  const [selectedFiles, setSelectedFiles] = useState([]);
  const [showSettings, setShowSettings] = useState(false);
  const [settingsForm, setSettingsForm] = useState({
    embedding_model: '',
    chunking_strategy: '',
    chunk_size: 512,
    chunk_overlap: 50
  });

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

  // Update settings mutation
  const updateSettingsMutation = useMutation(
    apiService.updateEmbeddingSettings,
    {
      onSuccess: () => {
        queryClient.invalidateQueries('embeddingSettings');
        queryClient.invalidateQueries('tenantFiles'); // Refresh to show no embeddings
        setShowSettings(false);
      }
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

  // Initialize form when settings load
  useEffect(() => {
    if (settings) {
      setSettingsForm({
        embedding_model: settings.embedding_model,
        chunking_strategy: settings.chunking_strategy,
        chunk_size: settings.chunk_size || 512,
        chunk_overlap: settings.chunk_overlap || 50
      });
    }
  }, [settings]);

  const handleUpdateSettings = () => {
    updateSettingsMutation.mutate(settingsForm);
  };

  const handleResetSettings = () => {
    if (settings) {
      setSettingsForm({
        embedding_model: settings.embedding_model,
        chunking_strategy: settings.chunking_strategy,
        chunk_size: settings.chunk_size || 512,
        chunk_overlap: settings.chunk_overlap || 50
      });
    }
  };

  const handleGenerate = () => {
    // Use current tenant settings, not form values
    generateEmbeddingsMutation.mutate({
      embedding_model: settings?.embedding_model || settingsForm.embedding_model,
      chunking_strategy: settings?.chunking_strategy || settingsForm.chunking_strategy,
      chunk_size: settings?.chunk_size || settingsForm.chunk_size,
      chunk_overlap: settings?.chunk_overlap || settingsForm.chunk_overlap,
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

  const getSelectedStrategy = () => {
    return strategies?.strategies?.find(s => s.name === settingsForm.chunking_strategy);
  };

  const getSelectedModel = () => {
    return models?.models?.find(m => m.name === settingsForm.embedding_model);
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
      <AnimatePresence>
        {metrics?.active && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="card border-primary-200 bg-primary-50"
          >
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center">
                <TrendingUp className="w-5 h-5 text-primary-600 mr-2" />
                <h3 className="text-lg font-semibold text-primary-900">Processing in Progress</h3>
              </div>
              <div className="flex items-center text-sm text-primary-700">
                <Clock className="w-4 h-4 mr-1" />
                {metrics.progress?.elapsed_time?.toFixed(1)}s
              </div>
            </div>
            
            <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
              <div className="text-center">
                <div className="text-2xl font-bold text-primary-900">
                  {metrics.progress?.files_processed || 0}
                </div>
                <div className="text-sm text-primary-600">Files Processed</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-primary-900">
                  {metrics.progress?.total_files || 0}
                </div>
                <div className="text-sm text-primary-600">Total Files</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-primary-900">
                  {metrics.progress?.total_chunks || 0}
                </div>
                <div className="text-sm text-primary-600">Chunks Generated</div>
              </div>
              <div className="text-center">
                <div className="text-2xl font-bold text-primary-900">
                  {metrics.progress?.total_tokens || 0}
                </div>
                <div className="text-sm text-primary-600">Tokens Processed</div>
              </div>
            </div>

            {/* Progress Bar */}
            {metrics.progress?.total_files > 0 && (
              <div className="mt-4">
                <div className="flex justify-between text-sm text-primary-700 mb-1">
                  <span>Progress</span>
                  <span>{metrics.progress.files_processed}/{metrics.progress.total_files} files</span>
                </div>
                <div className="w-full bg-primary-200 rounded-full h-2">
                  <div 
                    className="bg-primary-600 h-2 rounded-full transition-all duration-500"
                    style={{ 
                      width: `${(metrics.progress.files_processed / metrics.progress.total_files) * 100}%` 
                    }}
                  ></div>
                </div>
              </div>
            )}

            {/* Current/Recent Files */}
            {metrics.progress?.files && metrics.progress.files.length > 0 && (
              <div className="mt-4">
                <h4 className="text-sm font-medium text-primary-900 mb-2">
                  {metrics.progress.files_processed < metrics.progress.total_files 
                    ? 'Recent Files:' 
                    : 'Completed Files:'
                  }
                </h4>
                <div className="space-y-2">
                  {metrics.progress.files.slice(-3).map((file, idx) => (
                    <div key={idx} className="flex justify-between items-center text-sm bg-white rounded-lg p-2 border border-primary-200">
                      <div className="flex items-center">
                        <CheckCircle className="w-4 h-4 text-green-500 mr-2" />
                        <span className="text-primary-900 font-medium">{file.filename}</span>
                      </div>
                      <div className="text-primary-700">
                        {file.chunks_generated} chunks • {file.processing_time_sec}s
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>

      {/* Settings Panel */}
      <AnimatePresence>
        {showSettings && (
          <motion.div
            initial={{ opacity: 0, height: 0 }}
            animate={{ opacity: 1, height: 'auto' }}
            exit={{ opacity: 0, height: 0 }}
            className="card border-amber-200 bg-amber-50"
          >
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center">
                <Settings className="w-5 h-5 text-amber-600 mr-2" />
                <h3 className="text-lg font-semibold text-amber-900">Embedding Configuration</h3>
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={handleResetSettings}
                  className="flex items-center px-3 py-1 text-sm text-amber-700 hover:text-amber-800"
                >
                  <RotateCcw className="w-4 h-4 mr-1" />
                  Reset
                </button>
                <button
                  onClick={handleUpdateSettings}
                  disabled={updateSettingsMutation.isLoading}
                  className="flex items-center px-4 py-2 bg-amber-600 text-white rounded-lg hover:bg-amber-700 disabled:opacity-50"
                >
                  {updateSettingsMutation.isLoading ? (
                    <Loader className="w-4 h-4 mr-2 animate-spin" />
                  ) : (
                    <Save className="w-4 h-4 mr-2" />
                  )}
                  Save Settings
                </button>
              </div>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {/* Model Selection */}
              <div>
                <label className="block text-sm font-medium text-amber-800 mb-2">
                  Embedding Model
                </label>
                <select
                  value={settingsForm.embedding_model}
                  onChange={(e) => setSettingsForm(prev => ({ ...prev, embedding_model: e.target.value }))}
                  className="w-full p-3 border border-amber-300 rounded-lg focus:ring-2 focus:ring-amber-500 focus:border-transparent bg-white"
                >
                  {models?.models?.map((model) => (
                    <option key={model.name} value={model.name}>
                      {model.name} ({model.dimension}d)
                    </option>
                  ))}
                </select>
                <p className="text-sm text-amber-700 mt-1">
                  {getSelectedModel()?.description}
                </p>
              </div>

              {/* Chunking Strategy */}
              <div>
                <label className="block text-sm font-medium text-amber-800 mb-2">
                  Chunking Strategy
                </label>
                <select
                  value={settingsForm.chunking_strategy}
                  onChange={(e) => setSettingsForm(prev => ({ ...prev, chunking_strategy: e.target.value }))}
                  className="w-full p-3 border border-amber-300 rounded-lg focus:ring-2 focus:ring-amber-500 focus:border-transparent bg-white"
                >
                  {strategies?.strategies?.map((strategy) => (
                    <option key={strategy.name} value={strategy.name}>
                      {strategy.name}
                    </option>
                  ))}
                </select>
                <p className="text-sm text-amber-700 mt-1">
                  {getSelectedStrategy()?.description}
                </p>
              </div>

              {/* Chunk Size (for fixed_size and recursive) */}
              {(settingsForm.chunking_strategy === 'fixed_size' || settingsForm.chunking_strategy === 'recursive') && (
                <div>
                  <label className="block text-sm font-medium text-amber-800 mb-2">
                    Chunk Size (tokens)
                  </label>
                  <input
                    type="number"
                    min="50"
                    max="2000"
                    value={settingsForm.chunk_size}
                    onChange={(e) => setSettingsForm(prev => ({ ...prev, chunk_size: parseInt(e.target.value) }))}
                    className="w-full p-3 border border-amber-300 rounded-lg focus:ring-2 focus:ring-amber-500 focus:border-transparent"
                  />
                  <p className="text-sm text-amber-700 mt-1">Number of tokens per chunk</p>
                </div>
              )}

              {/* Chunk Overlap */}
              <div>
                <label className="block text-sm font-medium text-amber-800 mb-2">
                  Chunk Overlap (tokens)
                </label>
                <input
                  type="number"
                  min="0"
                  max="500"
                  value={settingsForm.chunk_overlap}
                  onChange={(e) => setSettingsForm(prev => ({ ...prev, chunk_overlap: parseInt(e.target.value) }))}
                  className="w-full p-3 border border-amber-300 rounded-lg focus:ring-2 focus:ring-amber-500 focus:border-transparent"
                />
                <p className="text-sm text-amber-700 mt-1">Overlapping tokens between chunks</p>
              </div>
            </div>

            <div className="mt-4 p-3 bg-amber-100 rounded-lg border border-amber-200">
              <div className="flex items-start">
                <AlertCircle className="w-5 h-5 text-amber-600 mr-2 mt-0.5" />
                <div className="text-sm text-amber-800">
                  <strong>Important:</strong> Changing settings will automatically delete all existing embeddings 
                  for this tenant to ensure consistency. You'll need to regenerate embeddings after saving.
                </div>
              </div>
            </div>
          </motion.div>
        )}
      </AnimatePresence>

      {/* File Selection */}
      <div className="card">
        <div className="flex items-center justify-between mb-4">
          <div className="flex items-center">
            <FileText className="w-5 h-5 text-primary-600 mr-2" />
            <h3 className="text-lg font-semibold text-gray-900">Select Files</h3>
          </div>
          <div className="text-sm text-gray-500">
            {selectedFiles.length > 0 ? `${selectedFiles.length} selected` : 'All files'}
          </div>
        </div>

        <div className="space-y-2 max-h-64 overflow-y-auto overflow-x-hidden">
          {files?.map((file) => (
            <motion.div
              key={file.id}
              whileHover={{ scale: 1.01 }}
              className={`p-3 border rounded-lg cursor-pointer transition-colors ${
                selectedFiles.includes(file.id)
                  ? 'border-primary-300 bg-primary-50'
                  : 'border-gray-200 hover:border-gray-300'
              }`}
              onClick={() => toggleFileSelection(file.id)}
            >
              <div className="flex items-center justify-between">
                <div>
                  <p className="font-medium text-gray-900">{file.filename}</p>
                  <p className="text-sm text-gray-500">
                    {(file.file_size / 1024).toFixed(1)} KB • {file.content_type}
                  </p>
                </div>
                {selectedFiles.includes(file.id) && (
                  <CheckCircle className="w-5 h-5 text-primary-600" />
                )}
              </div>
            </motion.div>
          ))}
        </div>

        <div className="mt-4 flex items-center justify-between">
          <button
            onClick={() => setSelectedFiles([])}
            className="text-sm text-gray-500 hover:text-gray-700"
          >
            Clear selection
          </button>
          <button
            onClick={() => setSelectedFiles(files?.map(f => f.id) || [])}
            className="text-sm text-primary-600 hover:text-primary-700"
          >
            Select all
          </button>
        </div>
      </div>

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
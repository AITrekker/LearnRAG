import React, { useState } from 'react';
import { useQuery, useMutation, useQueryClient } from 'react-query';
import { motion, AnimatePresence } from 'framer-motion';
import { Zap, Play, CheckCircle, FileText, Settings, Loader } from 'lucide-react';
import apiService from '../services/api';

const Embeddings = ({ apiKey }) => {
  const queryClient = useQueryClient();
  const [selectedModel, setSelectedModel] = useState('sentence-transformers/all-MiniLM-L6-v2');
  const [selectedStrategy, setSelectedStrategy] = useState('fixed_size');
  const [selectedFiles, setSelectedFiles] = useState([]);

  const { data: models, isLoading: modelsLoading, error: modelsError } = useQuery(
    'embeddingModels',
    apiService.getAvailableModels,
    { 
      enabled: !!apiKey,
      retry: 1,
      staleTime: 60000,
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

  const { mutate: generateEmbeddings, isLoading: generating } = useMutation(
    apiService.generateEmbeddings,
    {
      onSuccess: () => {
        queryClient.invalidateQueries('tenantFiles');
        setSelectedFiles([]);
      }
    }
  );

  const chunkingStrategies = [
    { value: 'fixed_size', label: 'Fixed Size', description: '512 tokens with 50 token overlap' }
  ];

  const handleGenerate = () => {
    generateEmbeddings({
      embedding_model: selectedModel,
      chunking_strategy: selectedStrategy,
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

  // Show errors if any API calls fail
  if (modelsError || filesError) {
    console.error('Embeddings API errors:', { modelsError, filesError });
  }

  if (modelsLoading || filesLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin w-8 h-8 border-4 border-primary-600 border-t-transparent rounded-full"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Embeddings</h1>
        <p className="text-gray-600 mt-1">Generate vector embeddings for your documents</p>
      </div>

      {/* Configuration Panel */}
      <div className="card">
        <div className="flex items-center mb-4">
          <Settings className="w-5 h-5 text-primary-600 mr-2" />
          <h3 className="text-lg font-semibold text-gray-900">Configuration</h3>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          {/* Model Selection */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Embedding Model
            </label>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            >
              {models?.models?.map((model) => (
                <option key={model.name} value={model.name}>
                  {model.name} ({model.dimension}d)
                </option>
              ))}
            </select>
            <p className="text-sm text-gray-500 mt-1">
              {models?.models?.find(m => m.name === selectedModel)?.description}
            </p>
          </div>

          {/* Chunking Strategy */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Chunking Strategy
            </label>
            <select
              value={selectedStrategy}
              onChange={(e) => setSelectedStrategy(e.target.value)}
              className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            >
              {chunkingStrategies.map((strategy) => (
                <option key={strategy.value} value={strategy.value}>
                  {strategy.label}
                </option>
              ))}
            </select>
            <p className="text-sm text-gray-500 mt-1">
              {chunkingStrategies.find(s => s.value === selectedStrategy)?.description}
            </p>
          </div>
        </div>
      </div>

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

        <div className="space-y-2 max-h-64 overflow-y-auto">
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
              Create vector embeddings using {selectedModel} with {selectedStrategy} chunking
            </p>
          </div>
          
          <motion.button
            whileHover={{ scale: generating ? 1 : 1.05 }}
            whileTap={{ scale: generating ? 1 : 0.95 }}
            onClick={handleGenerate}
            disabled={generating || !files?.length}
            className={`flex items-center px-6 py-3 rounded-lg font-medium transition-colors ${
              generating || !files?.length
                ? 'bg-gray-300 cursor-not-allowed'
                : 'bg-primary-600 hover:bg-primary-700 text-white'
            }`}
          >
            {generating ? (
              <>
                <Loader className="w-5 h-5 mr-2 animate-spin" />
                Generating...
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
          {generating && (
            <motion.div
              initial={{ opacity: 0, height: 0 }}
              animate={{ opacity: 1, height: 'auto' }}
              exit={{ opacity: 0, height: 0 }}
              className="mt-4 p-4 bg-primary-50 rounded-lg"
            >
              <div className="flex items-center">
                <Zap className="w-5 h-5 text-primary-600 mr-2" />
                <span className="text-primary-700">Processing embeddings in background...</span>
              </div>
            </motion.div>
          )}
        </AnimatePresence>
      </div>
    </div>
  );
};

export default Embeddings;
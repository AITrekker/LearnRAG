import React, { useState, useEffect } from 'react';
import { useMutation, useQueryClient } from 'react-query';
import { motion, AnimatePresence } from 'framer-motion';
import { Settings, Save, RotateCcw, X } from 'lucide-react';
import apiService from '../services/api';
import { useErrorHandler } from '../hooks/useErrorHandler';
import ErrorNotification from './ErrorNotification';
import { DEFAULTS } from '../config';

/**
 * EmbeddingSettings Component
 * 
 * Teaching Purpose: Demonstrates how RAG systems allow users to configure:
 * - Embedding models (different neural networks for text→vector conversion)
 * - Chunking strategies (how to split documents for processing)
 * - Chunk size/overlap parameters (balancing context vs precision)
 */
const EmbeddingSettings = ({ 
  apiKey, 
  settings, 
  models, 
  strategies, 
  showSettings, 
  setShowSettings 
}) => {
  const queryClient = useQueryClient();
  const { error, handleError, clearError, canRetry } = useErrorHandler();
  const [settingsForm, setSettingsForm] = useState({
    embedding_model: '',
    chunking_strategy: '',
    chunk_size: DEFAULTS.CHUNK_SIZE,
    chunk_overlap: DEFAULTS.CHUNK_OVERLAP
  });

  // Load current settings into form
  useEffect(() => {
    if (settings) {
      setSettingsForm({
        embedding_model: settings.embedding_model || '',
        chunking_strategy: settings.chunking_strategy || '',
        chunk_size: settings.chunk_size || DEFAULTS.CHUNK_SIZE,
        chunk_overlap: settings.chunk_overlap || DEFAULTS.CHUNK_OVERLAP
      });
    }
  }, [settings]);

  // Update settings mutation with enhanced error handling
  const updateSettingsMutation = useMutation(
    (settingsData) => apiService.updateEmbeddingSettings(settingsData),
    {
      onSuccess: () => {
        queryClient.invalidateQueries('embeddingSettings');
        queryClient.invalidateQueries('tenantFiles');
        clearError(); // Clear any previous errors
        setShowSettings(false);
      },
      onError: (error) => {
        handleError(error, { 
          operation: 'update_embedding_settings',
          settings: settingsForm 
        });
      }
    }
  );

  const handleSaveSettings = () => {
    updateSettingsMutation.mutate(settingsForm);
  };

  const resetToDefaults = () => {
    setSettingsForm({
      embedding_model: DEFAULTS.EMBEDDING_MODEL,
      chunking_strategy: DEFAULTS.CHUNKING_STRATEGY,
      chunk_size: DEFAULTS.CHUNK_SIZE,
      chunk_overlap: DEFAULTS.CHUNK_OVERLAP
    });
  };

  if (!showSettings) return null;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50"
      >
        <motion.div
          initial={{ scale: 0.9, opacity: 0 }}
          animate={{ scale: 1, opacity: 1 }}
          exit={{ scale: 0.9, opacity: 0 }}
          className="bg-white rounded-xl p-6 max-w-lg w-full mx-4 shadow-2xl"
        >
          <div className="flex items-center justify-between mb-6">
            <div className="flex items-center space-x-2">
              <Settings className="text-blue-600" size={24} />
              <h3 className="text-xl font-semibold">Embedding Settings</h3>
            </div>
            <button
              onClick={() => setShowSettings(false)}
              className="text-gray-400 hover:text-gray-600 transition-colors"
            >
              <X size={20} />
            </button>
          </div>

          {/* Error Notification */}
          {error && (
            <ErrorNotification 
              error={error}
              onDismiss={clearError}
              onRetry={canRetry ? () => updateSettingsMutation.mutate(settingsForm) : null}
              showRetry={canRetry}
              className="mb-4"
            />
          )}

          <div className="space-y-4">
            {/* Embedding Model Selection */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Embedding Model
              </label>
              <select
                value={settingsForm.embedding_model}
                onChange={(e) => setSettingsForm(prev => ({
                  ...prev,
                  embedding_model: e.target.value
                }))}
                className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                style={{ paddingRight: '48px' }}
              >
                {models?.models?.map(model => (
                  <option key={model.name} value={model.name}>
                    {model.name} ({model.dimension}d) - {model.description}
                  </option>
                ))}
              </select>
            </div>

            {/* Chunking Strategy */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Chunking Strategy
              </label>
              <select
                value={settingsForm.chunking_strategy}
                onChange={(e) => setSettingsForm(prev => ({
                  ...prev,
                  chunking_strategy: e.target.value
                }))}
                className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                style={{ paddingRight: '48px' }}
              >
                {strategies?.strategies?.map(strategy => (
                  <option key={strategy.name} value={strategy.name}>
                    {strategy.name} - {strategy.description}
                  </option>
                ))}
              </select>
            </div>

            {/* Chunk Size */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Chunk Size (tokens)
              </label>
              <input
                type="number"
                value={settingsForm.chunk_size}
                onChange={(e) => setSettingsForm(prev => ({
                  ...prev,
                  chunk_size: parseInt(e.target.value) || 512
                }))}
                className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                min="50"
                max="2048"
              />
            </div>

            {/* Chunk Overlap */}
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-2">
                Chunk Overlap (tokens)
              </label>
              <input
                type="number"
                value={settingsForm.chunk_overlap}
                onChange={(e) => setSettingsForm(prev => ({
                  ...prev,
                  chunk_overlap: parseInt(e.target.value) || 50
                }))}
                className="w-full p-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
                min="0"
                max="200"
              />
            </div>
          </div>

          {/* Action Buttons */}
          <div className="flex space-x-3 mt-6">
            <button
              onClick={handleSaveSettings}
              disabled={updateSettingsMutation.isLoading}
              className="flex-1 bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 transition-colors flex items-center justify-center space-x-2 disabled:opacity-50"
            >
              <Save size={16} />
              <span>{updateSettingsMutation.isLoading ? 'Saving...' : 'Save Settings'}</span>
            </button>
            <button
              onClick={resetToDefaults}
              className="px-4 py-2 border border-gray-300 text-gray-700 rounded-lg hover:bg-gray-50 transition-colors flex items-center space-x-2"
            >
              <RotateCcw size={16} />
              <span>Reset</span>
            </button>
          </div>
        </motion.div>
      </motion.div>
    </AnimatePresence>
  );
};

export default EmbeddingSettings;
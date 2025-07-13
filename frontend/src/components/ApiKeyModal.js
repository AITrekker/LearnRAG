import React, { useState } from 'react';
import { motion } from 'framer-motion';
import { Key, Loader } from 'lucide-react';
import apiService from '../services/api';

const ApiKeyModal = ({ onSubmit }) => {
  const [apiKey, setApiKey] = useState('');
  const [isValidating, setIsValidating] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!apiKey.trim()) return;

    setIsValidating(true);
    setError('');

    try {
      await apiService.validateApiKey(apiKey);
      onSubmit(apiKey);
    } catch (err) {
      setError('Invalid API key. Please check and try again.');
    } finally {
      setIsValidating(false);
    }
  };

  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      className="card max-w-md mx-auto"
    >
      <div className="text-center mb-6">
        <motion.div
          animate={{ rotate: 360 }}
          transition={{ duration: 2, repeat: Infinity, ease: "linear" }}
          className="inline-block"
        >
          <Key className="w-12 h-12 text-primary-600 mx-auto mb-4" />
        </motion.div>
        <h2 className="text-2xl font-bold text-gray-900 mb-2">Welcome to LearnRAG</h2>
        <p className="text-gray-600">Enter your API key to get started</p>
      </div>

      <form onSubmit={handleSubmit}>
        <div className="mb-4">
          <label htmlFor="apiKey" className="block text-sm font-medium text-gray-700 mb-2">
            API Key
          </label>
          <input
            type="text"
            id="apiKey"
            value={apiKey}
            onChange={(e) => setApiKey(e.target.value)}
            placeholder="lr_xxxxxxxxxx..."
            className="w-full px-3 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
            disabled={isValidating}
          />
        </div>

        {error && (
          <motion.div
            initial={{ opacity: 0, y: -10 }}
            animate={{ opacity: 1, y: 0 }}
            className="mb-4 p-3 bg-red-50 border border-red-200 rounded-lg text-red-600 text-sm"
          >
            {error}
          </motion.div>
        )}

        <button
          type="submit"
          disabled={!apiKey.trim() || isValidating}
          className="w-full btn-primary disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center"
        >
          {isValidating ? (
            <>
              <Loader className="w-4 h-4 mr-2 animate-spin" />
              Validating...
            </>
          ) : (
            'Connect'
          )}
        </button>
      </form>
    </motion.div>
  );
};

export default ApiKeyModal;
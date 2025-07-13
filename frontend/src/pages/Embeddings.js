import React from 'react';
import { motion } from 'framer-motion';
import { Zap, Clock, CheckCircle } from 'lucide-react';

const Embeddings = ({ apiKey }) => {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900">Embeddings</h1>
        <p className="text-gray-600 mt-1">
          Manage and generate embeddings for your documents
        </p>
      </div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="card text-center py-12"
      >
        <Zap className="w-16 h-16 text-primary-600 mx-auto mb-4" />
        <h3 className="text-xl font-semibold text-gray-900 mb-2">
          Embedding Management Coming Soon
        </h3>
        <p className="text-gray-600 mb-6">
          This feature will allow you to generate and manage embeddings using different models
        </p>
        <div className="space-y-2 text-left max-w-md mx-auto">
          <div className="flex items-center text-gray-600">
            <Clock className="w-4 h-4 mr-2" />
            Model selection dropdown
          </div>
          <div className="flex items-center text-gray-600">
            <Clock className="w-4 h-4 mr-2" />
            Chunking strategy options
          </div>
          <div className="flex items-center text-gray-600">
            <Clock className="w-4 h-4 mr-2" />
            Progress tracking
          </div>
          <div className="flex items-center text-green-600">
            <CheckCircle className="w-4 h-4 mr-2" />
            Delta sync (only re-embed when needed)
          </div>
        </div>
      </motion.div>
    </div>
  );
};

export default Embeddings;
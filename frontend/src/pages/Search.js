import React from 'react';
import { motion } from 'framer-motion';
import { Search as SearchIcon, Clock, CheckCircle } from 'lucide-react';

const Search = ({ apiKey }) => {
  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900">RAG Search</h1>
        <p className="text-gray-600 mt-1">
          Search through your documents using different RAG techniques
        </p>
      </div>

      <motion.div
        initial={{ opacity: 0, y: 20 }}
        animate={{ opacity: 1, y: 0 }}
        className="card text-center py-12"
      >
        <SearchIcon className="w-16 h-16 text-secondary-600 mx-auto mb-4" />
        <h3 className="text-xl font-semibold text-gray-900 mb-2">
          Interactive Search Coming Soon
        </h3>
        <p className="text-gray-600 mb-6">
          This feature will provide an interactive search interface to experiment with RAG techniques
        </p>
        <div className="space-y-2 text-left max-w-md mx-auto">
          <div className="flex items-center text-green-600">
            <CheckCircle className="w-4 h-4 mr-2" />
            Similarity search with pgvector
          </div>
          <div className="flex items-center text-gray-600">
            <Clock className="w-4 h-4 mr-2" />
            RAG technique dropdown
          </div>
          <div className="flex items-center text-gray-600">
            <Clock className="w-4 h-4 mr-2" />
            Real-time search results
          </div>
          <div className="flex items-center text-gray-600">
            <Clock className="w-4 h-4 mr-2" />
            Similarity score visualization
          </div>
          <div className="flex items-center text-gray-600">
            <Clock className="w-4 h-4 mr-2" />
            Search history and comparison
          </div>
        </div>
      </motion.div>
    </div>
  );
};

export default Search;
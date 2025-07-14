import React, { useState } from 'react';
import { useQuery, useMutation } from 'react-query';
import { motion, AnimatePresence } from 'framer-motion';
import { Search as SearchIcon, Clock, Target, FileText, BarChart3 } from 'lucide-react';
import apiService from '../services/api';

const Search = ({ apiKey }) => {
  const [query, setQuery] = useState('');
  const [selectedModel, setSelectedModel] = useState('sentence-transformers/all-MiniLM-L6-v2');
  const [selectedStrategy, setSelectedStrategy] = useState('fixed_size');
  const [selectedTechnique, setSelectedTechnique] = useState('similarity_search');
  const [topK, setTopK] = useState(5);
  const [searchResults, setSearchResults] = useState(null);

  const { data: models } = useQuery(
    'embeddingModels',
    apiService.getAvailableModels,
    { enabled: !!apiKey }
  );

  const { data: techniques } = useQuery(
    'ragTechniques',
    apiService.getRagTechniques,
    { enabled: !!apiKey }
  );

  const { mutate: performSearch, isLoading: searching } = useMutation(
    apiService.search,
    {
      onSuccess: (data) => setSearchResults(data),
      onError: (error) => console.error('Search failed:', error)
    }
  );

  const chunkingStrategies = [
    { value: 'fixed_size', label: 'Fixed Size', description: '512 tokens with 50 token overlap' }
  ];

  const handleSearch = (e) => {
    e.preventDefault();
    if (!query.trim()) return;

    performSearch({
      query: query.trim(),
      embedding_model: selectedModel,
      chunking_strategy: selectedStrategy,
      rag_technique: selectedTechnique,
      top_k: topK
    });
  };

  const getSimilarityColor = (score) => {
    if (score >= 0.8) return 'text-green-600 bg-green-50';
    if (score >= 0.6) return 'text-yellow-600 bg-yellow-50';
    return 'text-red-600 bg-red-50';
  };

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-3xl font-bold text-gray-900">RAG Search</h1>
        <p className="text-gray-600 mt-1">Search through your documents using different RAG techniques</p>
      </div>

      {/* Search Configuration */}
      <div className="card">
        <div className="flex items-center mb-4">
          <Target className="w-5 h-5 text-primary-600 mr-2" />
          <h3 className="text-lg font-semibold text-gray-900">Search Configuration</h3>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          {/* Model Selection */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Model</label>
            <select
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
              className="w-full p-2 border border-gray-300 rounded-lg text-sm"
            >
              {models?.models?.map((model) => (
                <option key={model.name} value={model.name}>
                  {model.name.split('/').pop()}
                </option>
              ))}
            </select>
          </div>

          {/* Chunking Strategy */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Chunking</label>
            <select
              value={selectedStrategy}
              onChange={(e) => setSelectedStrategy(e.target.value)}
              className="w-full p-2 border border-gray-300 rounded-lg text-sm"
            >
              {chunkingStrategies.map((strategy) => (
                <option key={strategy.value} value={strategy.value}>
                  {strategy.label}
                </option>
              ))}
            </select>
          </div>

          {/* RAG Technique */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">RAG Technique</label>
            <select
              value={selectedTechnique}
              onChange={(e) => setSelectedTechnique(e.target.value)}
              className="w-full p-2 border border-gray-300 rounded-lg text-sm"
            >
              {techniques?.techniques?.map((technique) => (
                <option key={technique.name} value={technique.name}>
                  {technique.name.replace('_', ' ')}
                </option>
              ))}
            </select>
          </div>

          {/* Top K Results */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">Top K</label>
            <select
              value={topK}
              onChange={(e) => setTopK(Number(e.target.value))}
              className="w-full p-2 border border-gray-300 rounded-lg text-sm"
            >
              {[3, 5, 10, 15, 20].map((k) => (
                <option key={k} value={k}>{k} results</option>
              ))}
            </select>
          </div>
        </div>
      </div>

      {/* Search Interface */}
      <div className="card">
        <form onSubmit={handleSearch} className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Search Query
            </label>
            <div className="relative">
              <input
                type="text"
                value={query}
                onChange={(e) => setQuery(e.target.value)}
                placeholder="Enter your search query..."
                className="w-full p-4 pr-12 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
              />
              <SearchIcon className="absolute right-4 top-4 w-5 h-5 text-gray-400" />
            </div>
          </div>

          <motion.button
            whileHover={{ scale: searching ? 1 : 1.02 }}
            whileTap={{ scale: searching ? 1 : 0.98 }}
            type="submit"
            disabled={searching || !query.trim()}
            className={`w-full flex items-center justify-center px-6 py-3 rounded-lg font-medium transition-colors ${
              searching || !query.trim()
                ? 'bg-gray-300 cursor-not-allowed'
                : 'bg-primary-600 hover:bg-primary-700 text-white'
            }`}
          >
            {searching ? (
              <>
                <Clock className="w-5 h-5 mr-2 animate-spin" />
                Searching...
              </>
            ) : (
              <>
                <SearchIcon className="w-5 h-5 mr-2" />
                Search Documents
              </>
            )}
          </motion.button>
        </form>
      </div>

      {/* Search Results */}
      <AnimatePresence>
        {searchResults && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="card"
          >
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center">
                <BarChart3 className="w-5 h-5 text-primary-600 mr-2" />
                <h3 className="text-lg font-semibold text-gray-900">Search Results</h3>
              </div>
              <div className="text-sm text-gray-500">
                {searchResults.total_results} results • {searchResults.rag_technique}
              </div>
            </div>

            <div className="mb-4 p-3 bg-gray-50 rounded-lg">
              <p className="text-sm text-gray-600">
                <span className="font-medium">Query:</span> "{searchResults.query}"
              </p>
              <p className="text-sm text-gray-600 mt-1">
                <span className="font-medium">Model:</span> {searchResults.embedding_model} • 
                <span className="font-medium ml-2">Strategy:</span> {searchResults.chunking_strategy}
              </p>
            </div>

            <div className="space-y-4">
              {searchResults.results.map((result, index) => (
                <motion.div
                  key={index}
                  initial={{ opacity: 0, x: -20 }}
                  animate={{ opacity: 1, x: 0 }}
                  transition={{ delay: index * 0.1 }}
                  className="border border-gray-200 rounded-lg p-4 hover:shadow-md transition-shadow"
                >
                  <div className="flex items-start justify-between mb-3">
                    <div className="flex items-center">
                      <FileText className="w-4 h-4 text-gray-400 mr-2 mt-1" />
                      <div>
                        <p className="font-medium text-gray-900">{result.filename}</p>
                        <p className="text-sm text-gray-500">Chunk {result.chunk_index + 1}</p>
                      </div>
                    </div>
                    <div className={`px-2 py-1 rounded-full text-xs font-medium ${getSimilarityColor(result.similarity_score)}`}>
                      {(result.similarity_score * 100).toFixed(1)}% match
                    </div>
                  </div>
                  
                  <div className="bg-gray-50 rounded-lg p-3">
                    <p className="text-gray-700 text-sm leading-relaxed">
                      {result.chunk_text}
                    </p>
                  </div>

                  {result.chunk_metadata && Object.keys(result.chunk_metadata).length > 0 && (
                    <div className="mt-2 flex items-center text-xs text-gray-500">
                      <span className="mr-4">
                        Length: {result.chunk_metadata.chunk_length} chars
                      </span>
                      <span>
                        Words: {result.chunk_metadata.chunk_words}
                      </span>
                    </div>
                  )}
                </motion.div>
              ))}
            </div>

            {searchResults.results.length === 0 && (
              <div className="text-center py-8 text-gray-500">
                <SearchIcon className="w-12 h-12 mx-auto mb-2 opacity-50" />
                <p>No results found for your query.</p>
                <p className="text-sm">Try different keywords or generate embeddings first.</p>
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};

export default Search;
/**
 * RAG Search Interface - Teaching Semantic Search and Answer Generation
 * Updated: 2025-07-15 (Phase 2 refactoring complete)
 * 
 * Teaching Purpose: This component demonstrates the complete RAG workflow:
 * 
 * 1. QUERY PROCESSING: User enters natural language question
 * 2. SEMANTIC SEARCH: Convert query to vector, find similar content chunks  
 * 3. CONTEXT RETRIEVAL: Display ranked results with similarity scores
 * 4. ANSWER GENERATION: Use LLM to synthesize answer from retrieved context
 * 
 * Key RAG Concepts:
 * - Embedding model choice affects search quality and speed
 * - Chunking strategy impacts what information gets retrieved
 * - top_k parameter balances precision vs context richness
 * - Similarity scores show how "related" content is to the query
 * - LLM takes retrieved chunks and generates human-readable answers
 */

import React from 'react';
import { useQuery } from 'react-query';
import { motion, AnimatePresence } from 'framer-motion';
import { Search as SearchIcon, Clock, Target, FileText, BarChart3, Brain, Database } from 'lucide-react';
import { useRagSearch } from '../hooks/useRagSearch';
import ErrorNotification from '../components/ErrorNotification';

const Search = ({ apiKey }) => {
  // Use custom RAG search hook for improved state management
  const {
    searchConfig,
    updateConfig,
    searchResults,
    answerResult,
    performSearch,
    generateAnswer,
    isSearching,
    isGeneratingAnswer,
    error,
    clearError
  } = useRagSearch();

  // Available models and techniques queries  
  const { data: models } = useQuery(
    'embeddingModels',
    async () => {
      const apiService = (await import('../services/api')).default;
      return apiService.getAvailableModels();
    },
    { enabled: !!apiKey }
  );

  const { data: techniques } = useQuery(
    'ragTechniques',
    async () => {
      const apiService = (await import('../services/api')).default;
      return apiService.getRagTechniques();
    },
    { enabled: !!apiKey }
  );

  const chunkingStrategies = [
    { value: 'fixed_size', label: 'Fixed Size', description: '512 tokens with 50 token overlap' }
  ];

  const handleSearch = (e) => {
    e.preventDefault();
    if (!searchConfig.query.trim()) return;
    performSearch();
  };

  const handleGenerateAnswer = (e) => {
    e.preventDefault(); 
    if (!searchConfig.query.trim()) return;
    generateAnswer();
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

      {/* Error Notification */}
      {error && (
        <ErrorNotification 
          error={error}
          onDismiss={clearError}
          className="mb-4"
        />
      )}

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
              value={searchConfig.embedding_model}
              onChange={(e) => updateConfig({ embedding_model: e.target.value })}
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
              value={searchConfig.chunking_strategy}
              onChange={(e) => updateConfig({ chunking_strategy: e.target.value })}
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
              value={searchConfig.rag_technique}
              onChange={(e) => updateConfig({ rag_technique: e.target.value })}
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
              value={searchConfig.top_k}
              onChange={(e) => updateConfig({ top_k: Number(e.target.value) })}
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
                value={searchConfig.query}
                onChange={(e) => updateConfig({ query: e.target.value })}
                placeholder="Enter your search query..."
                className="w-full p-4 pr-12 border border-gray-300 rounded-lg focus:ring-2 focus:ring-primary-500 focus:border-transparent"
              />
              <SearchIcon className="absolute right-4 top-4 w-5 h-5 text-gray-400" />
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
            {/* Retrieve Context Button */}
            <motion.button
              whileHover={{ scale: isSearching ? 1 : 1.02 }}
              whileTap={{ scale: isSearching ? 1 : 0.98 }}
              type="button"
              onClick={handleSearch}
              disabled={isSearching || isGeneratingAnswer || !searchConfig.query.trim()}
              className={`flex items-center justify-center px-6 py-3 rounded-lg font-medium transition-colors ${
                isSearching || isGeneratingAnswer || !searchConfig.query.trim()
                  ? 'bg-gray-300 cursor-not-allowed'
                  : 'bg-blue-600 hover:bg-blue-700 text-white'
              }`}
            >
              {isSearching ? (
                <>
                  <Clock className="w-5 h-5 mr-2 animate-spin" />
                  Searching...
                </>
              ) : (
                <>
                  <Database className="w-5 h-5 mr-2" />
                  🔍 Retrieve Context
                </>
              )}
            </motion.button>

            {/* Generate Response Button */}
            <motion.button
              whileHover={{ scale: isGeneratingAnswer ? 1 : 1.02 }}
              whileTap={{ scale: isGeneratingAnswer ? 1 : 0.98 }}
              type="button"
              onClick={handleGenerateAnswer}
              disabled={isSearching || isGeneratingAnswer || !searchConfig.query.trim()}
              className={`flex items-center justify-center px-6 py-3 rounded-lg font-medium transition-colors ${
                isSearching || isGeneratingAnswer || !searchConfig.query.trim()
                  ? 'bg-gray-300 cursor-not-allowed'
                  : 'bg-green-600 hover:bg-green-700 text-white'
              }`}
            >
              {isGeneratingAnswer ? (
                <>
                  <Clock className="w-5 h-5 mr-2 animate-spin" />
                  Generating...
                </>
              ) : (
                <>
                  <Brain className="w-5 h-5 mr-2" />
                  💡 Generate Response
                </>
              )}
            </motion.button>
          </div>
        </form>
      </div>

      {/* Answer Result */}
      <AnimatePresence>
        {answerResult && (
          <motion.div
            initial={{ opacity: 0, y: 20 }}
            animate={{ opacity: 1, y: 0 }}
            className="card"
          >
            <div className="flex items-center justify-between mb-4">
              <div className="flex items-center">
                <Brain className="w-5 h-5 text-green-600 mr-2" />
                <h3 className="text-lg font-semibold text-gray-900">AI Generated Answer</h3>
              </div>
              <div className="flex items-center space-x-3 text-sm text-gray-500">
                <span>Confidence: {(answerResult.confidence * 100).toFixed(1)}%</span>
                <span>•</span>
                <span>{answerResult.generation_time.toFixed(1)}s</span>
                <span>•</span>
                <span>{answerResult.model_used}</span>
              </div>
            </div>

            <div className="mb-4 p-3 bg-gray-50 rounded-lg">
              <p className="text-sm text-gray-600">
                <span className="font-medium">Query:</span> "{answerResult.query}"
              </p>
            </div>

            {/* Generated Answer */}
            <div className="mb-6 p-4 bg-green-50 border border-green-200 rounded-lg">
              <h4 className="font-semibold text-green-900 mb-2">Answer:</h4>
              <p className="text-green-800 leading-relaxed">{answerResult.answer}</p>
            </div>

            {/* Source Citations */}
            {answerResult.sources && answerResult.sources.length > 0 && (
              <div>
                <h4 className="font-semibold text-gray-900 mb-3">📚 Sources Used ({answerResult.sources.length})</h4>
                <div className="space-y-3">
                  {answerResult.sources.map((source, index) => (
                    <motion.div
                      key={index}
                      initial={{ opacity: 0, x: -20 }}
                      animate={{ opacity: 1, x: 0 }}
                      transition={{ delay: index * 0.1 }}
                      className="border border-gray-200 rounded-lg p-3"
                    >
                      <div className="flex items-start justify-between mb-2">
                        <div className="flex items-center">
                          <FileText className="w-4 h-4 text-gray-400 mr-2 mt-0.5" />
                          <div>
                            <p className="font-medium text-gray-900 text-sm">{source.file_name}</p>
                            <p className="text-xs text-gray-500">Chunk {source.chunk_index + 1}</p>
                          </div>
                        </div>
                        <div className={`px-2 py-1 rounded-full text-xs font-medium ${getSimilarityColor(source.similarity)}`}>
                          {(source.similarity * 100).toFixed(1)}%
                        </div>
                      </div>
                      <div className="bg-gray-50 rounded p-2">
                        <p className="text-gray-700 text-xs leading-relaxed line-clamp-3">
                          {source.chunk_text}
                        </p>
                      </div>
                    </motion.div>
                  ))}
                </div>
              </div>
            )}
          </motion.div>
        )}
      </AnimatePresence>

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
                        <p className="font-medium text-gray-900">{result.file_name}</p>
                        <p className="text-sm text-gray-500">Chunk {result.chunk_index + 1}</p>
                      </div>
                    </div>
                    <div className={`px-2 py-1 rounded-full text-xs font-medium ${getSimilarityColor(result.similarity)}`}>
                      {(result.similarity * 100).toFixed(1)}% match
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
import React from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { TrendingUp, Clock, CheckCircle } from 'lucide-react';

/**
 * EmbeddingProgress Component
 * 
 * Teaching Purpose: Shows real-time progress of RAG embedding generation
 * - Progress tracking with file counts and token processing
 * - Visual progress bars for user feedback
 * - Live metrics during background processing
 */
const EmbeddingProgress = ({ metrics }) => {
  if (!metrics?.active) return null;

  return (
    <AnimatePresence>
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
    </AnimatePresence>
  );
};

export default EmbeddingProgress;
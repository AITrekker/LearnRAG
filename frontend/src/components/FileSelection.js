import React from 'react';
import { motion } from 'framer-motion';
import { FileText, CheckCircle } from 'lucide-react';
import { formatFileSize } from '../utils';

/**
 * FileSelection Component
 * 
 * Teaching Purpose: Demonstrates file selection UI for RAG processing
 * - Multi-select file interface with visual feedback
 * - File metadata display (size, type)
 * - Batch selection controls (select all/clear)
 */
const FileSelection = ({ files, selectedFiles, onToggleFile, onClearSelection, onSelectAll }) => {
  return (
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
            onClick={() => onToggleFile(file.id)}
          >
            <div className="flex items-center justify-between">
              <div>
                <p className="font-medium text-gray-900">{file.name}</p>
                <p className="text-sm text-gray-500">
                  {formatFileSize(file.size)} • {file.type}
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
          onClick={onClearSelection}
          className="text-sm text-gray-500 hover:text-gray-700"
        >
          Clear selection
        </button>
        <button
          onClick={() => onSelectAll(files?.map(f => f.id) || [])}
          className="text-sm text-primary-600 hover:text-primary-700"
        >
          Select all
        </button>
      </div>
    </div>
  );
};

export default FileSelection;
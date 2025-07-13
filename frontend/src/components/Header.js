import React from 'react';
import { motion } from 'framer-motion';
import { Brain, LogOut } from 'lucide-react';

const Header = ({ onLogout, tenant }) => {
  return (
    <motion.header
      initial={{ y: -50, opacity: 0 }}
      animate={{ y: 0, opacity: 1 }}
      className="bg-white shadow-sm border-b border-gray-200 fixed top-0 left-0 right-0 z-50"
    >
      <div className="px-6 py-4 flex items-center justify-between">
        <div className="flex items-center">
          <motion.div
            animate={{ rotate: [0, 10, -10, 0] }}
            transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
          >
            <Brain className="w-8 h-8 text-primary-600 mr-3" />
          </motion.div>
          <h1 className="text-xl font-bold text-gray-900">LearnRAG</h1>
          <span className="ml-2 text-sm text-gray-500">
            {tenant ? `${tenant.name} • Interactive RAG Learning Platform` : 'Interactive RAG Learning Platform'}
          </span>
        </div>

        <button
          onClick={onLogout}
          className="flex items-center text-gray-600 hover:text-gray-900 transition-colors"
        >
          <LogOut className="w-4 h-4 mr-1" />
          <span className="text-sm">Logout</span>
        </button>
      </div>
    </motion.header>
  );
};

export default Header;
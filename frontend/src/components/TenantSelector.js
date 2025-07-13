import React from 'react';
import { motion } from 'framer-motion';
import { Building, Key, User } from 'lucide-react';

const TenantSelector = ({ tenants, onSelect, onManualEntry }) => {
  return (
    <motion.div
      initial={{ opacity: 0, scale: 0.9 }}
      animate={{ opacity: 1, scale: 1 }}
      className="card max-w-lg mx-auto"
    >
      <div className="text-center mb-6">
        <motion.div
          animate={{ rotate: [0, 10, -10, 0] }}
          transition={{ duration: 2, repeat: Infinity, ease: "easeInOut" }}
        >
          <Building className="w-12 h-12 text-primary-600 mx-auto mb-4" />
        </motion.div>
        <h2 className="text-2xl font-bold text-gray-900 mb-2">Select Your Tenant</h2>
        <p className="text-gray-600">Choose which organization you'd like to access</p>
      </div>

      <div className="space-y-3 mb-6">
        {tenants.map((tenant, index) => (
          <motion.button
            key={tenant.slug}
            initial={{ opacity: 0, x: -20 }}
            animate={{ opacity: 1, x: 0 }}
            transition={{ delay: index * 0.1 }}
            onClick={() => onSelect(tenant)}
            className="w-full p-4 bg-gradient-to-r from-primary-50 to-secondary-50 hover:from-primary-100 hover:to-secondary-100 border border-gray-200 hover:border-primary-300 rounded-lg transition-all duration-200 text-left"
          >
            <div className="flex items-center justify-between">
              <div className="flex items-center">
                <div className="w-10 h-10 bg-primary-600 rounded-lg flex items-center justify-center mr-3">
                  <User className="w-5 h-5 text-white" />
                </div>
                <div>
                  <h3 className="font-semibold text-gray-900">{tenant.name}</h3>
                  <p className="text-sm text-gray-500">Slug: {tenant.slug}</p>
                </div>
              </div>
              <div className="text-primary-600">
                <Key className="w-5 h-5" />
              </div>
            </div>
          </motion.button>
        ))}
      </div>

      <div className="border-t border-gray-200 pt-4">
        <button
          onClick={onManualEntry}
          className="w-full text-sm text-gray-500 hover:text-gray-700 transition-colors"
        >
          Or enter API key manually
        </button>
      </div>
    </motion.div>
  );
};

export default TenantSelector;
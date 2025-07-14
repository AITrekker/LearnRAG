import React, { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from 'react-query';
import { motion } from 'framer-motion';

import Header from './components/Header';
import Sidebar from './components/Sidebar';
import Dashboard from './pages/Dashboard';
import Embeddings from './pages/Embeddings';
import Search from './pages/Search';
import ApiKeyModal from './components/ApiKeyModal';
import TenantSelector from './components/TenantSelector';

const queryClient = new QueryClient();

function App() {
  const [apiKey, setApiKey] = useState('');
  const [tenants, setTenants] = useState([]);
  const [selectedTenant, setSelectedTenant] = useState(null);
  const [showApiKeyModal, setShowApiKeyModal] = useState(false);
  const [showTenantSelector, setShowTenantSelector] = useState(false);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    // Check for available tenants from API keys file
    const loadTenants = async () => {
      try {
        const response = await fetch('/api_keys.json');
        if (response.ok) {
          const data = await response.json();
          if (data.tenants && data.tenants.length > 0) {
            setTenants(data.tenants);
            
            // If only one tenant, auto-login
            if (data.tenants.length === 1) {
              const tenant = data.tenants[0];
              console.log('Auto-selecting single tenant:', tenant.name);
              setSelectedTenant(tenant);
              setApiKey(tenant.api_key);
              localStorage.setItem('apiKey', tenant.api_key);
              localStorage.setItem('selectedTenant', JSON.stringify(tenant));
            } else {
              // Multiple tenants - show selector
              setShowTenantSelector(true);
            }
          } else {
            // No tenants available - show manual API key entry
            setShowApiKeyModal(true);
          }
        } else {
          // API keys file not found - show manual entry
          setShowApiKeyModal(true);
        }
      } catch (error) {
        console.log('Could not load tenants from API keys file, showing manual entry');
        setShowApiKeyModal(true);
      } finally {
        setLoading(false);
      }
    };

    // Clear any existing localStorage and start fresh
    localStorage.removeItem('apiKey');
    localStorage.removeItem('selectedTenant');
    loadTenants();
  }, []);

  const handleApiKeySubmit = (key) => {
    setApiKey(key);
    localStorage.setItem('apiKey', key);
    setShowApiKeyModal(false);
  };

  const handleTenantSelect = (tenant) => {
    console.log('Selecting tenant:', tenant);
    setSelectedTenant(tenant);
    setApiKey(tenant.api_key);
    localStorage.setItem('apiKey', tenant.api_key);
    localStorage.setItem('selectedTenant', JSON.stringify(tenant));
    setShowTenantSelector(false);
    console.log('API key set:', tenant.api_key.substring(0, 10) + '...');
  };

  const handleLogout = () => {
    setApiKey('');
    setSelectedTenant(null);
    setTenants([]);
    localStorage.removeItem('apiKey');
    localStorage.removeItem('selectedTenant');
    setLoading(true);
    // Restart the tenant loading process
    window.location.reload();
  };

  if (loading) {
    return (
      <QueryClientProvider client={queryClient}>
        <div className="min-h-screen gradient-bg flex items-center justify-center">
          <div className="text-center">
            <div className="animate-spin w-8 h-8 border-4 border-primary-600 border-t-transparent rounded-full mx-auto mb-4"></div>
            <p className="text-gray-600">Loading tenants...</p>
          </div>
        </div>
      </QueryClientProvider>
    );
  }

  if (showTenantSelector) {
    return (
      <QueryClientProvider client={queryClient}>
        <div className="min-h-screen gradient-bg flex items-center justify-center">
          <TenantSelector 
            tenants={tenants} 
            onSelect={handleTenantSelect}
            onManualEntry={() => {
              setShowTenantSelector(false);
              setShowApiKeyModal(true);
            }}
          />
        </div>
      </QueryClientProvider>
    );
  }

  if (showApiKeyModal) {
    return (
      <QueryClientProvider client={queryClient}>
        <div className="min-h-screen gradient-bg flex items-center justify-center">
          <ApiKeyModal onSubmit={handleApiKeySubmit} />
        </div>
      </QueryClientProvider>
    );
  }

  return (
    <QueryClientProvider client={queryClient}>
      <Router future={{ v7_relativeSplatPath: true, v7_startTransition: true }}>
        <div className="min-h-screen bg-gray-50">
          <Header onLogout={handleLogout} tenant={selectedTenant} />
          
          <div className="flex">
            <Sidebar />
            
            <main className="flex-1 ml-64 pt-20 p-6">
              <motion.div
                initial={{ opacity: 0, y: 20 }}
                animate={{ opacity: 1, y: 0 }}
                transition={{ duration: 0.5 }}
              >
                <Routes>
                  <Route path="/" element={<Dashboard apiKey={apiKey} />} />
                  <Route path="/embeddings" element={<Embeddings apiKey={apiKey} />} />
                  <Route path="/search" element={<Search apiKey={apiKey} />} />
                </Routes>
              </motion.div>
            </main>
          </div>
        </div>
      </Router>
    </QueryClientProvider>
  );
}

export default App;
/**
 * API Service - Centralized HTTP Client for RAG Operations
 * 
 * This service demonstrates modern API communication patterns:
 * 
 * 1. AXIOS CONFIGURATION: Centralized HTTP client setup
 * 2. AUTHENTICATION: Automatic API key injection
 * 3. ERROR HANDLING: Standardized error processing and logging
 * 4. REQUEST INTERCEPTORS: Automatic header management
 * 5. RESPONSE INTERCEPTORS: Consistent error handling across all requests
 * 
 * Core Frontend API Concepts Illustrated:
 * - Singleton pattern for shared HTTP client
 * - Automatic authentication header injection
 * - Centralized error handling and logging
 * - Method binding for consistent context
 * - Timeout configuration for reliable operations
 */

import axios from 'axios';
import { parseApiError, logError } from '../utils/errorHandling';
import { API_BASE_URL, API_TIMEOUT } from '../config';

class ApiService {
  /**
   * Initialize API service with configured HTTP client
   * 
   * WHY CENTRALIZED API SERVICE?
   * - Single point of configuration for all HTTP requests
   * - Consistent error handling across the application
   * - Automatic authentication header management
   * - Standardized timeout and retry policies
   * 
   * METHOD BINDING:
   * - Ensures 'this' context is preserved in React components
   * - Prevents issues when methods are passed as callbacks
   * - Maintains consistent API surface across components
   */
  constructor() {
    this.initializeClient();
    
    // Bind all methods to preserve 'this' context
    this.validateApiKey = this.validateApiKey.bind(this);
    this.getTenantInfo = this.getTenantInfo.bind(this);
    this.syncFiles = this.syncFiles.bind(this);
    this.getTenantFiles = this.getTenantFiles.bind(this);
    this.getTenantStats = this.getTenantStats.bind(this);
    this.generateEmbeddings = this.generateEmbeddings.bind(this);
    this.getAvailableModels = this.getAvailableModels.bind(this);
    this.getEmbeddingStatus = this.getEmbeddingStatus.bind(this);
    this.deleteEmbeddings = this.deleteEmbeddings.bind(this);
    this.search = this.search.bind(this);
    this.generateAnswer = this.generateAnswer.bind(this);
    this.getRagTechniques = this.getRagTechniques.bind(this);
    this.getRagSessions = this.getRagSessions.bind(this);
    this.getLlmModels = this.getLlmModels.bind(this);
    this.getPromptTemplates = this.getPromptTemplates.bind(this);
    this.getEmbeddingSettings = this.getEmbeddingSettings.bind(this);
    this.updateEmbeddingSettings = this.updateEmbeddingSettings.bind(this);
    this.getChunkingStrategies = this.getChunkingStrategies.bind(this);
    this.getCurrentMetrics = this.getCurrentMetrics.bind(this);
  }

  initializeClient() {
    /**
     * Initialize HTTP client with interceptors - Request/Response Pipeline
     * 
     * WHY AXIOS INTERCEPTORS?
     * - Automatic authentication header injection
     * - Centralized error handling and logging
     * - Request/response transformation
     * - Consistent timeout and retry policies
     * 
     * REQUEST INTERCEPTOR:
     * - Automatically adds API key from localStorage
     * - Ensures all requests are authenticated
     * - Maintains consistent headers across requests
     * 
     * RESPONSE INTERCEPTOR:
     * - Standardizes error handling across all API calls
     * - Parses error responses into consistent format
     * - Logs errors for debugging and monitoring
     */
    this.client = axios.create({
      baseURL: `${API_BASE_URL}/api`,
      timeout: API_TIMEOUT,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Add request interceptor to include API key
    this.client.interceptors.request.use((config) => {
      const apiKey = localStorage.getItem('apiKey');
      if (apiKey) {
        config.headers['X-API-Key'] = apiKey;
      }
      return config;
    });

    // Add response interceptor for standardized error handling
    this.client.interceptors.response.use(
      (response) => response,
      (error) => {
        // Parse and log error using standardized utilities
        const parsedError = parseApiError(error);
        logError(error, { 
          url: error.config?.url,
          method: error.config?.method,
          timestamp: new Date().toISOString()
        });
        
        // Handle authentication errors by clearing stored credentials
        if (parsedError.error_type === 'auth') {
          localStorage.removeItem('apiKey');
          localStorage.removeItem('selectedTenant');
          window.location.reload();
        }
        
        return Promise.reject(error);
      }
    );
  }

  // Auth
  async validateApiKey(apiKey) {
    const response = await this.client.get('/auth/validate', {
      headers: { 'X-API-Key': apiKey }
    });
    return response.data;
  }

  // Tenants
  async getTenantInfo() {
    if (!this.client) {
      this.initializeClient();
    }
    const response = await this.client.get('/tenants/info');
    return response.data;
  }

  async syncFiles() {
    if (!this.client) {
      this.initializeClient();
    }
    const response = await this.client.post('/tenants/sync-files');
    return response.data;
  }

  async getTenantFiles() {
    if (!this.client) {
      this.initializeClient();
    }
    const response = await this.client.get('/tenants/files');
    return response.data;
  }

  async getTenantStats() {
    if (!this.client) {
      this.initializeClient();
    }
    const response = await this.client.get('/tenants/stats');
    return response.data;
  }

  // Embeddings
  async generateEmbeddings(data) {
    const response = await this.client.post('/embeddings/generate', data);
    return response.data;
  }

  async getAvailableModels() {
    const response = await this.client.get('/embeddings/models');
    return response.data;
  }

  async getEmbeddingStatus(fileId) {
    const response = await this.client.get(`/embeddings/status/${fileId}`);
    return response.data;
  }

  async deleteEmbeddings(fileId) {
    const response = await this.client.delete(`/embeddings/${fileId}`);
    return response.data;
  }

  // RAG
  async search(data) {
    const response = await this.client.post('/rag/search', data);
    return response.data;
  }

  async generateAnswer(data) {
    const response = await this.client.post('/rag/answer', data);
    return response.data;
  }

  async getRagTechniques() {
    const response = await this.client.get('/rag/techniques');
    return response.data;
  }

  async getRagSessions(limit = 10) {
    const response = await this.client.get(`/rag/sessions?limit=${limit}`);
    return response.data;
  }

  async getLlmModels() {
    const response = await this.client.get('/rag/llm-models');
    return response.data;
  }

  async getPromptTemplates() {
    const response = await this.client.get('/rag/prompt-templates');
    return response.data;
  }

  // Embedding Settings
  async getEmbeddingSettings() {
    const response = await this.client.get('/tenants/embedding-settings');
    return response.data;
  }

  async updateEmbeddingSettings(data) {
    const response = await this.client.post('/tenants/embedding-settings', data);
    return response.data;
  }

  async getChunkingStrategies() {
    const response = await this.client.get('/embeddings/chunking-strategies');
    return response.data;
  }

  async getCurrentMetrics() {
    const response = await this.client.get('/embeddings/metrics/current');
    return response.data;
  }
}

const apiService = new ApiService();
export default apiService;
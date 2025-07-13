import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

class ApiService {
  constructor() {
    this.client = axios.create({
      baseURL: `${API_BASE_URL}/api`,
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

    // Add response interceptor for error handling
    this.client.interceptors.response.use(
      (response) => response,
      (error) => {
        console.error('API Error:', error);
        if (error.response?.status === 401) {
          console.log('Unauthorized, clearing API key');
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
    const response = await this.client.get('/tenants/info');
    return response.data;
  }

  async syncFiles() {
    const response = await this.client.post('/tenants/sync-files');
    return response.data;
  }

  async getTenantFiles() {
    const response = await this.client.get('/tenants/files');
    return response.data;
  }

  async getTenantStats() {
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

  async getRagTechniques() {
    const response = await this.client.get('/rag/techniques');
    return response.data;
  }

  async getRagSessions(limit = 10) {
    const response = await this.client.get(`/rag/sessions?limit=${limit}`);
    return response.data;
  }
}

const apiService = new ApiService();
export default apiService;
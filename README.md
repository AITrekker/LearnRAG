# LearnRAG - Interactive RAG Learning Platform

A dockerized educational platform for learning and experimenting with different RAG (Retrieval-Augmented Generation) techniques and embedding models.

## Phase 1 - Foundation ✅ COMPLETE

### Features Implemented
- ✅ **Multi-tenant architecture** with API-key authentication
- ✅ **Auto-discovery** of tenants from `demo_data/` folder structure  
- ✅ **File processing** for text, PDF, DOC, XLS, PPT formats
- ✅ **Delta sync** - only re-embed when files or models change
- ✅ **PostgreSQL + pgvector** for vector storage with dynamic dimensions
- ✅ **5 embedding models** with automatic caching (384d-768d dimensions)
- ✅ **3 chunking strategies** (fixed_size, sentence, recursive)
- ✅ **FastAPI backend** with comprehensive API endpoints
- ✅ **Modern React frontend** with real-time progress tracking
- ✅ **Interactive embedding generation** with file-by-file progress
- ✅ **Similarity search** using pgvector cosine distance
- ✅ **Comprehensive test suite** with 85%+ API coverage

### Tech Stack
- **Backend**: FastAPI, PostgreSQL, pgvector, sentence-transformers, SQLAlchemy
- **Frontend**: React, React Query, Framer Motion, Tailwind CSS
- **Infrastructure**: Docker Compose with GPU/CPU support
- **Models**: 5 HuggingFace models (all-MiniLM-L6-v2, paraphrase variants, e5-small)
- **Testing**: Comprehensive API test suite with validation

## Quick Start

### 1. Initial Setup
```bash
# Clone and enter directory
cd LearnRAG

# For GPU support (RTX 5070 optimized)
docker-compose up --build

# For CPU-only (fallback)
docker-compose -f docker-compose.cpu.yml up --build
```

### 2. Access the Application
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

### 3. Smart Login Experience
The frontend automatically detects available tenants:

**First Time Setup**: 
- Database is empty → Auto-discovers tenants from your `demo_data/` folders
- Creates API keys and writes them to `api_keys.json`
- Frontend loads tenants automatically

**Single Tenant**: Auto-login (no modal needed)
**Multiple Tenants**: Shows tenant selection screen  
**No Tenants**: Falls back to manual API key entry

### 4. Use the Platform
1. **Dashboard**: View tenant info and file statistics
2. **Sync Files**: Process demo data (smart delta sync - only new/changed files)
3. **Configure Embeddings**: Choose from 5 models and 3 chunking strategies
4. **Generate Embeddings**: Create vector embeddings with real-time progress tracking
5. **Search & RAG**: Test similarity search with configurable parameters

### 5. API Keys Reference
Check `api_keys.json` for all available tenants:
```json
{
  "tenants": [
    {
      "slug": "ACMECorp",
      "name": "ACMECorp", 
      "api_key": "lr_abc123..."
    }
  ]
}
```

## Project Structure

```
LearnRAG/
├── docker-compose.yml          # Container orchestration
├── demo_data/                  # Your demo data (auto-discovered)
│   ├── ACMECorp/              # Tenant folder = tenant slug
│   ├── InnovateFast/          
│   └── RegionalSolns/         
├── backend/                    # FastAPI application
│   ├── app/
│   │   ├── main.py            # FastAPI app
│   │   ├── models/            # Database & Pydantic models
│   │   ├── routers/           # API endpoints
│   │   └── services/          # Business logic
│   └── requirements.txt       
├── frontend/                   # React application
│   ├── src/
│   │   ├── components/        # Reusable UI components
│   │   ├── pages/             # Main application pages
│   │   └── services/          # API client
│   └── package.json
└── data/                      # Persistent volumes
    ├── postgres/              # Database data
    └── files/                 # Internal file storage
```

## API Endpoints

### Authentication
- `GET /api/auth/validate` - Validate API key

### Tenants  
- `GET /api/tenants/info` - Get tenant information
- `POST /api/tenants/sync-files` - Sync files from demo_data
- `GET /api/tenants/files` - List tenant files
- `GET /api/tenants/stats` - Get tenant statistics
- `GET /api/tenants/embedding-settings` - Get embedding configuration
- `POST /api/tenants/embedding-settings` - Update embedding configuration
- `GET /api/tenants/embedding-summary` - Get embedding status summary

### Embeddings
- `POST /api/embeddings/generate` - Generate embeddings with progress tracking
- `GET /api/embeddings/models` - Available embedding models (5 models)
- `GET /api/embeddings/chunking-strategies` - Available chunking strategies (3 strategies)
- `GET /api/embeddings/status` - Current embedding generation status
- `GET /api/embeddings/metrics` - Real-time generation metrics

### RAG Operations
- `POST /api/rag/search` - Perform similarity search with configurable parameters
- `GET /api/rag/techniques` - Available RAG techniques
- `GET /api/rag/sessions` - Search history with pagination
- `POST /api/rag/compare` - Compare techniques (Phase 3 placeholder)

## Phase 2 - Enhanced RAG Features
- 🔄 Advanced RAG techniques (hybrid search, re-ranking)
- 🔄 Enhanced search interface with filters and sorting
- 🔄 Multi-query search capabilities
- 🔄 Search result explanations and relevance scoring

## Phase 3 - Advanced Analytics
- 🔄 Side-by-side RAG technique comparison
- 🔄 A/B testing framework for RAG techniques
- 🔄 Performance metrics and analytics dashboard
- 🔄 Export capabilities for research and analysis

## Development

### Testing
```bash
# Run comprehensive API test suite
python3 run_all_tests.py

# Test coverage: 18/21 tests passing (85%+)
# Validates all endpoints, authentication, and data integrity
```

### Logs
```bash
# View all logs
docker-compose logs -f

# View specific service
docker-compose logs -f backend
docker-compose logs -f frontend
```

### Database Access
```bash
# Connect to PostgreSQL
docker-compose exec postgres psql -U postgres -d learnrag

# View embedding tables
\d embeddings
\d tenant_embedding_settings
```

### Rebuild After Changes
```bash
# Rebuild and restart
docker-compose down
docker-compose up --build
```

## Notes
- Model downloads happen on first use and are cached in persistent volume
- Files are copied to internal storage for processing
- Delta sync prevents re-processing unchanged files
- API keys are generated automatically for each tenant folder
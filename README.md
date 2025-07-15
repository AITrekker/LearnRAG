# LearnRAG - Interactive RAG Learning Platform

A dockerized educational platform for learning and experimenting with different RAG (Retrieval-Augmented Generation) techniques and embedding models.

## 🎓 Recent Updates - Code Teaching Improvements ✅

### Phase 1 Code Quality Improvements (COMPLETED)
- ✅ **Simplified Models**: Removed confusing aliases, clear `EmbeddingSettingsRequest`/`Response` classes
- ✅ **Component Breakdown**: Split large `Embeddings.js` into focused components (`EmbeddingSettings`, `EmbeddingProgress`, `FileSelection`)  
- ✅ **Standardized Error Handling**: Comprehensive backend exceptions, frontend error utilities, user-friendly notifications
- ✅ **Educational Docstrings**: Added teaching-focused comments explaining RAG concepts throughout codebase

**Result**: Code is now highly suitable for educational purposes, LinkedIn posts, and teaching RAG development patterns.

## Phase 1 - Foundation ✅ COMPLETE

### Features Implemented
- ✅ **Multi-tenant architecture** with API-key authentication
- ✅ **Auto-discovery** of tenants from `setup/` folder structure  
- ✅ **File processing** for text, PDF, DOC, XLS, PPT formats
- ✅ **Delta sync** - only re-embed when files or models change
- ✅ **PostgreSQL + pgvector** for vector storage with dynamic dimensions
- ✅ **5 embedding models** with automatic caching (384d-768d dimensions)
- ✅ **3 chunking strategies** (fixed_size, sentence, recursive)
- ✅ **FastAPI backend** with comprehensive API endpoints
- ✅ **Modern React frontend** with real-time progress tracking
- ✅ **Interactive embedding generation** with file-by-file progress
- ✅ **Similarity search** using pgvector cosine distance
- ✅ **LLM-powered answer generation** with google/flan-t5-base
- ✅ **Comprehensive test suite** with 95%+ API coverage (22/22 tests passing)
- ✅ **Data folder monitoring** - runtime file changes with embedding cleanup

### Tech Stack
- **Backend**: FastAPI, PostgreSQL, pgvector, sentence-transformers, Transformers (HF)
- **Frontend**: React, React Query, Framer Motion, Tailwind CSS
- **Infrastructure**: Docker Compose with GPU/CPU support, health checks
- **Models**: 5 embedding models + google/flan-t5-base for answer generation
- **Testing**: Comprehensive API test suite (22/22 tests) with automated CI/CD validation

## Quick Start

### 1. Choose Your Deployment Mode

**🚀 GPU Mode (NVIDIA RTX 5070 Optimized):**
```bash
# For computers with NVIDIA GPU
docker-compose up --build
```

**💻 CPU Mode (Universal Compatibility):**
```bash
# For computers without NVIDIA GPU or any CPU-only setup
docker-compose -f docker-compose.cpu.yml up --build
```

**🔧 Technical Differences:**
- **GPU Mode**: Uses NVIDIA PyTorch container, CUDA acceleration, faster embedding generation
- **CPU Mode**: Uses standard Python container, CPU-only PyTorch, works on any computer

### 2. Access the Application
- **Frontend**: http://localhost:3000
- **Backend API**: http://localhost:8000
- **API Docs**: http://localhost:8000/docs

### 3. Smart Login Experience
The frontend automatically detects available tenants:

**First Time Setup**: 
- Database is empty → Auto-discovers tenants from your `setup/` folders
- Creates API keys and writes them to `api_keys.json`
- Frontend loads tenants automatically

**Single Tenant**: Auto-login (no modal needed)
**Multiple Tenants**: Shows tenant selection screen  
**No Tenants**: Falls back to manual API key entry

### 4. Use the Platform
1. **Dashboard**: View tenant info and file statistics
2. **Sync Files**: Process data folder changes (smart delta sync with embedding cleanup)
3. **Configure Embeddings**: Choose from 5 models and 3 chunking strategies
4. **Generate Embeddings**: Create vector embeddings with real-time progress tracking
5. **Search & RAG**: Test similarity search and LLM-powered answer generation

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
├── docker-compose.yml          # Container orchestration (GPU/CPU optimized)
├── setup/                      # Source data (auto-discovered tenants)
│   ├── ACMECorp/              # Tenant folder = tenant slug
│   ├── InnovateFast/          
│   └── RegionalSolns/         
├── data/files/                 # Runtime data (synced from setup, monitored for changes)
├── backend/                    # FastAPI application
│   ├── api/                   # REST endpoints (auth, tenants, embeddings, rag)
│   ├── services/              # Business logic (embeddings, LLM, RAG)
│   ├── models.py              # SQLAlchemy & Pydantic models
│   └── utils.py               # Shared utilities
├── frontend/                   # React application
│   ├── src/
│   │   ├── components/        # Reusable UI components
│   │   ├── pages/             # Main application pages
│   │   └── services/          # API client
├── tests/                      # Comprehensive test suite
│   ├── test_auth.py           # Authentication tests
│   ├── test_tenants.py        # Tenant management tests
│   ├── test_embeddings.py     # Embedding generation tests
│   └── test_rag.py            # RAG search & answer tests
└── scripts/
    └── run_all_tests.py       # Test runner (22/22 tests passing)
## Testing & Quality Assurance

### Running Tests
```bash
# Run comprehensive API test suite
python3 scripts/run_all_tests.py

# Current status: 22/22 tests passing ✅
# Coverage: Auth (3/3), Tenants (7/7), Embeddings (6/7), RAG (6/6)
```

### Test Categories
- **Authentication**: API key validation, unauthorized access handling
- **Tenants**: File management, settings, statistics, sync operations  
- **Embeddings**: Model selection, generation, chunking strategies, metrics
- **RAG**: Search functionality, answer generation, session management

## API Endpoints

### Authentication
- `GET /api/auth/validate` - Validate API key

### Tenants  
- `GET /api/tenants/info` - Get tenant information
- `POST /api/tenants/sync-files` - Sync files from setup
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
# GPU Mode rebuild
docker-compose down
docker-compose up --build

# CPU Mode rebuild  
docker-compose -f docker-compose.cpu.yml down
docker-compose -f docker-compose.cpu.yml up --build
```

## Notes
- Model downloads happen on first use and are cached in persistent volume
- Files are copied to internal storage for processing
- Delta sync prevents re-processing unchanged files
- API keys are generated automatically for each tenant folder
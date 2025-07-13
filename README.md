# LearnRAG - Interactive RAG Learning Platform

A dockerized educational platform for learning and experimenting with different RAG (Retrieval-Augmented Generation) techniques and embedding models.

## Phase 1 - Foundation ✅

### Features Implemented
- ✅ **Multi-tenant architecture** with API-key authentication
- ✅ **Auto-discovery** of tenants from `demo1data/` folder structure  
- ✅ **File processing** for text, PDF, DOC, XLS, PPT formats
- ✅ **Delta sync** - only re-embed when files or models change
- ✅ **PostgreSQL + pgvector** for vector storage
- ✅ **Embedding model caching** (persists across container restarts)
- ✅ **FastAPI backend** with clean API separation
- ✅ **Modern React frontend** with animations and responsive design
- ✅ **Basic similarity search** using pgvector cosine distance

### Tech Stack
- **Backend**: FastAPI, PostgreSQL, pgvector, sentence-transformers
- **Frontend**: React, Framer Motion, Tailwind CSS
- **Infrastructure**: Docker Compose
- **Models**: sentence-transformers/all-MiniLM-L6-v2 (384d)

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
3. **Generate Embeddings**: Create vector embeddings for search
4. **Search & RAG**: Test different retrieval techniques (coming in Phase 2)

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
├── demo1data/                  # Your demo data (auto-discovered)
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

### Embeddings
- `POST /api/embeddings/generate` - Generate embeddings
- `GET /api/embeddings/models` - Available models
- `GET /api/embeddings/status/{file_id}` - Embedding status

### RAG Operations
- `POST /api/rag/search` - Perform similarity search
- `GET /api/rag/techniques` - Available techniques
- `GET /api/rag/sessions` - Search history

## Coming in Phase 2
- 🔄 Interactive embedding generation UI
- 🔄 Real-time search interface  
- 🔄 Multiple embedding models
- 🔄 Multiple chunking strategies
- 🔄 Progress tracking and visualization

## Coming in Phase 3
- 🔄 Advanced RAG techniques (hybrid search, re-ranking)
- 🔄 Side-by-side comparison interface
- 🔄 RAG technique experimentation
- 🔄 Performance metrics and analytics

## Development

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
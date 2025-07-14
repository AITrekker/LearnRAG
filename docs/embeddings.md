# LearnRAG Database Architecture & Embeddings System

## Overview

LearnRAG uses a multi-tenant PostgreSQL database with pgvector extension for storing and managing vector embeddings. The system is designed for educational purposes to demonstrate different RAG (Retrieval-Augmented Generation) techniques and embedding strategies.

## Database Schema

### Core Tables

#### `tenants`
```sql
CREATE TABLE tenants (
    id SERIAL PRIMARY KEY,
    slug VARCHAR(50) UNIQUE NOT NULL,           -- URL-friendly identifier
    name VARCHAR(100) NOT NULL,                 -- Display name
    api_key VARCHAR(255) UNIQUE NOT NULL,       -- Authentication key
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```
- **Purpose**: Multi-tenant isolation - each tenant represents a different organization/project
- **API Keys**: Auto-generated on startup (format: `lr_<base64>`)
- **Examples**: ACMECorp, InnovateFast, RegionalSolns

#### `files`
```sql
CREATE TABLE files (
    id SERIAL PRIMARY KEY,
    tenant_id INTEGER REFERENCES tenants(id) ON DELETE CASCADE,
    filename VARCHAR(255) NOT NULL,
    file_path VARCHAR(500) NOT NULL,            -- Relative path within tenant folder
    file_hash VARCHAR(64) NOT NULL,             -- SHA-256 for delta sync
    file_size BIGINT NOT NULL,
    content_type VARCHAR(50) NOT NULL,          -- MIME type
    last_modified TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(tenant_id, file_path)                -- Prevent duplicates per tenant
);
```
- **Purpose**: Track documents that can be embedded
- **Delta Sync**: Uses file_hash + last_modified to detect changes
- **Supported Types**: text/plain, PDF, DOC, XLS, PPT
- **Storage**: Files copied to `/app/internal_files/{tenant_slug}/`

#### `embeddings`
```sql
CREATE TABLE embeddings (
    id SERIAL PRIMARY KEY,
    file_id INTEGER REFERENCES files(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,              -- Order within file
    chunk_text TEXT NOT NULL,                  -- The actual text chunk
    embedding_model VARCHAR(100) NOT NULL,     -- e.g., 'sentence-transformers/all-MiniLM-L6-v2'
    chunking_strategy VARCHAR(50) NOT NULL,    -- 'fixed_size', 'sentence', 'recursive'
    embedding VECTOR,                          -- pgvector - dynamic dimensions
    chunk_metadata JSONB DEFAULT '{}',         -- Statistics and metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    UNIQUE(file_id, chunk_index, embedding_model, chunking_strategy)
);
```
- **Purpose**: Store vector embeddings with full provenance tracking
- **Vector Storage**: Uses pgvector extension, supports 384d (MiniLM) to 768d (mpnet)
- **Multi-Model**: Same file can have embeddings from different models
- **Chunk Metadata**: JSON with statistics like `{"chunk_length": 512, "chunk_words": 89}`

#### `tenant_embedding_settings`
```sql
CREATE TABLE tenant_embedding_settings (
    tenant_id INTEGER PRIMARY KEY REFERENCES tenants(id) ON DELETE CASCADE,
    embedding_model VARCHAR(100) NOT NULL DEFAULT 'sentence-transformers/all-MiniLM-L6-v2',
    chunking_strategy VARCHAR(50) NOT NULL DEFAULT 'fixed_size',
    chunk_size INTEGER DEFAULT 512,            -- For fixed_size/recursive strategies
    chunk_overlap INTEGER NOT NULL DEFAULT 50,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```
- **Purpose**: Per-tenant configuration for embedding generation
- **Settings Cascade**: When changed, ALL existing embeddings for tenant are deleted
- **Default Creation**: Auto-populated when new tenants are discovered

#### `rag_sessions`
```sql
CREATE TABLE rag_sessions (
    id SERIAL PRIMARY KEY,
    tenant_id INTEGER REFERENCES tenants(id) ON DELETE CASCADE,
    embedding_model VARCHAR(100) NOT NULL,
    chunking_strategy VARCHAR(50) NOT NULL,
    rag_technique VARCHAR(50) NOT NULL,        -- Future: 'similarity', 'mmr', 'rerank'
    query TEXT NOT NULL,
    results JSONB NOT NULL,                    -- Search results and metadata
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);
```
- **Purpose**: Track search queries and results for analytics
- **Educational Value**: Compare different RAG approaches
- **Results Format**: JSON with chunks, scores, and timing data

## Key Relationships & Design Patterns

### Tenant Isolation Through Foreign Keys

```
tenants (1) ←→ (N) files (1) ←→ (N) embeddings
tenants (1) ←→ (1) tenant_embedding_settings
tenants (1) ←→ (N) rag_sessions
```

**Critical Design Decision**: Embeddings table does NOT directly reference `tenant_id`. Instead:
```sql
-- Tenant isolation achieved through file relationship
embeddings.file_id → files.id → files.tenant_id → tenants.id
```

**Why this approach?**
- ✅ **Normalized**: No data duplication
- ✅ **Referential Integrity**: CASCADE deletes work automatically
- ✅ **File-centric**: Embeddings naturally belong to files
- ❌ **Query Performance**: Requires JOIN for tenant filtering

### Embedding Deletion Strategy

When tenant embedding settings change, the system performs a **complete cleanup**:

```sql
-- Delete ALL embeddings for a tenant when settings change
DELETE FROM embeddings 
WHERE file_id IN (
    SELECT id FROM files 
    WHERE tenant_id = $1
);
```

**This "nuclear option" ensures:**
- 🔄 **Consistency**: No mixed embeddings from different configurations
- 🧹 **Clean Slate**: Forces regeneration with new settings
- 📚 **Educational**: Clear cause-and-effect for learning

### Delta Sync & Change Detection

Files are tracked with hash-based delta sync:
```sql
-- Only re-embed if file content changed OR embedding settings changed
SELECT * FROM files 
WHERE file_hash != $new_hash 
   OR last_modified != $new_timestamp;
```

Combined with embedding settings tracking:
```python
# Delete embeddings if ANY setting changed
settings_changed = (
    old.embedding_model != new.embedding_model or
    old.chunking_strategy != new.chunking_strategy or
    old.chunk_size != new.chunk_size or
    old.chunk_overlap != new.chunk_overlap
)
```

## API Integration Patterns

### Authentication Flow
```
Request → X-API-Key Header → Tenant Lookup → Dependency Injection
```

Every API endpoint uses the same pattern:
```python
@router.get("/endpoint")
async def endpoint(
    tenant: Tenant = Depends(get_current_tenant),  # Auto-resolved from API key
    db: AsyncSession = Depends(get_db)
):
```

### Embedding Generation Workflow

1. **Settings Resolution**: API uses tenant settings as defaults
2. **File Selection**: Process specific files or all tenant files
3. **Background Processing**: FastAPI BackgroundTasks for long-running operations
4. **Metrics Tracking**: Real-time progress via polling endpoint
5. **Storage**: Batch INSERT with SQLAlchemy for performance

```python
# Simplified workflow
async def generate_embeddings(files, model, strategy, tenant_name, db):
    session_id = metrics_service.start_session(tenant_name, model, strategy, len(files))
    
    for file in files:
        chunks = chunk_text(file.content, strategy)
        vectors = model.encode(chunks)
        
        # Batch insert embeddings
        embeddings = [
            Embedding(file_id=file.id, chunk_index=i, chunk_text=chunk, 
                     embedding=vector, embedding_model=model)
            for i, (chunk, vector) in enumerate(zip(chunks, vectors))
        ]
        db.add_all(embeddings)
        metrics_service.log_file_processed(file.id, len(chunks), processing_time)
    
    metrics_service.end_session()
```

### Vector Search & RAG

```python
# Similarity search using pgvector
query_vector = model.encode(query)

results = await db.execute(
    select(Embedding)
    .join(File)
    .where(File.tenant_id == tenant.id)
    .where(Embedding.embedding_model == model_name)
    .order_by(Embedding.embedding.cosine_distance(query_vector))
    .limit(k)
)
```

## Performance Considerations

### Indexes
```sql
-- Critical indexes for performance
CREATE INDEX idx_files_tenant_id ON files(tenant_id);
CREATE INDEX idx_files_hash ON files(file_hash);
CREATE INDEX idx_embeddings_model ON embeddings(embedding_model);
CREATE INDEX idx_embeddings_strategy ON embeddings(chunking_strategy);

-- Vector similarity search (automatic with pgvector)
-- Uses HNSW or IVF indexes for approximate nearest neighbor
```

### Query Patterns
- **Tenant Isolation**: All queries filtered by `tenant_id`
- **Model-Specific Search**: Filter by `embedding_model` for consistency
- **Batch Operations**: Use SQLAlchemy bulk operations for embeddings
- **Connection Pooling**: AsyncPG with connection reuse

## Educational Features

### Multi-Model Comparison
The schema supports storing embeddings from different models for the same content:
```sql
-- Same file, different models
SELECT chunk_text, embedding_model, 
       array_length(embedding::float[], 1) as dimensions
FROM embeddings 
WHERE file_id = 1;
```

### Metrics & Analytics
Comprehensive tracking via JSON Lines logging:
```json
{
  "session_id": "uuid",
  "tenant": "ACMECorp", 
  "model": "sentence-transformers/all-MiniLM-L6-v2",
  "strategy": "fixed_size",
  "total_files": 8,
  "total_chunks": 127,
  "total_tokens": 65024,
  "processing_time_sec": 23.4,
  "files": [...]
}
```

### Configuration Experiments
Students can:
1. Change embedding model → See dimension differences
2. Adjust chunk size → Observe chunk count impact  
3. Try different strategies → Compare semantic boundaries
4. Compare search results → Understand model trade-offs

## Data Flows

### File Ingestion
```
demo_data/{tenant}/*.txt → internal_files/{tenant}/ → files table → embedding generation
```

### Embedding Pipeline
```
Text File → Chunking Strategy → Model Encoding → Vector Storage → Search Index
```

### Search Flow
```
Query → Embedding → Vector Search → Ranked Results → RAG Response
```

This architecture provides a solid foundation for learning about vector databases, multi-tenancy, and RAG systems while maintaining educational clarity and production-like patterns.
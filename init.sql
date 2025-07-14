-- Enable pgvector extension
CREATE EXTENSION IF NOT EXISTS vector;

-- Create tables
CREATE TABLE tenants (
    id SERIAL PRIMARY KEY,
    slug VARCHAR(50) UNIQUE NOT NULL,
    name VARCHAR(100) NOT NULL,
    api_key VARCHAR(255) UNIQUE NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE files (
    id SERIAL PRIMARY KEY,
    tenant_id INTEGER REFERENCES tenants(id) ON DELETE CASCADE,
    filename VARCHAR(255) NOT NULL,
    file_path VARCHAR(500) NOT NULL,
    file_hash VARCHAR(64) NOT NULL,
    file_size BIGINT NOT NULL,
    content_type VARCHAR(50) NOT NULL,
    last_modified TIMESTAMP NOT NULL,
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(tenant_id, file_path)
);

CREATE TABLE embeddings (
    id SERIAL PRIMARY KEY,
    file_id INTEGER REFERENCES files(id) ON DELETE CASCADE,
    chunk_index INTEGER NOT NULL,
    chunk_text TEXT NOT NULL,
    embedding_model VARCHAR(100) NOT NULL,
    chunking_strategy VARCHAR(50) NOT NULL,
    embedding VECTOR, -- Dynamic dimensions based on model
    chunk_metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(file_id, chunk_index, embedding_model, chunking_strategy)
);

CREATE TABLE tenant_embedding_settings (
    tenant_id INTEGER PRIMARY KEY REFERENCES tenants(id) ON DELETE CASCADE,
    embedding_model VARCHAR(100) NOT NULL DEFAULT 'sentence-transformers/all-MiniLM-L6-v2',
    chunking_strategy VARCHAR(50) NOT NULL DEFAULT 'fixed_size',
    chunk_size INTEGER DEFAULT 512,
    chunk_overlap INTEGER NOT NULL DEFAULT 50,
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE TABLE rag_sessions (
    id SERIAL PRIMARY KEY,
    tenant_id INTEGER REFERENCES tenants(id) ON DELETE CASCADE,
    embedding_model VARCHAR(100) NOT NULL,
    chunking_strategy VARCHAR(50) NOT NULL,
    rag_technique VARCHAR(50) NOT NULL,
    query TEXT NOT NULL,
    results JSONB NOT NULL,
    answer TEXT,
    confidence VARCHAR,
    answer_model VARCHAR(100),
    generation_time VARCHAR(10),
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX idx_files_tenant_id ON files(tenant_id);
CREATE INDEX idx_files_hash ON files(file_hash);
CREATE INDEX idx_embeddings_file_id ON embeddings(file_id);
CREATE INDEX idx_embeddings_model_strategy ON embeddings(embedding_model, chunking_strategy);
CREATE INDEX idx_rag_sessions_tenant_id ON rag_sessions(tenant_id);
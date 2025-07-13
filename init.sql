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
    embedding VECTOR(384), -- all-MiniLM-L6-v2 dimension
    chunk_metadata JSONB DEFAULT '{}',
    created_at TIMESTAMP DEFAULT NOW(),
    UNIQUE(file_id, chunk_index, embedding_model, chunking_strategy)
);

CREATE TABLE rag_sessions (
    id SERIAL PRIMARY KEY,
    tenant_id INTEGER REFERENCES tenants(id) ON DELETE CASCADE,
    embedding_model VARCHAR(100) NOT NULL,
    chunking_strategy VARCHAR(50) NOT NULL,
    rag_technique VARCHAR(50) NOT NULL,
    query TEXT NOT NULL,
    results JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT NOW()
);

-- Create indexes for performance
CREATE INDEX idx_files_tenant_id ON files(tenant_id);
CREATE INDEX idx_files_hash ON files(file_hash);
CREATE INDEX idx_embeddings_file_id ON embeddings(file_id);
CREATE INDEX idx_embeddings_model_strategy ON embeddings(embedding_model, chunking_strategy);
CREATE INDEX idx_rag_sessions_tenant_id ON rag_sessions(tenant_id);
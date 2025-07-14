-- Migration: Add tenant embedding settings table
-- Run this after stopping the backend container

-- Create tenant_embedding_settings table
CREATE TABLE IF NOT EXISTS tenant_embedding_settings (
    tenant_id INTEGER PRIMARY KEY REFERENCES tenants(id) ON DELETE CASCADE,
    embedding_model VARCHAR(100) NOT NULL DEFAULT 'sentence-transformers/all-MiniLM-L6-v2',
    chunking_strategy VARCHAR(50) NOT NULL DEFAULT 'fixed_size',
    chunk_size INTEGER DEFAULT 512,
    chunk_overlap INTEGER NOT NULL DEFAULT 50,
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Create trigger for updating updated_at
CREATE OR REPLACE FUNCTION update_tenant_embedding_settings_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ language 'plpgsql';

CREATE TRIGGER update_tenant_embedding_settings_updated_at
    BEFORE UPDATE ON tenant_embedding_settings
    FOR EACH ROW
    EXECUTE FUNCTION update_tenant_embedding_settings_updated_at();

-- Insert default settings for existing tenants
INSERT INTO tenant_embedding_settings (tenant_id, embedding_model, chunking_strategy, chunk_size, chunk_overlap)
SELECT 
    id,
    'sentence-transformers/all-MiniLM-L6-v2',
    'fixed_size',
    512,
    50
FROM tenants
ON CONFLICT (tenant_id) DO NOTHING;

-- Enhance embeddings table if needed (chunk_metadata already exists)
-- The chunk_metadata JSONB field will store:
-- {
--   "chunk_summary": "Auto-generated summary of chunk content",
--   "token_count": 245,
--   "sentence_count": 3,
--   "paragraph_index": 2,
--   "keywords": ["AI", "machine learning"],
--   "readability_score": 0.75,
--   "chunk_type": "paragraph|sentence|fixed",
--   "source_section": "Introduction"
-- }
"""
Unified models for LearnRAG API
Clean, minimal models for database and API with no duplication
"""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
from datetime import datetime

# SQLAlchemy imports
from sqlalchemy import Column, Integer, String, Text, DateTime, BigInteger, ForeignKey, UniqueConstraint, Float
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from sqlalchemy.dialects.postgresql import JSONB
from pgvector.sqlalchemy import Vector
from database import Base

# =============================================================================
# DATABASE MODELS (SQLAlchemy)
# =============================================================================

class Tenant(Base):
    __tablename__ = "tenants"
    
    id = Column(Integer, primary_key=True)
    slug = Column(String(50), unique=True, nullable=False)
    name = Column(String(100), nullable=False)
    api_key = Column(String(255), unique=True, nullable=False)
    created_at = Column(DateTime, default=func.now())
    
    files = relationship("File", back_populates="tenant", cascade="all, delete-orphan")
    rag_sessions = relationship("RagSession", back_populates="tenant", cascade="all, delete-orphan")
    embedding_settings = relationship("TenantEmbeddingSettings", back_populates="tenant", uselist=False, cascade="all, delete-orphan")

class File(Base):
    __tablename__ = "files"
    
    id = Column(Integer, primary_key=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False)
    filename = Column(String(255), nullable=False)
    file_path = Column(String(500), nullable=False)
    file_hash = Column(String(64), nullable=False)
    file_size = Column(BigInteger, nullable=False)
    content_type = Column(String(50), nullable=False)
    last_modified = Column(DateTime, nullable=False)
    created_at = Column(DateTime, default=func.now())
    
    tenant = relationship("Tenant", back_populates="files")
    embeddings = relationship("Embedding", back_populates="file", cascade="all, delete-orphan")
    
    __table_args__ = (UniqueConstraint('tenant_id', 'file_path'),)

class Embedding(Base):
    __tablename__ = "embeddings"
    
    id = Column(Integer, primary_key=True)
    file_id = Column(Integer, ForeignKey("files.id", ondelete="CASCADE"), nullable=False)
    chunk_index = Column(Integer, nullable=False)
    chunk_text = Column(Text, nullable=False)
    embedding_model = Column(String(100), nullable=False)
    chunking_strategy = Column(String(50), nullable=False)
    embedding = Column(Vector)
    chunk_metadata = Column(JSONB, default={})
    created_at = Column(DateTime, default=func.now())
    
    file = relationship("File", back_populates="embeddings")
    
    __table_args__ = (UniqueConstraint('file_id', 'chunk_index', 'embedding_model', 'chunking_strategy'),)

class TenantEmbeddingSettings(Base):
    __tablename__ = "tenant_embedding_settings"
    
    tenant_id = Column(Integer, ForeignKey("tenants.id", ondelete="CASCADE"), primary_key=True)
    embedding_model = Column(String(100), nullable=False, default='sentence-transformers/all-MiniLM-L6-v2')
    chunking_strategy = Column(String(50), nullable=False, default='fixed_size')
    chunk_size = Column(Integer, default=512)
    chunk_overlap = Column(Integer, nullable=False, default=50)
    updated_at = Column(DateTime, default=func.now())
    
    tenant = relationship("Tenant", back_populates="embedding_settings")

class RagSession(Base):
    __tablename__ = "rag_sessions"
    
    id = Column(Integer, primary_key=True)
    tenant_id = Column(Integer, ForeignKey("tenants.id", ondelete="CASCADE"), nullable=False)
    embedding_model = Column(String(100), nullable=False)
    chunking_strategy = Column(String(50), nullable=False)
    rag_technique = Column(String(50), nullable=False)
    query = Column(Text, nullable=False)
    results = Column(JSONB, nullable=False)
    answer = Column(Text)
    answer_model = Column(String(100))
    created_at = Column(DateTime, default=func.now())
    
    tenant = relationship("Tenant", back_populates="rag_sessions")

# =============================================================================
# API MODELS (Pydantic) - Clean and minimal
# =============================================================================

# Core search and answer models
class SearchRequest(BaseModel):
    query: str = Field(..., description="Search query")
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", description="Embedding model")
    chunking_strategy: str = Field(default="fixed_size", description="Chunking strategy")
    rag_technique: str = Field(default="similarity_search", description="RAG technique")
    top_k: int = Field(default=5, description="Number of results")

class AnswerRequest(BaseModel):
    query: str = Field(..., description="Question to answer")
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", description="Embedding model")
    chunking_strategy: str = Field(default="fixed_size", description="Chunking strategy")
    answer_model: str = Field(default="google/flan-t5-base", description="LLM model")
    top_k: int = Field(default=10, description="Number of chunks")
    min_similarity: float = Field(default=0.3, description="Minimum similarity")
    max_length: int = Field(default=200, description="Max answer length")

class SearchResult(BaseModel):
    chunk_text: str = Field(..., description="Text content")
    similarity: float = Field(..., description="Similarity score")
    file_name: str = Field(..., description="Source file name")
    file_path: str = Field(..., description="Source file path")
    chunk_index: int = Field(..., description="Chunk index")
    chunk_metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadata")

class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    embedding_model: str
    chunking_strategy: str
    rag_technique: str
    total_results: int

class AnswerResponse(BaseModel):
    query: str
    answer: str
    confidence: float
    sources: List[SearchResult]
    generation_time: float
    model_used: str
    fallback_used: bool = Field(default=False)
    error: Optional[str] = Field(None)

# Embedding models
class GenerateEmbeddingsRequest(BaseModel):
    """
    Request to generate embeddings for RAG system.
    
    Teaching Note: This demonstrates the core parameters needed for RAG:
    - embedding_model: Which neural network model to use for text → vector conversion
    - chunking_strategy: How to split documents (fixed_size, sentence, recursive)
    - chunk_size: How many tokens per chunk (balance between context vs precision)
    - chunk_overlap: Overlap between chunks to preserve context across boundaries
    """
    embedding_model: Optional[str] = Field(None, description="Embedding model")
    chunking_strategy: Optional[str] = Field(None, description="Chunking strategy")
    chunk_size: Optional[int] = Field(None, description="Chunk size")
    chunk_overlap: Optional[int] = Field(None, description="Chunk overlap")
    force_regenerate: bool = Field(default=False, description="Force regeneration")

class GenerateEmbeddingsResponse(BaseModel):
    message: str
    task_id: Optional[str] = Field(None)
    files_processed: int
    total_chunks: int

class EmbeddingStatus(BaseModel):
    file_id: int
    filename: str
    has_embeddings: bool
    chunk_count: int
    embedding_model: Optional[str] = Field(None)
    chunking_strategy: Optional[str] = Field(None)
    last_updated: Optional[datetime] = Field(None)

class GeneralEmbeddingStatus(BaseModel):
    total_files: int
    files_with_embeddings: int
    files_without_embeddings: int
    total_chunks: int
    available_models: List[str] = Field(default_factory=list)
    available_strategies: List[str] = Field(default_factory=list)
    last_updated: Optional[datetime] = Field(None)

# Tenant models 
class TenantInfo(BaseModel):
    id: int
    name: str
    slug: str
    created_at: datetime
    file_count: int

class FileInfo(BaseModel):
    id: int
    name: str
    path: str
    size: int
    type: str
    created_at: datetime
    updated_at: datetime

class EmbeddingSettings(BaseModel):
    """Unified settings model for both request and response"""
    embedding_model: str
    chunking_strategy: str
    chunk_size: int = Field(default=512)
    chunk_overlap: int = Field(default=50)
    updated_at: Optional[datetime] = Field(None)

class CompareRequest(BaseModel):
    queries: List[str] = Field(..., description="Queries to test")
    techniques: List[str] = Field(..., description="Techniques to compare")

# Request/Response models for embedding settings
class EmbeddingSettingsRequest(BaseModel):
    """Request model for updating tenant embedding settings"""
    embedding_model: str = Field(..., description="Embedding model name")
    chunking_strategy: str = Field(..., description="Text chunking strategy") 
    chunk_size: int = Field(512, description="Size of text chunks")
    chunk_overlap: int = Field(50, description="Overlap between chunks")

class EmbeddingSettingsResponse(BaseModel):
    """Response model for tenant embedding settings"""
    embedding_model: str
    chunking_strategy: str
    chunk_size: int
    chunk_overlap: int
    updated_at: Optional[datetime] = None
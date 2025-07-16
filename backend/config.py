"""
Simple Configuration for LearnRAG Backend
Teaching Purpose: Shows how to use environment variables for configuration
- Standard practice for containerized applications
- No complex loaders, just simple os.getenv calls
- Easy to override for different environments
"""
import os
from pathlib import Path

# Database Configuration
DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://postgres:postgres@postgres:5432/learnrag")
DATABASE_POOL_SIZE = int(os.getenv("DATABASE_POOL_SIZE", "10"))
DATABASE_MAX_OVERFLOW = int(os.getenv("DATABASE_MAX_OVERFLOW", "20"))
DATABASE_POOL_RECYCLE = int(os.getenv("DATABASE_POOL_RECYCLE", "3600"))

# File Paths
INTERNAL_FILES_DIR = Path(os.getenv("INTERNAL_FILES_DIR", "/app/internal_files"))
DEMO_DATA_DIR = Path(os.getenv("DEMO_DATA_DIR", "/app/setup"))
MODELS_CACHE_DIR = Path(os.getenv("MODELS_CACHE_DIR", "/app/models_cache"))
OUTPUT_DIR = Path(os.getenv("OUTPUT_DIR", "/app/output"))

# Model Configuration
DEFAULT_EMBEDDING_MODEL = os.getenv("DEFAULT_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "allenai/unifiedqa-t5-base")

# Processing Defaults
DEFAULT_CHUNK_SIZE = int(os.getenv("DEFAULT_CHUNK_SIZE", "512"))
DEFAULT_CHUNK_OVERLAP = int(os.getenv("DEFAULT_CHUNK_OVERLAP", "50"))
DEFAULT_CHUNKING_STRATEGY = os.getenv("DEFAULT_CHUNKING_STRATEGY", "fixed_size")
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "5"))
DEFAULT_MAX_ANSWER_LENGTH = int(os.getenv("DEFAULT_MAX_ANSWER_LENGTH", "200"))
DEFAULT_TEMPERATURE = float(os.getenv("DEFAULT_TEMPERATURE", "0.7"))

# Processing Performance
EMBEDDING_BATCH_SIZE = int(os.getenv("EMBEDDING_BATCH_SIZE", "32"))
EMBEDDING_MAX_WORKERS = int(os.getenv("EMBEDDING_MAX_WORKERS", "4"))
EMBEDDING_TIMEOUT = int(os.getenv("EMBEDDING_TIMEOUT", "300"))

# Security
CORS_ORIGINS = os.getenv("CORS_ORIGINS", "http://localhost:3000").split(",")
API_KEY_LENGTH = int(os.getenv("API_KEY_LENGTH", "32"))

# Metrics
METRICS_FILE = Path(os.getenv("METRICS_FILE", "/app/metrics/embedding_metrics.jsonl"))

# Available Models (can be overridden with JSON env var)
AVAILABLE_EMBEDDING_MODELS = [
    {
        "name": "sentence-transformers/all-MiniLM-L6-v2",
        "dimension": 384,
        "description": "Lightweight, fast model (384 dimensions)",
        "default": True
    },
    {
        "name": "sentence-transformers/all-mpnet-base-v2", 
        "dimension": 768,
        "description": "Higher quality, slower (768 dimensions)",
        "default": False
    },
    {
        "name": "BAAI/bge-small-en-v1.5",
        "dimension": 384,
        "description": "Recent, good performance (384 dimensions)",
        "default": False
    },
    {
        "name": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
        "dimension": 384,
        "description": "Optimized for Q&A (384 dimensions)",
        "default": False
    },
    {
        "name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        "dimension": 384,
        "description": "Multilingual support (384 dimensions)",
        "default": False
    }
]

AVAILABLE_LLM_MODELS = [
    {
        "name": "allenai/unifiedqa-t5-base",
        "description": "QA-optimized T5 (250M) - Best for literature Q&A",
        "default_temperature": 0.1,
        "default_top_p": 0.8,
        "default_repetition_penalty": 1.0,
        "recommended": True
    },
    {
        "name": "google/flan-t5-base",
        "description": "Instruction-tuned T5 (250M) - Best balance",
        "default_temperature": 0.2,
        "default_top_p": 0.85,
        "default_repetition_penalty": 1.1,
        "recommended": True
    },
    {
        "name": "facebook/bart-large-cnn",
        "description": "BART CNN-trained (400M) - Different architecture",
        "default_temperature": 0.3,
        "default_top_p": 0.9,
        "default_repetition_penalty": 1.0,
        "recommended": False
    },
    {
        "name": "google/flan-t5-large",
        "description": "High-quality T5 (780M) - Best comprehension",
        "default_temperature": 0.2,
        "default_top_p": 0.85,
        "default_repetition_penalty": 1.1,
        "recommended": False
    },
    {
        "name": "google/flan-t5-small",
        "description": "Fast T5 (80M) - Quick literature demos",
        "default_temperature": 0.2,
        "default_top_p": 0.85,
        "default_repetition_penalty": 1.1,
        "recommended": False
    }
]

# Generation defaults
DEFAULT_CONTEXT_CHUNKS = int(os.getenv("DEFAULT_CONTEXT_CHUNKS", "5"))
DEFAULT_REPETITION_PENALTY = float(os.getenv("DEFAULT_REPETITION_PENALTY", "1.1"))
DEFAULT_TOP_P = float(os.getenv("DEFAULT_TOP_P", "0.9"))
DEFAULT_MIN_SIMILARITY = float(os.getenv("DEFAULT_MIN_SIMILARITY", "0.3"))

# Chunking strategies
CHUNKING_STRATEGIES = [
    {
        "name": "fixed_size",
        "description": "Fixed-size chunks with word-based splitting and overlap",
        "parameters": ["chunk_size", "chunk_overlap"],
        "default": True
    },
    {
        "name": "sentence",
        "description": "Sentence-based chunks with natural boundaries",
        "parameters": ["max_sentences"],
        "default": False
    },
    {
        "name": "recursive",
        "description": "Recursive splitting with multiple separators",
        "parameters": ["chunk_size", "chunk_overlap"],
        "default": False
    }
]

# RAG techniques
RAG_TECHNIQUES = ["similarity_search", "mmr", "semantic_search"]
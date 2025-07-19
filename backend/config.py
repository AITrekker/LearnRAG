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
DEFAULT_LLM_MODEL = os.getenv("DEFAULT_LLM_MODEL", "google/flan-t5-base")

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
        "description": "Fastest, lightweight (384d) - Speed champion",
        "default": True
    },
    {
        "name": "BAAI/bge-small-en-v1.5",
        "dimension": 384,
        "description": "Modern, balanced (384d) - SOTA 2024",
        "default": False
    },
    {
        "name": "sentence-transformers/all-mpnet-base-v2", 
        "dimension": 768,
        "description": "High quality, proven (768d) - Classic performer",
        "default": False
    },
    {
        "name": "intfloat/e5-large-v2",
        "dimension": 1024,
        "description": "Document specialist (1024d) - Long text expert",
        "default": False
    },
    {
        "name": "BAAI/bge-large-en-v1.5",
        "dimension": 1024,
        "description": "Best quality, tables (1024d) - PDF/table expert",
        "default": False
    }
]

AVAILABLE_LLM_MODELS = [
    {
        "name": "deepset/roberta-base-squad2",
        "description": "Question answering expert (125M) - SQuAD 2.0 trained ✅ WORKING",
        "default_temperature": 0.8,
        "default_top_p": 0.9,
        "default_repetition_penalty": 1.0,
        "recommended": True
    },
    {
        "name": "distilbert/distilbert-base-cased-distilled-squad",
        "description": "DistilBERT QA (67M) - Fast, 87.1 F1 score on SQuAD v1.1",
        "default_temperature": 0.8,
        "default_top_p": 0.9,
        "default_repetition_penalty": 1.0,
        "recommended": True
    },
    {
        "name": "distilbert/distilbert-base-uncased-distilled-squad",
        "description": "DistilBERT QA uncased (67M) - 86.9 F1 score, fastest",
        "default_temperature": 0.8,
        "default_top_p": 0.9,
        "default_repetition_penalty": 1.0,
        "recommended": True
    },
    {
        "name": "bert-large-uncased-whole-word-masking-finetuned-squad",
        "description": "BERT Large QA (340M) - 88.5 F1 score, high quality",
        "default_temperature": 0.8,
        "default_top_p": 0.9,
        "default_repetition_penalty": 1.0,
        "recommended": True
    },
    {
        "name": "csarron/bert-base-uncased-squad-v1",
        "description": "BERT Base QA (110M) - Well-tested community model",
        "default_temperature": 0.8,
        "default_top_p": 0.9,
        "default_repetition_penalty": 1.0,
        "recommended": True
    }
]

# Generation defaults
DEFAULT_CONTEXT_CHUNKS = int(os.getenv("DEFAULT_CONTEXT_CHUNKS", "5"))
DEFAULT_REPETITION_PENALTY = float(os.getenv("DEFAULT_REPETITION_PENALTY", "1.1"))
DEFAULT_TOP_P = float(os.getenv("DEFAULT_TOP_P", "0.9"))
DEFAULT_MIN_SIMILARITY = float(os.getenv("DEFAULT_MIN_SIMILARITY", "0.3"))
DEFAULT_PROMPT_TEMPLATE = os.getenv("DEFAULT_PROMPT_TEMPLATE", "factual")

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
RAG_TECHNIQUES = ["similarity_search", "hybrid_search", "hierarchical_search"]

# Prompt templates for LLM answer generation
PROMPT_TEMPLATES = {
    "quote_based": {
        "name": "Quote-Based",
        "description": "Forces model to quote directly from the provided text",
        "template": """You must answer the question using ONLY information from the context below. Quote directly from the text to support your answer.

Context:
{context}

Question: {query}

Instructions: 
1. Find the relevant information in the context above
2. Quote the exact text that answers the question
3. Provide a brief explanation based only on what you quoted

Answer:"""
    },
    "reading_comprehension": {
        "name": "Reading Comprehension",
        "description": "Optimized for extracting information from provided text",
        "template": """Read the following text carefully and answer the question based on what you read.

Text:
{context}

Question: {query}

Based on the text above, the answer is:"""
    },
    "factual": {
        "name": "Factual",
        "description": "Precise, context-only answers with no speculation",
        "template": """Answer the following question based only on the provided context. Give a complete, informative answer in 1-3 sentences. If the context doesn't contain enough information to answer the question, say "I don't have enough information to answer this question."

Context:
{context}

Question: {query}

Answer:"""
    },
    "conversational": {
        "name": "Conversational", 
        "description": "Friendly, helpful explanations with natural tone",
        "template": """You are a helpful assistant. Answer the question using the provided context in a friendly, conversational way. Explain your reasoning and provide helpful details when possible.

Context:
{context}

Question: {query}

Answer:"""
    },
    "detailed": {
        "name": "Detailed",
        "description": "Comprehensive answers with reasoning and examples", 
        "template": """Provide a comprehensive answer to the question using the provided context. Include relevant details, explain the reasoning behind your answer, and provide examples when available. Be thorough but stay grounded in the provided information.

Context:
{context}

Question: {query}

Detailed Answer:"""
    },
    "concise": {
        "name": "Concise",
        "description": "Brief, direct answers with minimal elaboration",
        "template": """Answer the question briefly and directly using only the provided context. Use the fewest words possible while being accurate.

Context:
{context}

Question: {query}

Brief Answer:"""
    }
}
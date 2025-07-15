"""
Embedding Service - Core RAG Text-to-Vector Conversion System

Teaching Purpose: This service demonstrates the fundamental concepts of RAG (Retrieval-Augmented Generation):

1. TEXT → VECTORS: How neural networks convert human language into numerical representations
2. SEMANTIC SIMILARITY: Why vectors enable "meaning-based" search instead of keyword matching  
3. CHUNKING STRATEGIES: How to split documents for optimal retrieval vs context balance
4. MODEL CACHING: Production patterns for managing AI models in memory and disk
5. DELTA SYNC: Avoiding redundant work when files/settings haven't changed

Key RAG Concepts Illustrated:
- Embedding models transform text into high-dimensional vectors (384d-768d)
- Similar meanings cluster together in vector space (cosine similarity)
- Different chunking affects what information gets retrieved
- Model choice affects speed vs quality trade-offs in production
"""

import os
import asyncio
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import pickle

import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete

from models import File, Embedding
from services.file_processor import FileProcessor
from services.metrics_service import MetricsService
from utils import PathConfig, DatabaseUtils


class EmbeddingService:
    """
    Core service for converting text documents into vector embeddings for RAG systems.
    
    Teaching Concepts:
    - Model lifecycle management (download → cache → load → inference)
    - Memory vs disk caching strategies for production systems
    - Batch processing for efficient GPU/CPU utilization
    - Delta sync to avoid redundant processing
    """
    
    def __init__(self):
        # Production Pattern: Persistent model cache to survive container restarts
        self.models_cache_dir = PathConfig.MODELS_CACHE
        self.models_cache_dir.mkdir(exist_ok=True)
        
        self.file_processor = FileProcessor()
        
        # Performance Pattern: In-memory cache for active models
        # Avoids reloading models from disk on every request
        self._loaded_models = {}  # model_name → SentenceTransformer instance
        
        self.metrics_service = MetricsService()

    async def generate_embeddings_for_files(
        self, 
        files: List[File], 
        embedding_model: str, 
        chunking_strategy: str, 
        db: AsyncSession,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        tenant_name: str = "Unknown"
    ):
        """
        Generate vector embeddings for multiple files - Core RAG Pipeline Entry Point
        
        Teaching Flow:
        1. Load neural network model (download if needed, cache for reuse)
        2. For each file: extract text → chunk → embed → store vectors
        3. Track metrics for learning about processing efficiency
        
        RAG Concepts:
        - batch_size: Process multiple chunks together for GPU efficiency
        - delta_sync: Skip files that haven't changed since last embedding
        - model_caching: Avoid reloading multi-GB models on every request
        - chunking_strategy: Different ways to split text affect retrieval quality
        
        Why This Matters:
        - Text becomes searchable by meaning, not just keywords
        - Chunks balance context (larger = more context) vs precision (smaller = more specific)
        - Model choice affects speed vs quality in production systems
        """
        # Start metrics session
        session_id = self.metrics_service.start_session(
            tenant_name, embedding_model, chunking_strategy, 
            chunk_size if chunking_strategy == "fixed_size" else None, 
            chunk_overlap, len(files)
        )
        
        model = await self._get_model(embedding_model)
        
        try:
            for file in files:
                await self._generate_embeddings_for_file(
                    file, model, embedding_model, chunking_strategy, chunk_size, chunk_overlap, db
                )
        finally:
            # End metrics session
            self.metrics_service.end_session()

    async def _generate_embeddings_for_file(
        self,
        file: File,
        model: SentenceTransformer,
        embedding_model: str,
        chunking_strategy: str,
        chunk_size: int,
        chunk_overlap: int,
        db: AsyncSession
    ):
        """Generate embeddings for a single file with delta sync"""
        # Check if embeddings already exist with same model and strategy
        existing_embeddings = await db.execute(
            select(Embedding).where(
                Embedding.file_id == file.id,
                Embedding.embedding_model == embedding_model,
                Embedding.chunking_strategy == chunking_strategy
            )
        )
        existing = existing_embeddings.scalars().all()

        # If embeddings exist, check if file has changed
        if existing:
            # For delta sync, we compare file hash with when embeddings were created
            # If file hasn't changed and model/strategy are same, skip
            print(f"Embeddings already exist for file {file.filename} with model {embedding_model}")
            return

        # Delete existing embeddings for this file+model+strategy if any
        if existing:
            await db.execute(
                delete(Embedding).where(
                    Embedding.file_id == file.id,
                    Embedding.embedding_model == embedding_model,
                    Embedding.chunking_strategy == chunking_strategy
                )
            )

        # Track file processing start time
        file_start_time = time.time()
        
        # Extract text from file
        # Get tenant info
        from models import Tenant
        tenant_result = await db.execute(select(Tenant).where(Tenant.id == file.tenant_id))
        tenant = tenant_result.scalar_one()
        
        file_path = Path("/app/internal_files") / tenant.slug / file.file_path
        text_content = await self.file_processor.extract_text(file_path, file.content_type)

        if not text_content.strip():
            print(f"No text content extracted from {file.filename}")
            return

        # Chunk the text
        chunks = await self._chunk_text(text_content, chunking_strategy, chunk_size, chunk_overlap)
        
        # Calculate file metrics
        file_size = file_path.stat().st_size if file_path.exists() else 0
        tokens_processed = sum(len(chunk.split()) for chunk in chunks)
        chunk_distribution = [len(chunk.split()) for chunk in chunks]

        # Generate embeddings in batches
        batch_size = 32
        for i, chunk in enumerate(chunks):
            try:
                # Generate embedding
                embedding_vector = model.encode([chunk], convert_to_tensor=False)[0]
                
                # Create embedding record
                embedding = Embedding(
                    file_id=file.id,
                    chunk_index=i,
                    chunk_text=chunk,
                    embedding_model=embedding_model,
                    chunking_strategy=chunking_strategy,
                    embedding=embedding_vector.tolist(),
                    chunk_metadata={
                        "chunk_length": len(chunk),
                        "chunk_words": len(chunk.split())
                    }
                )
                db.add(embedding)

                # Commit in batches
                if (i + 1) % batch_size == 0:
                    await db.commit()

            except Exception as e:
                print(f"Error generating embedding for chunk {i} of file {file.filename}: {e}")
                continue

        # Final commit
        await db.commit()
        
        # Log file metrics
        file_processing_time = time.time() - file_start_time
        self.metrics_service.log_file_processed(
            file.id, file.filename, file_size,
            len(chunks), tokens_processed, file_processing_time, chunk_distribution
        )
        
        print(f"Generated {len(chunks)} embeddings for {file.filename}")

    async def _get_model(self, model_name: str) -> SentenceTransformer:
        """Get model with caching"""
        if model_name in self._loaded_models:
            return self._loaded_models[model_name]

        # Check if model is cached on disk
        model_cache_path = self.models_cache_dir / model_name.replace("/", "_")
        
        if model_cache_path.exists():
            print(f"Loading cached model from {model_cache_path}")
            model = SentenceTransformer(str(model_cache_path))
        else:
            print(f"Downloading and caching model {model_name}")
            model = SentenceTransformer(model_name)
            # Save to cache
            model.save(str(model_cache_path))
            print(f"Model cached to {model_cache_path}")

        # Keep in memory
        self._loaded_models[model_name] = model
        return model

    async def _chunk_text(self, text: str, strategy: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
        """
        Split text into chunks for embedding generation - Critical RAG Design Decision
        
        Teaching Concepts:
        WHY CHUNKING MATTERS:
        - Large documents exceed model context windows (usually 512-2048 tokens)
        - Smaller chunks = more precise retrieval but less context
        - Larger chunks = more context but less precise matching
        - Overlap preserves context across chunk boundaries
        
        STRATEGY TRADE-OFFS:
        - fixed_size: Predictable performance, may break sentences
        - sentence: Natural boundaries, variable chunk sizes  
        - recursive: Smart fallbacks, handles diverse document structures
        
        PRODUCTION CONSIDERATIONS:
        - chunk_size affects retrieval precision vs context richness
        - overlap prevents important information from being split
        - strategy choice depends on document type and use case
        """
        if strategy == "fixed_size":
            return await self._fixed_size_chunking(text, chunk_size, overlap)
        elif strategy == "sentence":
            return await self._sentence_based_chunking(text)
        elif strategy == "recursive":
            return await self._recursive_chunking(text, chunk_size, overlap)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")

    async def _fixed_size_chunking(
        self, 
        text: str, 
        chunk_size: int = 512, 
        overlap: int = 50
    ) -> List[str]:
        """
        Fixed-size chunking with overlap - Simple and Predictable Strategy
        
        Teaching Concepts:
        WHEN TO USE: Documents with consistent structure, performance-critical applications
        
        ALGORITHM:
        1. Split text into words (simple tokenization)
        2. Create chunks of exactly `chunk_size` words
        3. Overlap chunks by `overlap` words to preserve context
        
        EXAMPLE with chunk_size=4, overlap=2:
        Text: "The quick brown fox jumps over the lazy dog"
        Chunk 1: "The quick brown fox"        (words 0-3)
        Chunk 2: "brown fox jumps over"       (words 2-5, overlaps with chunk 1)
        Chunk 3: "jumps over the lazy"       (words 4-7, overlaps with chunk 2)
        Chunk 4: "the lazy dog"              (words 6-8, overlaps with chunk 3)
        
        TRADE-OFFS:
        ✅ Predictable chunk sizes (good for performance tuning)
        ✅ Simple implementation and debugging
        ❌ May break sentences/paragraphs mid-thought
        ❌ Less natural for human-readable content
        """
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = " ".join(chunk_words)
            if chunk_text.strip():
                chunks.append(chunk_text)
        
        return chunks

    async def _sentence_based_chunking(self, text: str, max_sentences: int = 3) -> List[str]:
        """
        Sentence-based chunking - Natural Language Boundaries Strategy
        
        Teaching Concepts:
        WHEN TO USE: Human-readable documents, narrative content, articles
        
        ALGORITHM:
        1. Split text at sentence boundaries (., !, ?)
        2. Group `max_sentences` into each chunk
        3. Preserve complete thoughts and natural flow
        
        EXAMPLE with max_sentences=2:
        Text: "AI is transforming society. It enables new capabilities. However, it raises ethical concerns. We must proceed carefully."
        Chunk 1: "AI is transforming society. It enables new capabilities."
        Chunk 2: "However, it raises ethical concerns. We must proceed carefully."
        
        TRADE-OFFS:
        ✅ Preserves complete thoughts and natural language flow
        ✅ Better for human-readable content
        ✅ Respects grammatical boundaries
        ❌ Variable chunk sizes (harder to predict performance)
        ❌ Some chunks may be very short or long
        ❌ Sentence detection isn't perfect (abbreviations, etc.)
        """
        import re
        
        # Split into sentences using regex
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = []
        
        for sentence in sentences:
            current_chunk.append(sentence)
            
            if len(current_chunk) >= max_sentences:
                chunk_text = '. '.join(current_chunk) + '.'
                chunks.append(chunk_text)
                current_chunk = []
        
        # Add remaining sentences
        if current_chunk:
            chunk_text = '. '.join(current_chunk) + '.'
            chunks.append(chunk_text)
        
        return chunks

    async def _recursive_chunking(self, text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
        """Recursive chunking: paragraphs -> sentences -> fixed size"""
        # First try paragraph splitting
        paragraphs = text.split('\n\n')
        chunks = []
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            # If paragraph is small enough, use as chunk
            words = paragraph.split()
            if len(words) <= chunk_size:
                chunks.append(paragraph)
            else:
                # Fall back to sentence chunking
                sentence_chunks = await self._sentence_based_chunking(paragraph)
                for chunk in sentence_chunks:
                    chunk_words = chunk.split()
                    if len(chunk_words) <= chunk_size:
                        chunks.append(chunk)
                    else:
                        # Final fallback to fixed size
                        fixed_chunks = await self._fixed_size_chunking(chunk, chunk_size, overlap)
                        chunks.extend(fixed_chunks)
        
        return chunks

    async def generate_query_embedding(self, query: str, model_name: str) -> List[float]:
        """Generate embedding for a query"""
        model = await self._get_model(model_name)
        embedding = model.encode([query], convert_to_tensor=False)[0]
        return embedding.tolist()

    def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Available embedding models - The Neural Networks That Power RAG
        
        Teaching Concepts:
        WHAT ARE EMBEDDING MODELS?
        - Neural networks trained to convert text → high-dimensional vectors
        - Similar meanings cluster together in vector space (semantic similarity)
        - Different models optimized for different tasks and trade-offs
        
        KEY TRADE-OFFS:
        - Dimensions: Higher = more nuanced but slower/more memory
        - Speed: Smaller models faster for real-time applications  
        - Quality: Larger models often better at capturing nuance
        - Specialization: Some optimized for Q&A, others for general text
        
        PRODUCTION CONSIDERATIONS:
        - Model size affects memory usage and inference speed
        - Vector dimensions must match throughout the system
        - Language support varies (multilingual vs English-only)
        """
        return [
            {
                "name": "sentence-transformers/all-MiniLM-L6-v2",
                "description": "🚀 Default: Lightweight, fast model (384 dimensions)",
                "dimension": 384,
                "default": True,
                "use_case": "General purpose, good speed/quality balance"
            },
            {
                "name": "sentence-transformers/all-mpnet-base-v2", 
                "description": "🎯 Premium: Higher quality, slower (768 dimensions)",
                "dimension": 768,
                "default": False,
                "use_case": "Best quality when speed is less critical"
            },
            {
                "name": "BAAI/bge-small-en-v1.5",
                "description": "⭐ Modern: Recent, good performance (384 dimensions)",
                "dimension": 384,
                "default": False,
                "use_case": "State-of-the-art efficiency and quality"
            },
            {
                "name": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
                "description": "❓ Q&A Specialist: Optimized for questions (384 dimensions)",
                "dimension": 384,
                "default": False,
                "use_case": "Question-answering and FAQ systems"
            },
            {
                "name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                "description": "🌍 Global: Multilingual support (384 dimensions)",
                "dimension": 384,
                "default": False,
                "use_case": "Non-English content and cross-language search"
            }
        ]
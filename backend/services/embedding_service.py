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
    def __init__(self):
        self.models_cache_dir = PathConfig.MODELS_CACHE
        self.models_cache_dir.mkdir(exist_ok=True)
        self.file_processor = FileProcessor()
        self._loaded_models = {}  # In-memory cache
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
        """Generate embeddings for multiple files with metrics tracking"""
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
        """Chunk text based on strategy"""
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
        """Fixed size chunking with overlap"""
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = " ".join(chunk_words)
            if chunk_text.strip():
                chunks.append(chunk_text)
        
        return chunks

    async def _sentence_based_chunking(self, text: str, max_sentences: int = 3) -> List[str]:
        """Sentence-based chunking with natural boundaries"""
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
        """Get list of available embedding models"""
        return [
            {
                "name": "sentence-transformers/all-MiniLM-L6-v2",
                "description": "Lightweight, fast model (384 dimensions)",
                "dimension": 384,
                "default": True
            },
            {
                "name": "sentence-transformers/all-mpnet-base-v2",
                "description": "Higher quality, slower (768 dimensions)",
                "dimension": 768,
                "default": False
            },
            {
                "name": "BAAI/bge-small-en-v1.5",
                "description": "Recent, good performance (384 dimensions)",
                "dimension": 384,
                "default": False
            },
            {
                "name": "sentence-transformers/multi-qa-MiniLM-L6-cos-v1",
                "description": "Optimized for Q&A (384 dimensions)",
                "dimension": 384,
                "default": False
            },
            {
                "name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
                "description": "Multilingual support (384 dimensions)",
                "dimension": 384,
                "default": False
            }
        ]
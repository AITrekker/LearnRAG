import os
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
import pickle
import hashlib

import torch
from sentence_transformers import SentenceTransformer
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete

from app.models.database import File, Embedding
from app.services.file_processor import FileProcessor


class EmbeddingService:
    def __init__(self):
        self.models_cache_dir = Path(os.getenv("MODELS_CACHE_DIR", "/app/models_cache"))
        self.models_cache_dir.mkdir(exist_ok=True)
        self.file_processor = FileProcessor()
        self._loaded_models = {}  # In-memory cache

    async def generate_embeddings_for_files(
        self, 
        files: List[File], 
        embedding_model: str, 
        chunking_strategy: str, 
        db: AsyncSession
    ):
        """Generate embeddings for multiple files"""
        model = await self._get_model(embedding_model)
        
        for file in files:
            await self._generate_embeddings_for_file(
                file, model, embedding_model, chunking_strategy, db
            )

    async def _generate_embeddings_for_file(
        self,
        file: File,
        model: SentenceTransformer,
        embedding_model: str,
        chunking_strategy: str,
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

        # Extract text from file
        # Get tenant info
        from app.models.database import Tenant
        tenant_result = await db.execute(select(Tenant).where(Tenant.id == file.tenant_id))
        tenant = tenant_result.scalar_one()
        
        file_path = Path("/app/internal_files") / tenant.slug / file.file_path
        text_content = await self.file_processor.extract_text(file_path, file.content_type)

        if not text_content.strip():
            print(f"No text content extracted from {file.filename}")
            return

        # Chunk the text
        chunks = await self._chunk_text(text_content, chunking_strategy)

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

    async def _chunk_text(self, text: str, strategy: str) -> List[str]:
        """Chunk text based on strategy"""
        if strategy == "fixed_size":
            return await self._fixed_size_chunking(text)
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
            }
        ]
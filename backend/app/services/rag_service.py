from typing import List, Optional
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text

from app.models.database import Embedding, File
from app.models.responses import SearchResult
from app.services.embedding_service import EmbeddingService


class RagService:
    def __init__(self):
        self.embedding_service = EmbeddingService()

    async def generate_query_embedding(self, query: str, model_name: str) -> List[float]:
        """Generate embedding for search query"""
        return await self.embedding_service.generate_query_embedding(query, model_name)

    async def similarity_search(
        self,
        query_embedding: List[float],
        tenant_id: int,
        embedding_model: str,
        chunking_strategy: str,
        top_k: int,
        db: AsyncSession
    ) -> List[SearchResult]:
        """Perform cosine similarity search using pgvector"""
        
        # Convert query embedding to pgvector format
        query_vector = str(query_embedding)
        
        # SQL query using pgvector cosine distance
        sql_query = text("""
            SELECT 
                e.chunk_text,
                e.chunk_index,
                e.chunk_metadata,
                f.id as file_id,
                f.filename,
                f.file_path,
                1 - (e.embedding <=> :query_vector) as similarity_score
            FROM embeddings e
            JOIN files f ON e.file_id = f.id
            WHERE f.tenant_id = :tenant_id
            AND e.embedding_model = :embedding_model
            AND e.chunking_strategy = :chunking_strategy
            ORDER BY e.embedding <=> :query_vector
            LIMIT :top_k
        """)

        result = await db.execute(
            sql_query,
            {
                "query_vector": query_vector,
                "tenant_id": tenant_id,
                "embedding_model": embedding_model,
                "chunking_strategy": chunking_strategy,
                "top_k": top_k
            }
        )

        search_results = []
        for row in result:
            search_result = SearchResult(
                chunk_text=row.chunk_text,
                similarity=float(row.similarity_score),
                file_name=row.filename,
                file_path=row.file_path,
                chunk_index=row.chunk_index,
                chunk_metadata=row.chunk_metadata or {}
            )
            search_results.append(search_result)

        return search_results

    async def hybrid_search(
        self,
        query: str,
        query_embedding: List[float],
        tenant_id: int,
        embedding_model: str,
        chunking_strategy: str,
        top_k: int,
        db: AsyncSession,
        semantic_weight: float = 0.7
    ) -> List[SearchResult]:
        """Hybrid search combining semantic and keyword search (Future implementation)"""
        # For Phase 1, just return similarity search
        return await self.similarity_search(
            query_embedding, tenant_id, embedding_model, chunking_strategy, top_k, db
        )

    def get_available_techniques(self) -> List[dict]:
        """Get list of available RAG techniques"""
        return [
            {
                "name": "similarity_search",
                "description": "Basic cosine similarity search using pgvector",
                "default": True
            }
            # Future techniques will be added here
        ]
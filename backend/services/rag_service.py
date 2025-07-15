"""
RAG Service - The Heart of Retrieval-Augmented Generation

Teaching Purpose: This service demonstrates the core RAG retrieval pipeline:

1. QUERY PROCESSING: Convert user questions into searchable vectors
2. SIMILARITY SEARCH: Find relevant chunks using mathematical similarity
3. CONTEXT RETRIEVAL: Return the most relevant text chunks for LLM processing

Core RAG Concepts Illustrated:
- Vector similarity search using cosine distance in high-dimensional space
- pgvector database optimization for semantic search at scale
- Trade-offs between search precision (top-k) and context richness
- Future extensibility for hybrid and advanced RAG techniques

Mathematical Foundation:
- Cosine similarity: measures angle between vectors (0-1, higher = more similar)
- L2 distance: geometric distance in vector space
- Vector space: high-dimensional representation where similar meanings cluster
"""

from typing import List, Optional
import numpy as np
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text

from models import Embedding, File
from models import SearchResult
from services.embedding_service import EmbeddingService


class RagService:
    """
    Core RAG retrieval service implementing semantic search over vector databases.
    
    Teaching Concepts:
    - Query-to-vector transformation for semantic search
    - Vector similarity mathematics (cosine distance)
    - Database optimization for high-dimensional vector queries
    - Scalable retrieval patterns for production RAG systems
    """
    
    def __init__(self):
        self.embedding_service = EmbeddingService()

    async def generate_query_embedding(self, query: str, model_name: str) -> List[float]:
        """
        Convert user query into searchable vector - RAG Step 1
        
        Teaching Concepts:
        WHY EMBED QUERIES?
        - User questions must be in same vector space as document chunks
        - Same model ensures consistent similarity measurements
        - Query vectors enable "meaning-based" rather than "keyword-based" search
        
        EXAMPLE:
        Query: "How do I reset my password?"
        Vector: [0.1, -0.3, 0.8, ...] (384 or 768 dimensions)
        
        This vector will be closest to chunks containing password reset info,
        even if they don't use the exact words "reset" or "password"
        """
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
        """
        Core semantic search using vector similarity - RAG Step 2
        
        Teaching Concepts:
        VECTOR SIMILARITY MATHEMATICS:
        - Cosine distance: 1 - cosine_similarity (0 = identical, 1 = orthogonal)
        - pgvector <=> operator: optimized cosine distance for PostgreSQL
        - Lower distance = higher similarity = more relevant
        
        SQL EXPLANATION:
        1. JOIN embeddings with files for metadata
        2. Filter by tenant/model/strategy (data isolation)
        3. ORDER BY cosine distance (closest vectors first)
        4. LIMIT to top_k results (balance relevance vs context)
        
        PRODUCTION OPTIMIZATIONS:
        - pgvector indexes for sub-second search over millions of vectors
        - Batch processing for multiple queries
        - Caching for frequently accessed content
        
        SEARCH QUALITY FACTORS:
        - top_k: More results = more context but potentially less relevant
        - Model consistency: Query and chunks must use same embedding model
        - Chunking strategy: Affects granularity of retrieved information
        """
        
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
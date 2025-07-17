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
    
    Key concepts demonstrated:
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
        
        WHY VECTOR SIMILARITY?
        - Uses cosine distance to find similar vectors (lower = more similar)
        - pgvector <=> operator provides optimized PostgreSQL vector search
        - Results filtered by tenant/model/strategy for data isolation
        - top_k balances relevance vs context richness
        
        SQL Process:
        1. JOIN embeddings with files for metadata
        2. Filter by tenant/model/strategy
        3. ORDER BY cosine distance (closest vectors first)
        4. LIMIT to top_k results
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

    async def keyword_search(
        self,
        query: str,
        tenant_id: int,
        embedding_model: str,
        chunking_strategy: str,
        top_k: int,
        db: AsyncSession
    ) -> List[SearchResult]:
        """
        PostgreSQL full-text search for keyword matching - Traditional Search Method
        
        WHY KEYWORD SEARCH?
        - Complements semantic search by finding exact term matches
        - Essential for proper nouns, technical terms, and specific phrases
        - Faster than vector search for simple queries
        - Provides fallback when semantic search misses obvious matches
        
        EXAMPLE:
        Query: "API key authentication"
        Finds: chunks containing exactly "API", "key", "authentication"
        vs Semantic: might find "login credentials" or "access tokens"
        
        PostgreSQL Full-Text Search:
        - to_tsvector(): converts text to searchable tokens
        - plainto_tsquery(): converts query to search terms
        - ts_rank(): scores relevance based on term frequency/position
        - @@ operator: matches query against document vectors
        """
        
        sql_query = text("""
            SELECT 
                e.chunk_text,
                e.chunk_index,
                e.chunk_metadata,
                f.id as file_id,
                f.filename,
                f.file_path,
                ts_rank(to_tsvector('english', e.chunk_text), plainto_tsquery('english', :query)) as keyword_score
            FROM embeddings e
            JOIN files f ON e.file_id = f.id
            WHERE f.tenant_id = :tenant_id
            AND e.embedding_model = :embedding_model
            AND e.chunking_strategy = :chunking_strategy
            AND to_tsvector('english', e.chunk_text) @@ plainto_tsquery('english', :query)
            ORDER BY ts_rank(to_tsvector('english', e.chunk_text), plainto_tsquery('english', :query)) DESC
            LIMIT :top_k
        """)

        result = await db.execute(sql_query, {
            "query": query,
            "tenant_id": tenant_id,
            "embedding_model": embedding_model,
            "chunking_strategy": chunking_strategy,
            "top_k": top_k
        })

        search_results = []
        for row in result:
            search_result = SearchResult(
                chunk_text=row.chunk_text,
                similarity=float(row.keyword_score),
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
        """
        Hybrid search combining semantic and keyword results - Best of Both Worlds
        
        WHY HYBRID SEARCH?
        - Semantic search: finds conceptually similar content (meaning-based)
        - Keyword search: finds exact term matches (precision-based)
        - Hybrid: combines both strengths while avoiding weaknesses
        - Weighted scoring prevents one method from dominating results
        
        EXAMPLE:
        Query: "reset password"
        Semantic finds: "change your login credentials", "update account access"
        Keyword finds: exact matches for "reset" AND "password"
        Hybrid: Returns both types, weighted by semantic_weight (0.7 default)
        
        ALGORITHM:
        1. Run both searches in parallel
        2. Deduplicate results by chunk location
        3. Weight scores: semantic_weight * semantic_score + (1-weight) * keyword_score
        4. Sort by combined score and return top_k
        
        PRODUCTION BENEFITS:
        - Better recall (finds more relevant results)
        - Better precision (exact matches ranked highly)
        - Handles diverse query types (technical terms + concepts)
        """
        
        # Get results from both methods
        semantic_results = await self.similarity_search(
            query_embedding, tenant_id, embedding_model, chunking_strategy, top_k, db
        )
        
        keyword_results = await self.keyword_search(
            query, tenant_id, embedding_model, chunking_strategy, top_k, db
        )
        
        # Simple combination: merge and deduplicate by chunk
        seen_chunks = set()
        combined_results = []
        
        # Add semantic results first (higher weight)
        for result in semantic_results:
            chunk_key = f"{result.file_path}_{result.chunk_index}"
            if chunk_key not in seen_chunks:
                result.similarity = semantic_weight * result.similarity
                combined_results.append(result)
                seen_chunks.add(chunk_key)
        
        # Add keyword results if not already seen
        keyword_weight = 1.0 - semantic_weight
        for result in keyword_results:
            chunk_key = f"{result.file_path}_{result.chunk_index}"
            if chunk_key not in seen_chunks:
                result.similarity = keyword_weight * result.similarity
                combined_results.append(result)
                seen_chunks.add(chunk_key)
        
        # Sort by combined score and return top_k
        combined_results.sort(key=lambda x: x.similarity, reverse=True)
        return combined_results[:top_k]

    def get_available_techniques(self) -> List[dict]:
        """Get list of available RAG techniques"""
        return [
            {
                "name": "similarity_search",
                "description": "Basic cosine similarity search using pgvector",
                "default": True
            },
            {
                "name": "hybrid_search",
                "description": "Combines semantic similarity with keyword matching",
                "default": False
            }
        ]
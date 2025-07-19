"""
Core Search Service - Foundation of RAG Systems

This module demonstrates basic vector similarity search - the core concept behind RAG:

WHY START HERE?
- Vector similarity is the foundation of all RAG systems
- Demonstrates core concepts: embeddings, similarity, ranking
- Simple, focused implementation for learning
- Building block for more advanced techniques

CORE CONCEPTS TAUGHT:
- Text → Vector conversion (embeddings)
- Vector similarity computation (cosine similarity)  
- Similarity-based ranking and retrieval
- Database vector operations with pgvector
"""

from typing import List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text
import numpy as np

from models import Embedding, File, SearchResult
from services.embedding_service import EmbeddingService


class CoreSearchService:
    """
    Core search functionality - Basic vector similarity search
    
    This service teaches fundamental RAG concepts through the simplest possible implementation
    """
    
    def __init__(self):
        self.embedding_service = EmbeddingService()
    
    async def similarity_search(
        self, query_embedding: List[float], tenant_id: int,
        embedding_model: str, chunking_strategy: str, top_k: int, db: AsyncSession
    ) -> List[SearchResult]:
        """
        Basic vector similarity search - Core RAG Functionality
        
        WHY VECTOR SIMILARITY?
        - Finds semantically similar content, not just keyword matches
        - Works across languages and synonyms
        - Captures conceptual relationships between query and documents
        
        HOW IT WORKS:
        1. Query is converted to vector (embedding)
        2. Compare query vector to all document chunk vectors
        3. Rank by similarity score (cosine similarity)
        4. Return top-k most similar chunks
        
        TECHNICAL DETAILS:
        - Uses PostgreSQL pgvector extension for efficient vector operations
        - Cosine similarity: 1 - (vector1 <=> vector2) where <=> is distance operator
        - Higher scores = more similar content
        """
        try:
            query_vector = str(query_embedding)
            
            # Simple vector similarity query using pgvector
            sql_query = text("""
                SELECT 
                    e.chunk_text, e.chunk_index, e.chunk_metadata,
                    f.filename, f.file_path,
                    1 - (e.embedding <=> :query_vector) as similarity_score
                FROM embeddings e
                JOIN files f ON e.file_id = f.id
                WHERE f.tenant_id = :tenant_id
                AND e.embedding_model = :embedding_model
                AND e.chunking_strategy = :chunking_strategy
                ORDER BY e.embedding <=> :query_vector
                LIMIT :top_k
            """)
            
            result = await db.execute(sql_query, {
                "query_vector": query_vector,
                "tenant_id": tenant_id,
                "embedding_model": embedding_model,
                "chunking_strategy": chunking_strategy,
                "top_k": top_k
            })
            
            # Convert database results to SearchResult objects
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
            
        except Exception as e:
            print(f"Error in similarity search: {e}")
            return []
    
    async def generate_query_embedding(self, query: str, model_name: str) -> List[float]:
        """
        Convert query text to vector embedding - Text → Vector Conversion
        
        WHY QUERY EMBEDDINGS?
        - Enables semantic comparison between query and document chunks
        - Same model used for both query and documents ensures compatibility
        - Vector representation captures semantic meaning, not just keywords
        
        PROCESS:
        1. Use same embedding model that created document embeddings
        2. Convert query text to vector representation
        3. Return vector for similarity comparison
        """
        return await self.embedding_service.generate_query_embedding(query, model_name)
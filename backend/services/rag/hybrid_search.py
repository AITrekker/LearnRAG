"""
Hybrid Search Service - Combining Semantic and Keyword Search

This module demonstrates advanced RAG techniques by combining multiple search signals:

WHY HYBRID SEARCH?
- Vector similarity captures semantic meaning but may miss exact terms
- Keyword search finds precise matches but misses semantic relationships  
- Combining both provides comprehensive retrieval coverage
- Improves recall for both conceptual and factual queries

HYBRID APPROACH:
- Semantic search: Vector similarity for conceptual matches
- Keyword search: Full-text search for exact term matches
- Score combination: Weighted blend of both signals
- Result fusion: Merge and rank combined results
"""

from typing import List
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text
import numpy as np

from models import Embedding, File, SearchResult
from .core_search import CoreSearchService


class HybridSearchService:
    """
    Hybrid search combining semantic similarity with keyword matching
    
    This demonstrates how to improve RAG by combining multiple retrieval signals
    """
    
    def __init__(self):
        self.core_search = CoreSearchService()
    
    async def search(
        self, query: str, query_embedding: List[float], tenant_id: int,
        embedding_model: str, chunking_strategy: str, top_k: int, db: AsyncSession
    ) -> List[SearchResult]:
        """
        Hybrid search combining semantic and keyword signals
        
        WHY COMBINE SEARCH METHODS?
        - Semantic search: Good for conceptual queries ("who was the captain")
        - Keyword search: Good for specific terms ("Captain Ahab", exact names)
        - Together: Comprehensive coverage for different query types
        
        SEARCH PROCESS:
        1. Semantic search: Find conceptually similar chunks using vectors
        2. Keyword search: Find chunks containing query terms using full-text search  
        3. Score combination: Blend semantic and keyword scores
        4. Result fusion: Merge results and re-rank by combined scores
        
        SCORING STRATEGY:
        - 70% semantic similarity (for meaning)
        - 30% keyword relevance (for specific terms)
        - Adjustable weights based on query characteristics
        """
        try:
            # Get semantic similarity results (vector search)
            semantic_results = await self._semantic_search(
                query_embedding, tenant_id, embedding_model, chunking_strategy, top_k * 2, db
            )
            
            # Get keyword search results (full-text search)
            keyword_results = await self._keyword_search(
                query, tenant_id, embedding_model, chunking_strategy, top_k * 2, db
            )
            
            # Combine and rank results
            combined_results = self._combine_search_results(
                semantic_results, keyword_results, semantic_weight=0.7, keyword_weight=0.3
            )
            
            return combined_results[:top_k]
            
        except Exception as e:
            print(f"Error in hybrid search: {e}")
            # Fallback to semantic search only
            return await self.core_search.similarity_search(
                query_embedding, tenant_id, embedding_model, chunking_strategy, top_k, db
            )
    
    async def _semantic_search(
        self, query_embedding: List[float], tenant_id: int,
        embedding_model: str, chunking_strategy: str, top_k: int, db: AsyncSession
    ) -> List[SearchResult]:
        """Semantic search using vector similarity"""
        return await self.core_search.similarity_search(
            query_embedding, tenant_id, embedding_model, chunking_strategy, top_k, db
        )
    
    async def _keyword_search(
        self, query: str, tenant_id: int, embedding_model: str, 
        chunking_strategy: str, top_k: int, db: AsyncSession
    ) -> List[SearchResult]:
        """
        Keyword search using PostgreSQL full-text search
        
        WHY FULL-TEXT SEARCH?
        - Finds exact word matches that vectors might miss
        - Handles proper nouns, technical terms, specific phrases
        - PostgreSQL's built-in ranking for text relevance
        - Fast and efficient for large document collections
        """
        try:
            # PostgreSQL full-text search with ranking
            sql_query = text("""
                SELECT 
                    e.chunk_text, e.chunk_index, e.chunk_metadata,
                    f.filename, f.file_path,
                    ts_rank(to_tsvector('english', e.chunk_text), plainto_tsquery('english', :query)) as text_score
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
            
            keyword_results = []
            for row in result:
                search_result = SearchResult(
                    chunk_text=row.chunk_text,
                    similarity=float(row.text_score),  # Use text relevance as similarity
                    file_name=row.filename,
                    file_path=row.file_path,
                    chunk_index=row.chunk_index,
                    chunk_metadata=row.chunk_metadata or {}
                )
                keyword_results.append(search_result)
            
            return keyword_results
            
        except Exception as e:
            print(f"Error in keyword search: {e}")
            return []
    
    def _combine_search_results(
        self, semantic_results: List[SearchResult], keyword_results: List[SearchResult],
        semantic_weight: float = 0.7, keyword_weight: float = 0.3
    ) -> List[SearchResult]:
        """
        Combine semantic and keyword search results - Result Fusion
        
        WHY WEIGHTED COMBINATION?
        - Different search methods have different strength patterns
        - Semantic search: Better for conceptual queries
        - Keyword search: Better for specific term queries
        - Weighted combination balances both strengths
        
        FUSION PROCESS:
        1. Normalize scores from both methods to 0-1 range
        2. Apply weights to each score type
        3. Combine scores for chunks that appear in both result sets
        4. Include unique results from each method
        5. Sort by final combined score
        """
        # Create combined results map
        combined_map = {}
        
        # Normalize and add semantic results
        max_semantic_score = max([r.similarity for r in semantic_results]) if semantic_results else 1.0
        for result in semantic_results:
            key = f"{result.file_name}:{result.chunk_index}"
            normalized_semantic = result.similarity / max_semantic_score
            combined_map[key] = {
                'result': result,
                'semantic_score': normalized_semantic,
                'keyword_score': 0.0
            }
        
        # Normalize and add keyword results
        max_keyword_score = max([r.similarity for r in keyword_results]) if keyword_results else 1.0
        for result in keyword_results:
            key = f"{result.file_name}:{result.chunk_index}"
            normalized_keyword = result.similarity / max_keyword_score
            
            if key in combined_map:
                # Chunk found in both searches - update keyword score
                combined_map[key]['keyword_score'] = normalized_keyword
            else:
                # Keyword-only result
                combined_map[key] = {
                    'result': result,
                    'semantic_score': 0.0,
                    'keyword_score': normalized_keyword
                }
        
        # Calculate combined scores and create final results
        combined_results = []
        for key, data in combined_map.items():
            # Weighted combination of normalized scores
            combined_score = (
                data['semantic_score'] * semantic_weight + 
                data['keyword_score'] * keyword_weight
            )
            
            # Update result with combined score
            result = data['result']
            result.similarity = combined_score
            combined_results.append(result)
        
        # Sort by combined score and return
        combined_results.sort(key=lambda x: x.similarity, reverse=True)
        return combined_results
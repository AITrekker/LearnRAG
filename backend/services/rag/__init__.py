"""
RAG Service Module - Main Interface for Retrieval-Augmented Generation

This module demonstrates clean architecture patterns for complex RAG systems:

WHY MODULAR DESIGN?
- Each search technique has focused, single-responsibility modules
- Easier to understand, test, and maintain individual components
- Students can study one technique at a time
- Clear separation between core search logic and result formatting

MODULAR STRUCTURE:
- core_search.py: Basic vector similarity search (foundational RAG)
- hybrid_search.py: Semantic + keyword search combination
- hierarchical_search.py: Multi-level document → section → chunk search
- result_converter.py: Standardized result formatting across techniques

TEACHING APPROACH:
- Start with core_search.py to understand basic RAG concepts
- Progress to hybrid_search.py for combining multiple signals
- Advanced students explore hierarchical_search.py for complex retrieval
"""

from .core_search import CoreSearchService
from .hybrid_search import HybridSearchService  
from .hierarchical_search import HierarchicalSearchService
from .result_converter import ResultConverter

from typing import List
from sqlalchemy.ext.asyncio import AsyncSession
from models import SearchResult, HierarchicalSearchResult


class RagService:
    """
    Main RAG Service - Unified Interface for All Search Techniques
    
    WHY UNIFIED INTERFACE?
    - Single entry point for all RAG operations
    - Consistent API regardless of search technique used
    - Easy to switch between techniques for comparison
    - Maintains backward compatibility with existing code
    
    DELEGATION PATTERN:
    - Routes requests to appropriate specialized service
    - Handles result format conversion
    - Provides fallback mechanisms for robustness
    """
    
    def __init__(self):
        self.core_search_service = CoreSearchService()
        self.hybrid_search_service = HybridSearchService()
        self.hierarchical_search_service = HierarchicalSearchService()
        self.result_converter = ResultConverter()
    
    async def similarity_search(
        self, query_embedding: List[float], tenant_id: int,
        embedding_model: str, chunking_strategy: str, top_k: int, db: AsyncSession
    ) -> List[SearchResult]:
        """Basic vector similarity search - Foundation of RAG"""
        return await self.core_search_service.similarity_search(
            query_embedding, tenant_id, embedding_model, chunking_strategy, top_k, db
        )
    
    async def generate_query_embedding(self, query: str, model_name: str) -> List[float]:
        """Generate query embedding using core search service"""
        return await self.core_search_service.generate_query_embedding(query, model_name)
    
    async def hybrid_search(
        self, query: str, query_embedding: List[float], tenant_id: int,
        embedding_model: str, chunking_strategy: str, top_k: int, db: AsyncSession
    ) -> List[SearchResult]:
        """Hybrid semantic + keyword search"""
        return await self.hybrid_search_service.search(
            query, query_embedding, tenant_id, embedding_model, chunking_strategy, top_k, db
        )
    
    async def hierarchical_search(
        self, query: str, query_embedding: List[float], tenant_id: int,
        embedding_model: str, chunking_strategy: str, top_k: int, db: AsyncSession
    ) -> List[HierarchicalSearchResult]:
        """Multi-level hierarchical search"""
        return await self.hierarchical_search_service.search(
            query, query_embedding, tenant_id, embedding_model, chunking_strategy, top_k, db
        )
    
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
            },
            {
                "name": "hierarchical_search",
                "description": "Multi-level search using document/section summaries for better context",
                "default": False
            }
        ]
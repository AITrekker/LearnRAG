"""
Hierarchical Search Service - Multi-Level Document Retrieval

This module demonstrates advanced RAG architecture using document hierarchy:

WHY HIERARCHICAL SEARCH?
- Solves "needle in haystack" problem for character/entity queries
- Uses document structure: document → section → chunk progression
- Provides better context through multi-level summaries
- Improves precision by filtering irrelevant content early

HIERARCHICAL APPROACH:
- Level 1: Document summaries for high-level relevance
- Level 2: Section summaries for mid-level filtering  
- Level 3: Chunk retrieval with full hierarchical context
- Context enrichment: Each chunk includes document and section context
"""

from typing import List, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text

from models import HierarchicalSearchResult, SearchResult
from .core_search import CoreSearchService
from .result_converter import ResultConverter


class HierarchicalSearchService:
    """
    Hierarchical search using document and section summaries
    
    This demonstrates advanced RAG techniques for complex document understanding
    """
    
    def __init__(self):
        self.core_search = CoreSearchService()
        self.result_converter = ResultConverter()
    
    async def search(
        self, query: str, query_embedding: List[float], tenant_id: int,
        embedding_model: str, chunking_strategy: str, top_k: int, db: AsyncSession
    ) -> List[HierarchicalSearchResult]:
        """
        Hierarchical search - Multi-level RAG for better character/topic queries
        
        WHY HIERARCHICAL SEARCH?
        - Solves "needle in haystack" problem (e.g., "who was ahab" gets better results)
        - Uses document structure: document → section → chunk progression
        - Filters irrelevant content early for precision
        
        EXAMPLE IMPROVEMENT:
        Query: "who was ahab"
        - Regular search: finds random chunks, may return "mild white hairs"
        - Hierarchical: finds documents about Captain Ahab, then relevant sections
        
        SIMPLE 2-STEP PROCESS:
        1. Find relevant documents using summaries
        2. Get best chunks from those documents with context
        """
        # Step 1: Find documents that mention our query topic
        relevant_documents = await self._find_relevant_documents(query, tenant_id, db)
        
        if not relevant_documents:
            print("No relevant documents found, using regular search")
            # Fallback to regular search and convert results
            regular_results = await self.core_search.similarity_search(
                query_embedding, tenant_id, embedding_model, chunking_strategy, top_k, db
            )
            return [self.result_converter.to_hierarchical_result(r) for r in regular_results]
        
        # Step 2: Get chunks from relevant documents with hierarchical context
        return await self._get_chunks_from_documents(
            query_embedding, relevant_documents, embedding_model, chunking_strategy, top_k, db
        )
    
    async def _find_relevant_documents(self, query: str, tenant_id: int, db: AsyncSession) -> List[Dict[str, Any]]:
        """
        Find documents that mention the query topic - Simple Text Search
        
        WHY DOCUMENT-LEVEL FILTERING?
        - Quickly eliminates irrelevant documents (e.g., financial docs for character queries)
        - Uses PostgreSQL full-text search for fast keyword matching
        - Reduces search space before expensive vector operations
        
        TECHNICAL DETAILS:
        - Uses PostgreSQL's built-in text search for simplicity
        - Searches document summaries (not raw content) for efficiency
        - Returns top 3 matching documents to keep results focused
        """
        try:
            # Simple PostgreSQL text search in document summaries
            sql_query = text("""
                SELECT ds.file_id, f.filename, f.file_path, ds.summary_text
                FROM document_summaries ds
                JOIN files f ON ds.file_id = f.id
                WHERE f.tenant_id = :tenant_id
                AND to_tsvector('english', ds.summary_text) @@ plainto_tsquery('english', :query)
                ORDER BY ts_rank(to_tsvector('english', ds.summary_text), plainto_tsquery('english', :query)) DESC
                LIMIT 3
            """)
            
            result = await db.execute(sql_query, {"query": query, "tenant_id": tenant_id})
            return [{"file_id": row.file_id, "filename": row.filename, "file_path": row.file_path} for row in result]
            
        except Exception as e:
            print(f"Error finding relevant documents: {e}")
            return []
    
    async def _get_chunks_from_documents(
        self, query_embedding: List[float], relevant_documents: List[Dict[str, Any]],
        embedding_model: str, chunking_strategy: str, top_k: int, db: AsyncSession
    ) -> List[HierarchicalSearchResult]:
        """
        Get best chunks from relevant documents - Vector Search with Context
        
        WHY LIMIT TO RELEVANT DOCUMENTS?
        - Focuses search on documents likely to contain the answer
        - Provides hierarchical context (document + section info)
        - Better than searching all chunks across all documents
        
        TECHNICAL DETAILS:
        - Uses vector similarity within filtered document set
        - Includes document and section context for richer results
        - Returns chunks with full hierarchical metadata
        """
        if not relevant_documents:
            return []
            
        try:
            file_ids = [doc['file_id'] for doc in relevant_documents]
            placeholders = ','.join([f':file_id_{i}' for i in range(len(file_ids))])
            query_vector = str(query_embedding)
            
            # Simple query: get chunks from relevant documents with context
            sql_query = text(f"""
                SELECT 
                    e.chunk_text, e.chunk_context, e.chunk_index, e.chunk_metadata,
                    f.filename, f.file_path,
                    COALESCE(ss.title, 'Section') as section_title,
                    COALESCE(ds.summary_text, 'Document') as document_summary,
                    1 - (e.embedding <=> :query_vector) as similarity_score
                FROM embeddings e
                JOIN files f ON e.file_id = f.id
                LEFT JOIN section_summaries ss ON e.section_id = ss.id
                LEFT JOIN document_summaries ds ON f.id = ds.file_id
                WHERE e.file_id IN ({placeholders})
                AND e.embedding_model = :embedding_model
                AND e.chunking_strategy = :chunking_strategy
                ORDER BY e.embedding <=> :query_vector
                LIMIT :top_k
            """)
            
            params = {
                "query_vector": query_vector,
                "embedding_model": embedding_model,
                "chunking_strategy": chunking_strategy,
                "top_k": top_k
            }
            for i, file_id in enumerate(file_ids):
                params[f'file_id_{i}'] = file_id
            
            result = await db.execute(sql_query, params)
            
            # Convert to hierarchical results
            hierarchical_results = []
            for row in result:
                hierarchical_result = HierarchicalSearchResult(
                    chunk_text=row.chunk_text,
                    chunk_context=row.chunk_context,
                    similarity=float(row.similarity_score),
                    file_name=row.filename,
                    file_path=row.file_path,
                    chunk_index=row.chunk_index,
                    section_title=row.section_title,
                    document_summary=row.document_summary,
                    chunk_metadata=row.chunk_metadata or {}
                )
                hierarchical_results.append(hierarchical_result)
            
            return hierarchical_results
            
        except Exception as e:
            print(f"Error getting chunks from documents: {e}")
            return []
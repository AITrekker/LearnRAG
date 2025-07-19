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

from models import Embedding, File, DocumentSummary, SectionSummary
from models import SearchResult, HierarchicalSearchResult
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

    async def hierarchical_search(
        self,
        query: str,
        query_embedding: List[float],
        tenant_id: int,
        embedding_model: str,
        chunking_strategy: str,
        top_k: int,
        db: AsyncSession
    ) -> List[HierarchicalSearchResult]:
        """
        Hierarchical search using document and section summaries - Multi-Level Retrieval
        
        WHY HIERARCHICAL SEARCH?
        - Solves the "needle in haystack" problem for character/entity queries
        - Query routing: find relevant documents first, then drill down to sections
        - Better context: returns chunks with document/section context
        - Improved precision: filters irrelevant sections before chunk retrieval
        
        EXAMPLE: "who was ahab"
        Traditional search: searches all chunks, may miss character introductions
        Hierarchical search:
        1. Find documents mentioning "Ahab" in document summaries
        2. Find sections about Ahab in those documents
        3. Retrieve chunks from relevant sections
        4. Return results with full hierarchical context
        
        SEARCH ALGORITHM:
        1. Search document summaries for query relevance
        2. Search section summaries within relevant documents
        3. Search chunks within relevant sections
        4. Combine and rank results with hierarchical context
        """
        
        # Step 1: Search document summaries
        relevant_documents = await self._search_document_summaries(
            query, query_embedding, tenant_id, db
        )
        
        if not relevant_documents:
            # Fallback to regular search if no relevant documents found
            print("No relevant documents found in summaries, falling back to regular search")
            regular_results = await self.similarity_search(
                query_embedding, tenant_id, embedding_model, chunking_strategy, top_k, db
            )
            return [self._convert_to_hierarchical_result(r) for r in regular_results]
        
        # Step 2: Search section summaries within relevant documents
        relevant_sections = await self._search_section_summaries(
            query, query_embedding, relevant_documents, db
        )
        
        # Step 3: Search chunks within relevant sections (or documents if no sections)
        if relevant_sections:
            hierarchical_results = await self._search_chunks_in_sections(
                query_embedding, relevant_sections, embedding_model, chunking_strategy, top_k, db
            )
        else:
            hierarchical_results = await self._search_chunks_in_documents(
                query_embedding, relevant_documents, embedding_model, chunking_strategy, top_k, db
            )
        
        return hierarchical_results[:top_k]

    async def _search_document_summaries(
        self, query: str, query_embedding: List[float], tenant_id: int, db: AsyncSession
    ) -> List[dict]:
        """Search document summaries for query relevance"""
        try:
            # Simple text matching in document summaries using PostgreSQL full-text search
            sql_query = text("""
                SELECT 
                    ds.id, ds.file_id, ds.summary_text,
                    f.filename, f.file_path,
                    ts_rank(to_tsvector('english', ds.summary_text), plainto_tsquery('english', :query)) as text_similarity
                FROM document_summaries ds
                JOIN files f ON ds.file_id = f.id
                WHERE f.tenant_id = :tenant_id
                AND to_tsvector('english', ds.summary_text) @@ plainto_tsquery('english', :query)
                ORDER BY ts_rank(to_tsvector('english', ds.summary_text), plainto_tsquery('english', :query)) DESC
                LIMIT 5
            """)
            
            result = await db.execute(sql_query, {
                "query": query,
                "tenant_id": tenant_id
            })
            
            documents = []
            for row in result:
                documents.append({
                    'summary_id': row.id,
                    'file_id': row.file_id,
                    'filename': row.filename,
                    'file_path': row.file_path,
                    'summary_text': row.summary_text,
                    'similarity': float(row.text_similarity)
                })
            
            return documents
            
        except Exception as e:
            print(f"Error searching document summaries: {e}")
            return []

    async def _search_section_summaries(
        self, query: str, query_embedding: List[float], relevant_documents: List[dict], db: AsyncSession
    ) -> List[dict]:
        """Search section summaries within relevant documents"""
        if not relevant_documents:
            return []
            
        try:
            document_ids = [doc['summary_id'] for doc in relevant_documents]
            placeholders = ','.join([f':doc_id_{i}' for i in range(len(document_ids))])
            
            sql_query = text(f"""
                SELECT 
                    ss.id, ss.document_summary_id, ss.section_number, ss.title, ss.summary_text,
                    ds.file_id, f.filename, f.file_path,
                    ts_rank(to_tsvector('english', ss.summary_text), plainto_tsquery('english', :query)) as text_similarity
                FROM section_summaries ss
                JOIN document_summaries ds ON ss.document_summary_id = ds.id
                JOIN files f ON ds.file_id = f.id
                WHERE ss.document_summary_id IN ({placeholders})
                AND to_tsvector('english', ss.summary_text) @@ plainto_tsquery('english', :query)
                ORDER BY ts_rank(to_tsvector('english', ss.summary_text), plainto_tsquery('english', :query)) DESC
                LIMIT 10
            """)
            
            params = {"query": query}
            for i, doc_id in enumerate(document_ids):
                params[f'doc_id_{i}'] = doc_id
            
            result = await db.execute(sql_query, params)
            
            sections = []
            for row in result:
                sections.append({
                    'section_id': row.id,
                    'document_summary_id': row.document_summary_id,
                    'section_number': row.section_number,
                    'title': row.title,
                    'summary_text': row.summary_text,
                    'file_id': row.file_id,
                    'filename': row.filename,
                    'file_path': row.file_path,
                    'similarity': float(row.text_similarity)
                })
            
            return sections
            
        except Exception as e:
            print(f"Error searching section summaries: {e}")
            return []

    async def _search_chunks_in_sections(
        self, query_embedding: List[float], relevant_sections: List[dict],
        embedding_model: str, chunking_strategy: str, top_k: int, db: AsyncSession
    ) -> List[HierarchicalSearchResult]:
        """Search chunks within relevant sections"""
        if not relevant_sections:
            return []
            
        try:
            section_ids = [section['section_id'] for section in relevant_sections]
            placeholders = ','.join([f':section_id_{i}' for i in range(len(section_ids))])
            query_vector = str(query_embedding)
            
            sql_query = text(f"""
                SELECT 
                    e.chunk_text, e.chunk_context, e.chunk_index, e.chunk_metadata,
                    f.id as file_id, f.filename, f.file_path,
                    ss.title as section_title, ss.summary_text as section_summary,
                    ds.summary_text as document_summary,
                    1 - (e.embedding <=> :query_vector) as similarity_score
                FROM embeddings e
                JOIN files f ON e.file_id = f.id
                LEFT JOIN section_summaries ss ON e.section_id = ss.id
                LEFT JOIN document_summaries ds ON f.id = ds.file_id
                WHERE e.section_id IN ({placeholders})
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
            for i, section_id in enumerate(section_ids):
                params[f'section_id_{i}'] = section_id
            
            result = await db.execute(sql_query, params)
            
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
            print(f"Error searching chunks in sections: {e}")
            return []

    async def _search_chunks_in_documents(
        self, query_embedding: List[float], relevant_documents: List[dict],
        embedding_model: str, chunking_strategy: str, top_k: int, db: AsyncSession
    ) -> List[HierarchicalSearchResult]:
        """Search chunks within relevant documents (fallback when no sections match)"""
        if not relevant_documents:
            return []
            
        try:
            file_ids = [doc['file_id'] for doc in relevant_documents]
            placeholders = ','.join([f':file_id_{i}' for i in range(len(file_ids))])
            query_vector = str(query_embedding)
            
            sql_query = text(f"""
                SELECT 
                    e.chunk_text, e.chunk_context, e.chunk_index, e.chunk_metadata,
                    f.id as file_id, f.filename, f.file_path,
                    ss.title as section_title, ss.summary_text as section_summary,
                    ds.summary_text as document_summary,
                    1 - (e.embedding <=> :query_vector) as similarity_score
                FROM embeddings e
                JOIN files f ON e.file_id = f.id
                LEFT JOIN section_summaries ss ON e.section_id = ss.id
                LEFT JOIN document_summaries ds ON f.id = ds.file_id
                WHERE f.id IN ({placeholders})
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
            print(f"Error searching chunks in documents: {e}")
            return []

    def _convert_to_hierarchical_result(self, search_result: SearchResult) -> HierarchicalSearchResult:
        """Convert regular SearchResult to HierarchicalSearchResult"""
        return HierarchicalSearchResult(
            chunk_text=search_result.chunk_text,
            chunk_context=None,
            similarity=search_result.similarity,
            file_name=search_result.file_name,
            file_path=search_result.file_path,
            chunk_index=search_result.chunk_index,
            section_title=None,
            document_summary=None,
            chunk_metadata=search_result.chunk_metadata
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
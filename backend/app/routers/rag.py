from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text
from typing import List

from app.core.database import get_db
from app.models.database import Tenant, File, Embedding, RagSession
from app.models.schemas import SearchRequest, SearchResponse, SearchResult
from app.routers.auth import get_current_tenant
from app.services.rag_service import RagService

router = APIRouter()


@router.post("/search", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    tenant: Tenant = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db)
):
    """Perform RAG search"""
    rag_service = RagService()
    
    # Generate query embedding
    query_embedding = await rag_service.generate_query_embedding(
        request.query, request.embedding_model
    )
    
    # Perform similarity search
    search_results = await rag_service.similarity_search(
        query_embedding,
        tenant.id,
        request.embedding_model,
        request.chunking_strategy,
        request.top_k,
        db
    )
    
    # Save search session
    session = RagSession(
        tenant_id=tenant.id,
        embedding_model=request.embedding_model,
        chunking_strategy=request.chunking_strategy,
        rag_technique=request.rag_technique,
        query=request.query,
        results=[result.dict() for result in search_results]
    )
    db.add(session)
    await db.commit()
    
    return SearchResponse(
        query=request.query,
        results=search_results,
        embedding_model=request.embedding_model,
        chunking_strategy=request.chunking_strategy,
        rag_technique=request.rag_technique,
        total_results=len(search_results)
    )


@router.get("/techniques")
async def get_rag_techniques():
    """Get list of available RAG techniques"""
    return {
        "techniques": [
            {
                "name": "similarity_search",
                "description": "Basic cosine similarity search",
                "default": True
            }
        ]
    }


@router.post("/compare")
async def compare_techniques(
    queries: List[str],
    techniques: List[str],
    tenant: Tenant = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db)
):
    """Compare different RAG techniques side by side"""
    # TODO: Implement in Phase 3
    return {"message": "Comparison feature coming in Phase 3"}


@router.get("/sessions")
async def get_rag_sessions(
    limit: int = 10,
    tenant: Tenant = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db)
):
    """Get recent RAG search sessions"""
    result = await db.execute(
        select(RagSession)
        .where(RagSession.tenant_id == tenant.id)
        .order_by(RagSession.created_at.desc())
        .limit(limit)
    )
    sessions = result.scalars().all()
    
    return {
        "sessions": [
            {
                "id": session.id,
                "query": session.query,
                "embedding_model": session.embedding_model,
                "chunking_strategy": session.chunking_strategy,
                "rag_technique": session.rag_technique,
                "result_count": len(session.results),
                "created_at": session.created_at
            }
            for session in sessions
        ]
    }
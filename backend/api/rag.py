from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, text
from typing import List

from database import get_db
from models import Tenant, File, Embedding, RagSession
from models import SearchRequest, AnswerRequest, CompareRequest
from models import SearchResponse, AnswerResponse, SearchResult, HierarchicalSearchResult
from api.auth import get_current_tenant
from services.rag_service import RagService
from services.rag.result_converter import ResultConverter
from services.llm_service import llm_service
from config import AVAILABLE_LLM_MODELS, PROMPT_TEMPLATES

router = APIRouter()


@router.post("/search", response_model=SearchResponse)
async def search(
    request: SearchRequest,
    tenant: Tenant = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db)
):
    """Perform RAG search"""
    rag_service = RagService()
    
    try:
        # Generate query embedding
        query_embedding = await rag_service.generate_query_embedding(
            request.query, request.embedding_model
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    
    # Perform search based on technique
    if request.rag_technique == "hybrid_search":
        search_results = await rag_service.hybrid_search(
            request.query,
            query_embedding,
            tenant.id,
            request.embedding_model,
            request.chunking_strategy,
            request.top_k,
            db
        )
    elif request.rag_technique == "hierarchical_search":
        # Get hierarchical results and convert using purpose-built converter
        hierarchical_results = await rag_service.hierarchical_search(
            request.query,
            query_embedding,
            tenant.id,
            request.embedding_model,
            request.chunking_strategy,
            request.top_k,
            db
        )
        # Use ResultConverter for clean conversion
        converter = ResultConverter()
        search_results = [converter.to_search_result(hr) for hr in hierarchical_results]
    else:
        # Default to similarity search
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
    rag_service = RagService()
    return {
        "techniques": rag_service.get_available_techniques()
    }


@router.post("/answer", response_model=AnswerResponse)
async def generate_answer(
    request: AnswerRequest,
    tenant: Tenant = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db)
):
    """Generate an answer from retrieved chunks using LLM"""
    rag_service = RagService()
    
    # Step 1: Generate query embedding and retrieve chunks
    try:
        query_embedding = await rag_service.generate_query_embedding(
            request.query, request.embedding_model
        )
    except ValueError as e:
        raise HTTPException(status_code=422, detail=str(e))
    
    # Retrieve more chunks for better context
    search_results = await rag_service.similarity_search(
        query_embedding,
        tenant.id,
        request.embedding_model,
        request.chunking_strategy,
        request.top_k,
        db
    )
    
    # Filter by minimum similarity threshold
    filtered_results = [
        result for result in search_results 
        if result.similarity >= request.min_similarity
    ]
    
    # Step 2: Generate answer using LLM
    answer_data = await llm_service.generate_answer(
        query=request.query,
        chunks=filtered_results,
        model_name=request.answer_model,
        prompt_template=request.prompt_template,
        max_length=request.max_length,
        temperature=request.temperature,
        top_p=request.top_p,
        repetition_penalty=request.repetition_penalty,
        context_chunks=request.context_chunks
    )
    
    # Step 3: Save answer session
    session = RagSession(
        tenant_id=tenant.id,
        embedding_model=request.embedding_model,
        chunking_strategy=request.chunking_strategy,
        rag_technique="answer_generation",
        query=request.query,
        results=[result.dict() for result in filtered_results],
        answer=answer_data["answer"],
        answer_model=answer_data["model_used"]
    )
    db.add(session)
    await db.commit()
    
    return AnswerResponse(
        query=request.query,
        answer=answer_data["answer"],
        confidence=float(answer_data["confidence"]),  # Convert back to float for API response
        sources=filtered_results,
        generation_time=float(answer_data["generation_time"]),  # Convert back to float for API response
        model_used=answer_data["model_used"],
        prompt_template=request.prompt_template,
        fallback_used=answer_data["error"] is not None,
        error=answer_data["error"]
    )


@router.post("/compare")
async def compare_techniques(
    request: CompareRequest,
    tenant: Tenant = Depends(get_current_tenant),
    db: AsyncSession = Depends(get_db)
):
    """Compare different RAG techniques side by side"""
    # TODO: Implement in Phase 3
    return {"message": "Comparison feature coming in Phase 3"}


@router.get("/llm-models")
async def get_llm_models():
    """Get available LLM models for answer generation"""
    return {"models": AVAILABLE_LLM_MODELS}


@router.get("/prompt-templates")
async def get_prompt_templates():
    """Get available prompt templates for answer generation"""
    templates = []
    for template_id, config in PROMPT_TEMPLATES.items():
        templates.append({
            "id": template_id,
            "name": config["name"],
            "description": config["description"]
        })
    return {"templates": templates}


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
"""
Embedding Service - Modular Text-to-Vector Conversion System

Teaching Purpose: This modular service demonstrates clean service architecture:

1. MAIN SERVICE INTERFACE: Unified API for embedding operations
2. FILE PROCESSOR: Handles file-level embedding generation and delta sync
3. CHUNKING STRATEGIES: Text splitting methods for optimal embedding
4. MODEL MANAGER: Model loading, caching, and validation

Core Architecture Benefits:
- Single Responsibility Principle: Each module has one focused task
- Dependency Injection: Clean separation of concerns
- Testability: Each module can be tested independently
- Maintainability: Easy to modify individual components
"""

from .file_processor import FileProcessor
from .chunking_strategies import ChunkingStrategies
from .model_manager import ModelManager
from typing import List, Dict, Any
from sqlalchemy.ext.asyncio import AsyncSession
from models import File


class EmbeddingService:
    """
    Main embedding service interface - coordinates all embedding operations
    
    Architecture Pattern: Facade + Dependency Injection
    - Provides simple unified API for complex operations
    - Delegates to specialized services for actual work
    - Manages cross-cutting concerns like error handling
    """
    
    def __init__(self):
        """Initialize embedding service with all dependencies"""
        self.model_manager = ModelManager()
        self.chunking_strategies = ChunkingStrategies()
        self.file_processor = FileProcessor(self.model_manager, self.chunking_strategies)

    async def generate_embeddings_for_files(
        self, 
        files: List[File], 
        embedding_model: str, 
        chunking_strategy: str, 
        db: AsyncSession,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        tenant_name: str = "Unknown"
    ):
        """
        Generate vector embeddings for multiple files - Main Entry Point
        
        Delegates to FileProcessor for actual implementation while providing
        a clean, simple interface for the API layer.
        """
        return await self.file_processor.generate_embeddings_for_files(
            files, embedding_model, chunking_strategy, db,
            chunk_size, chunk_overlap, tenant_name
        )

    async def generate_query_embedding(self, query: str, model_name: str) -> List[float]:
        """
        Generate embedding for a query - Search Interface
        
        Used by RAG search to convert queries to vectors for similarity search.
        """
        model = await self.model_manager.get_model(model_name)
        embedding = model.encode([query], convert_to_tensor=False)[0]
        return embedding.tolist()

    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get available embedding models - Configuration Interface"""
        return self.model_manager.get_available_models()
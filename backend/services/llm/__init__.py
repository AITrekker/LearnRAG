"""
LLM Service - Modular Answer Generation from Retrieved Context

Teaching Purpose: This modular service demonstrates LLM integration patterns:

1. MAIN SERVICE INTERFACE: Unified API for answer generation
2. MODEL MANAGER: Device detection, model loading, and memory management
3. ANSWER GENERATOR: Core inference logic with fallback handling
4. PROMPT ENGINEER: Template management and context formatting

Core Architecture Benefits:
- Separation of Concerns: Each module handles specific responsibilities
- Device Optimization: Automatic GPU/CPU detection and optimization
- Template System: Configurable prompt engineering
- Error Handling: Graceful degradation with fallbacks
"""

from .model_manager import ModelManager
from .answer_generator import AnswerGenerator
from .prompt_engineer import PromptEngineer
from typing import List, Dict, Any
from models import SearchResult


class LLMService:
    """
    Main LLM service interface - coordinates answer generation operations
    
    Architecture Pattern: Facade + Strategy + Dependency Injection
    - Provides simple unified API for complex LLM operations
    - Delegates to specialized components for actual work
    - Manages cross-cutting concerns like error handling and metrics
    """
    
    def __init__(self):
        """Initialize LLM service with all dependencies"""
        self.model_manager = ModelManager()
        self.prompt_engineer = PromptEngineer()
        self.answer_generator = AnswerGenerator(self.model_manager, self.prompt_engineer)

    async def generate_answer(
        self,
        query: str,
        chunks: List[SearchResult],
        model_name: str = "deepset/roberta-base-squad2",
        prompt_template: str = "factual",
        max_length: int = 200,
        temperature: float = 0.7,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        context_chunks: int = 5
    ) -> Dict[str, Any]:
        """
        Generate an answer from retrieved chunks using LLM - Main Entry Point
        
        Delegates to AnswerGenerator for actual implementation while providing
        a clean, simple interface for the API layer.
        """
        return await self.answer_generator.generate_answer(
            query, chunks, model_name, prompt_template, max_length,
            temperature, top_p, repetition_penalty, context_chunks
        )

    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get available LLM models - Configuration Interface"""
        return self.model_manager.get_available_models()

    def clear_model(self) -> None:
        """Clear loaded model to free memory - Memory Management Interface"""
        self.model_manager.clear_model()


# Global instance for backward compatibility
llm_service = LLMService()
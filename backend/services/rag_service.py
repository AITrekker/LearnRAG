"""
RAG Service - Clean, Modular Interface for Retrieval-Augmented Generation

This demonstrates modern software architecture patterns for complex RAG systems:

WHY MODULAR ARCHITECTURE?
- Single Responsibility: Each module handles one search technique
- Easy Testing: Individual components can be tested in isolation
- Clear Learning Path: Students can study one technique at a time
- Maintainability: Changes to one technique don't affect others

TEACHING PROGRESSION:
1. Start with core_search.py to understand basic vector similarity
2. Explore hybrid_search.py to learn about combining search signals
3. Study hierarchical_search.py for advanced document understanding
4. Examine result_converter.py for clean data transformation patterns

This replaces the previous 600+ line monolithic file with focused, teachable modules.
"""

# Import the modular RAG service from the rag package
from services.rag import RagService

# Export the main class for backward compatibility
__all__ = ['RagService']
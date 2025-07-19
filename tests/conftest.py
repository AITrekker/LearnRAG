"""
Test configuration for LearnRAG modular services

Provides shared fixtures and setup for testing the modular architecture.
"""

import pytest
import sys
import os
from pathlib import Path

# Add the backend directory to Python path for imports
backend_dir = Path(__file__).parent.parent / "backend"
sys.path.insert(0, str(backend_dir))

@pytest.fixture
def sample_search_results():
    """Fixture providing sample SearchResult objects for testing"""
    from models import SearchResult
    
    return [
        SearchResult(
            chunk_text="Machine learning is a subset of artificial intelligence.",
            similarity=0.92,
            file_name="ai_basics.txt",
            file_path="docs/ai_basics.txt",
            chunk_index=0,
            chunk_metadata={"section": "introduction"}
        ),
        SearchResult(
            chunk_text="Neural networks are inspired by biological neurons.",
            similarity=0.88,
            file_name="neural_networks.txt", 
            file_path="docs/neural_networks.txt",
            chunk_index=5,
            chunk_metadata={"section": "fundamentals"}
        ),
        SearchResult(
            chunk_text="Deep learning uses multiple layers to process data.",
            similarity=0.85,
            file_name="deep_learning.txt",
            file_path="docs/deep_learning.txt", 
            chunk_index=2,
            chunk_metadata={"section": "architecture"}
        )
    ]

@pytest.fixture
def sample_hierarchical_results():
    """Fixture providing sample HierarchicalSearchResult objects for testing"""
    from models import HierarchicalSearchResult
    
    return [
        HierarchicalSearchResult(
            chunk_text="Vector databases store high-dimensional data efficiently.",
            chunk_context="Document: Database Technologies. Section: Vector Storage. This section covers modern approaches to storing and querying vector embeddings for AI applications.",
            similarity=0.94,
            file_name="vector_db.txt",
            file_path="docs/vector_db.txt",
            chunk_index=3,
            section_title="Vector Storage Systems",
            document_summary="Comprehensive guide to vector databases and their role in AI applications.",
            chunk_metadata={"has_context": True, "processing_time": 0.15}
        ),
        HierarchicalSearchResult(
            chunk_text="Similarity search finds the most relevant documents.",
            chunk_context="Document: Search Algorithms. Section: Similarity Metrics. This section explains different approaches to measuring similarity between vectors and documents.",
            similarity=0.89,
            file_name="search_algos.txt", 
            file_path="docs/search_algos.txt",
            chunk_index=7,
            section_title="Similarity Metrics",
            document_summary="Technical overview of search algorithms used in information retrieval systems.",
            chunk_metadata={"has_context": True, "processing_time": 0.12}
        )
    ]

@pytest.fixture
def long_sample_text():
    """Fixture providing long text for chunking tests"""
    return """
    The field of artificial intelligence has evolved dramatically over the past few decades. 
    Machine learning, a subset of AI, focuses on algorithms that can learn from data without being explicitly programmed. 
    Deep learning, a further subset of machine learning, uses neural networks with multiple layers to process complex patterns.
    
    Natural language processing represents one of the most challenging areas in AI. 
    It involves teaching computers to understand, interpret, and generate human language. 
    Recent advances in transformer models have revolutionized this field.
    
    Computer vision is another critical domain of artificial intelligence. 
    It enables machines to interpret and understand visual information from the world. 
    Applications range from medical imaging to autonomous vehicles.
    
    The integration of these technologies has led to remarkable breakthroughs. 
    Multi-modal AI systems can now process text, images, and audio simultaneously. 
    This convergence opens new possibilities for human-computer interaction.
    """.strip()

@pytest.fixture(scope="session")
def test_config():
    """Session-scoped configuration for tests"""
    return {
        "chunk_sizes": [128, 256, 512],
        "overlap_sizes": [20, 50, 100],
        "similarity_thresholds": [0.7, 0.8, 0.9],
        "test_queries": [
            "What is machine learning?",
            "How do neural networks work?", 
            "Explain vector databases"
        ]
    }
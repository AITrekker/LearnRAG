"""
Unit Tests for Modular Services

Tests the new modular architecture components:
- RAG service modules
- Embedding service modules  
- LLM service modules
"""

import pytest
from unittest.mock import Mock, AsyncMock
from services.rag.result_converter import ResultConverter
from services.embedding.chunking_strategies import ChunkingStrategies
from services.llm.prompt_engineer import PromptEngineer
from models import SearchResult, HierarchicalSearchResult


class TestResultConverter:
    """Test result format conversion between different search types"""
    
    def setup_method(self):
        self.converter = ResultConverter()
        
    def test_hierarchical_to_search_result(self):
        """Test converting hierarchical result to basic search result"""
        hierarchical = HierarchicalSearchResult(
            chunk_text="Test content",
            chunk_context="Enhanced context",
            similarity=0.85,
            file_name="test.txt",
            file_path="path/test.txt",
            chunk_index=1,
            section_title="Introduction",
            document_summary="Test document",
            chunk_metadata={"test": "data"}
        )
        
        result = self.converter.to_search_result(hierarchical)
        
        assert result.chunk_text == "Test content"
        assert result.similarity == 0.85
        assert result.file_name == "test.txt"
        assert result.chunk_metadata["chunk_context"] == "Enhanced context"
        assert result.chunk_metadata["section_title"] == "Introduction"
        
    def test_search_to_hierarchical_result(self):
        """Test converting basic search result to hierarchical format"""
        search = SearchResult(
            chunk_text="Test content",
            similarity=0.85,
            file_name="test.txt",
            file_path="path/test.txt",
            chunk_index=1,
            chunk_metadata={"test": "data"}
        )
        
        result = self.converter.to_hierarchical_result(search)
        
        assert result.chunk_text == "Test content"
        assert result.similarity == 0.85
        assert result.file_name == "test.txt"
        assert result.chunk_context is None  # Not available in basic search
        assert result.section_title is None


class TestChunkingStrategies:
    """Test different text chunking approaches"""
    
    def setup_method(self):
        self.chunker = ChunkingStrategies()
        self.sample_text = "This is sentence one. This is sentence two. This is sentence three. This is sentence four."
        
    @pytest.mark.asyncio
    async def test_fixed_size_chunking(self):
        """Test fixed-size word-based chunking"""
        chunks = await self.chunker._fixed_size_chunking(self.sample_text, chunk_size=5, overlap=2)
        
        assert len(chunks) > 1
        assert all(len(chunk.split()) <= 7 for chunk in chunks)  # 5 + 2 overlap max
        
    @pytest.mark.asyncio
    async def test_sentence_chunking(self):
        """Test sentence-based chunking"""
        chunks = await self.chunker._sentence_based_chunking(self.sample_text, max_sentences=2)
        
        assert len(chunks) >= 2
        assert all("." in chunk for chunk in chunks)
        
    @pytest.mark.asyncio
    async def test_recursive_chunking(self):
        """Test recursive hierarchical chunking"""
        text_with_paragraphs = "Paragraph one.\n\nParagraph two with multiple sentences. Another sentence here.\n\nParagraph three."
        chunks = await self.chunker._recursive_chunking(text_with_paragraphs, chunk_size=10, overlap=2)
        
        assert len(chunks) > 0
        assert all(isinstance(chunk, str) for chunk in chunks)
        
    @pytest.mark.asyncio
    async def test_chunk_text_dispatcher(self):
        """Test the main chunking method dispatcher"""
        chunks = await self.chunker.chunk_text(self.sample_text, "fixed_size", chunk_size=5, overlap=1)
        assert len(chunks) > 0
        
        with pytest.raises(ValueError):
            await self.chunker.chunk_text(self.sample_text, "invalid_strategy")


class TestPromptEngineer:
    """Test prompt template management and formatting"""
    
    def setup_method(self):
        self.engineer = PromptEngineer()
        self.mock_chunks = [
            SearchResult(
                chunk_text="Test content one",
                similarity=0.9,
                file_name="test1.txt",
                file_path="path/test1.txt",
                chunk_index=0,
                chunk_metadata={}
            ),
            SearchResult(
                chunk_text="Test content two",
                similarity=0.8,
                file_name="test2.txt", 
                file_path="path/test2.txt",
                chunk_index=0,
                chunk_metadata={}
            )
        ]
        
    def test_create_prompt_factual(self):
        """Test factual prompt template creation"""
        prompt = self.engineer.create_prompt("What is the test about?", self.mock_chunks, "factual")
        
        assert "What is the test about?" in prompt
        assert "Test content one" in prompt
        assert "Test content two" in prompt
        assert "[Source 1]" in prompt
        assert "[Source 2]" in prompt
        
    def test_create_prompt_conversational(self):
        """Test conversational prompt template creation"""
        prompt = self.engineer.create_prompt("Explain this test", self.mock_chunks, "conversational")
        
        assert "Explain this test" in prompt
        assert "helpful assistant" in prompt.lower()
        
    def test_calculate_confidence_high(self):
        """Test confidence calculation for good answers"""
        answer = "This is a comprehensive answer with good length and detail."
        confidence = self.engineer.calculate_confidence(answer, self.mock_chunks)
        
        assert 0.5 <= confidence <= 0.95
        
    def test_calculate_confidence_low(self):
        """Test confidence calculation for poor answers"""
        answer = "I don't have enough information to answer this question."
        confidence = self.engineer.calculate_confidence(answer, self.mock_chunks)
        
        assert confidence == 0.2
        
    def test_extract_qa_context(self):
        """Test extracting question and context for QA models"""
        prompt = """Context:
This is test context.

Question: What is this about?

Answer:"""
        
        question, context = self.engineer.extract_qa_context(prompt)
        
        assert question == "What is this about?"
        assert "This is test context." in context
        
    def test_optimize_context_length(self):
        """Test context length optimization for model limits"""
        # Create chunks that exceed token limit
        long_chunks = [
            SearchResult(
                chunk_text=" ".join(["word"] * 200),  # ~200 words
                similarity=0.9,
                file_name="test1.txt",
                file_path="path/test1.txt", 
                chunk_index=0,
                chunk_metadata={}
            ),
            SearchResult(
                chunk_text=" ".join(["word"] * 200),  # ~200 words
                similarity=0.8,
                file_name="test2.txt",
                file_path="path/test2.txt",
                chunk_index=0,
                chunk_metadata={}
            ),
            SearchResult(
                chunk_text=" ".join(["word"] * 200),  # ~200 words
                similarity=0.7,
                file_name="test3.txt",
                file_path="path/test3.txt",
                chunk_index=0,
                chunk_metadata={}
            )
        ]
        
        optimized = self.engineer.optimize_context_length(long_chunks, max_tokens=300)
        
        # Should keep only the highest similarity chunks that fit
        assert len(optimized) < len(long_chunks)
        assert optimized[0].similarity >= optimized[-1].similarity  # Sorted by similarity
        
    def test_get_available_templates(self):
        """Test retrieving available prompt templates"""
        templates = self.engineer.get_available_templates()
        
        assert len(templates) > 0
        assert all("id" in template for template in templates)
        assert all("name" in template for template in templates)
        assert all("description" in template for template in templates)


# Test integration patterns
class TestServiceIntegration:
    """Test how modular services work together"""
    
    def test_result_converter_with_real_data(self):
        """Test result converter with realistic data"""
        converter = ResultConverter()
        
        # Simulate hierarchical search result
        hierarchical = HierarchicalSearchResult(
            chunk_text="Captain Ahab was the monomaniacal captain of the Pequod.",
            chunk_context="Document: Moby Dick analysis. Section: Character descriptions. This section discusses the main characters of the novel, focusing on their psychological profiles and motivations.",
            similarity=0.92,
            file_name="MobyDick.txt",
            file_path="setup/ACMECorp/MobyDick.txt",
            chunk_index=45,
            section_title="Character Analysis",
            document_summary="Analysis of Herman Melville's Moby Dick, covering themes, characters, and literary significance.",
            chunk_metadata={"has_context": True, "chunk_length": 58}
        )
        
        # Convert to basic format
        basic = converter.to_search_result(hierarchical)
        
        # Verify all important data preserved
        assert basic.chunk_text == hierarchical.chunk_text
        assert basic.similarity == hierarchical.similarity
        assert basic.file_name == hierarchical.file_name
        assert basic.chunk_metadata["chunk_context"] == hierarchical.chunk_context
        assert basic.chunk_metadata["section_title"] == hierarchical.section_title
        assert basic.chunk_metadata["document_summary"] == hierarchical.document_summary
        
        # Convert back to hierarchical
        back_to_hierarchical = converter.to_hierarchical_result(basic)
        
        # Basic conversion won't have hierarchical context
        assert back_to_hierarchical.chunk_text == basic.chunk_text
        assert back_to_hierarchical.similarity == basic.similarity
        assert back_to_hierarchical.chunk_context is None  # Lost in conversion
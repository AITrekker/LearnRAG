"""
Chunking Strategies - Text Splitting Methods for Optimal Embedding

Teaching Purpose: This module demonstrates text chunking approaches:

1. FIXED SIZE CHUNKING: Simple word-based splitting with overlap
2. SENTENCE CHUNKING: Natural language boundary preservation
3. RECURSIVE CHUNKING: Hierarchical splitting with fallbacks

Core Chunking Concepts Illustrated:
- Trade-offs between context preservation and precision
- Overlap strategies for context continuity
- Natural language boundaries for coherent chunks
- Hierarchical approaches for complex documents
"""

from typing import List
import re


class ChunkingStrategies:
    """
    Text chunking strategies for embedding generation
    
    Different strategies optimize for different use cases:
    - Fixed size: Predictable, uniform chunk sizes
    - Sentence: Natural language boundaries
    - Recursive: Hierarchical approach with fallbacks
    """

    async def chunk_text(self, text: str, strategy: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
        """
        Split text into chunks for embedding generation - Strategy Dispatcher
        
        WHY DIFFERENT CHUNKING STRATEGIES?
        - Fixed size: Consistent embedding dimensions, good for similarity search
        - Sentence: Preserves meaning, better for question answering
        - Recursive: Handles complex documents with mixed content types
        """
        if strategy == "fixed_size":
            return await self._fixed_size_chunking(text, chunk_size, overlap)
        elif strategy == "sentence":
            return await self._sentence_based_chunking(text)
        elif strategy == "recursive":
            return await self._recursive_chunking(text, chunk_size, overlap)
        else:
            raise ValueError(f"Unknown chunking strategy: {strategy}")

    async def _fixed_size_chunking(
        self, 
        text: str, 
        chunk_size: int = 512, 
        overlap: int = 50
    ) -> List[str]:
        """
        Fixed-size chunking with overlap - Word-Based Splitting
        
        WHY WORD-BASED CHUNKING?
        - Predictable chunk sizes for consistent embeddings
        - Overlap preserves context across chunk boundaries
        - Simple implementation, works for most document types
        
        OVERLAP STRATEGY:
        - Each chunk overlaps with previous by 'overlap' words
        - Prevents information loss at chunk boundaries
        - Improves retrieval for queries spanning chunks
        """
        words = text.split()
        chunks = []
        
        for i in range(0, len(words), chunk_size - overlap):
            chunk_words = words[i:i + chunk_size]
            chunk_text = " ".join(chunk_words)
            if chunk_text.strip():
                chunks.append(chunk_text)
        
        return chunks

    async def _sentence_based_chunking(self, text: str, max_sentences: int = 3) -> List[str]:
        """
        Sentence-based chunking with natural language boundaries
        
        WHY SENTENCE BOUNDARIES?
        - Preserves semantic coherence within chunks
        - Natural breakpoints don't split thoughts mid-sentence
        - Better for question-answering where context matters
        - Improves embedding quality for complex reasoning
        
        SENTENCE DETECTION:
        - Uses period, exclamation, question mark as delimiters
        - Handles common abbreviations and edge cases
        - Groups sentences to maintain reasonable chunk sizes
        """
        # Split on sentence boundaries
        sentences = re.split(r'[.!?]+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        current_chunk = []
        
        for sentence in sentences:
            current_chunk.append(sentence)
            
            # Group sentences up to max_sentences
            if len(current_chunk) >= max_sentences:
                chunk_text = '. '.join(current_chunk) + '.'
                chunks.append(chunk_text)
                current_chunk = []
        
        # Handle remaining sentences
        if current_chunk:
            chunk_text = '. '.join(current_chunk) + '.'
            chunks.append(chunk_text)
        
        return chunks

    async def _recursive_chunking(self, text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
        """
        Recursive chunking: paragraphs → sentences → fixed size
        
        WHY RECURSIVE APPROACH?
        - Respects document structure (paragraphs first)
        - Falls back to sentence boundaries when needed
        - Uses fixed-size as final fallback for very long content
        - Optimizes for both structure preservation and size constraints
        
        HIERARCHY STRATEGY:
        1. Try to keep paragraphs intact if they fit
        2. Split long paragraphs into sentences
        3. Split long sentences into fixed-size chunks
        """
        # First level: split by paragraphs
        paragraphs = text.split('\n\n')
        chunks = []
        
        for paragraph in paragraphs:
            paragraph = paragraph.strip()
            if not paragraph:
                continue
                
            words = paragraph.split()
            
            # If paragraph fits in chunk size, keep it intact
            if len(words) <= chunk_size:
                chunks.append(paragraph)
            else:
                # Split into sentences first
                sentence_chunks = await self._sentence_based_chunking(paragraph)
                for chunk in sentence_chunks:
                    chunk_words = chunk.split()
                    
                    # If sentence chunk fits, keep it
                    if len(chunk_words) <= chunk_size:
                        chunks.append(chunk)
                    else:
                        # Final fallback: fixed-size chunking
                        fixed_chunks = await self._fixed_size_chunking(chunk, chunk_size, overlap)
                        chunks.extend(fixed_chunks)
        
        return chunks
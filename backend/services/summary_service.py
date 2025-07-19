"""
Summary Generation Service - Hierarchical Content Summarization

This service demonstrates multi-level summarization for improved RAG performance:

1. DOCUMENT SUMMARIES: High-level overview of entire documents
2. SECTION SUMMARIES: Mid-level summaries of logical document sections  
3. CHUNK AWARENESS: Maintains relationship between summaries and chunks
4. LLM INTEGRATION: Uses local LLM for consistent summary generation
5. HIERARCHICAL RETRIEVAL: Enables top-down search from general to specific

Core Hierarchical RAG Concepts Illustrated:
- Multi-level content abstraction for better query matching
- Summary-first retrieval to identify relevant document sections
- Chunk-level retrieval within relevant sections for specific details
- Query routing based on abstraction level requirements
- Performance optimization through summary-based filtering
"""

import asyncio
from typing import List, Dict, Any, Optional
from pathlib import Path
import re
from transformers import pipeline, AutoTokenizer
import torch

from services.llm_service import llm_service
from exceptions import SystemError


class SummaryService:
    """
    Service for generating hierarchical summaries to improve RAG performance
    
    WHY HIERARCHICAL SUMMARIES?
    - Addresses the "needle in haystack" problem for character queries
    - Enables query routing: use summaries for general questions, chunks for specific details
    - Improves retrieval precision by matching queries to appropriate abstraction levels
    - Reduces irrelevant chunk retrieval through summary-based filtering
    
    SUMMARY LEVELS:
    1. Document Summary: 2-3 sentences capturing main themes, characters, plot
    2. Section Summary: 1-2 sentences per logical section (chapters, topics)
    3. Chunk Context: How each chunk relates to its section and document
    
    EXAMPLE IMPROVEMENT:
    Query: "who was ahab" 
    - Without summaries: Searches all chunks, may miss character introductions
    - With summaries: Finds document summary mentioning "Captain Ahab", then retrieves relevant sections
    """
    
    def __init__(self):
        self.llm_service = llm_service
        self.tokenizer = None
        self.summarizer = None
        self._initialize_models()
    
    def _initialize_models(self):
        """
        Initialize summarization models with proper resource management
        
        IMPROVEMENTS OVER PREVIOUS VERSION:
        - Use lighter, more stable models
        - Lazy loading (load on first use, not at startup)
        - Proper error handling and fallbacks
        - Resource management to prevent hangs
        """
        print(f"✅ Summary service initialized (models will load on first use)")
        self.summarizer = None  # Lazy loading
        self.tokenizer = None
        self._model_loading_attempted = False
    
    async def generate_document_summary(self, content: str, filename: str) -> str:
        """
        Generate high-level document summary - Document Overview
        
        WHY DOCUMENT SUMMARIES?
        - Provides quick overview for relevance assessment
        - Captures main themes, characters, and concepts
        - Enables fast filtering before detailed chunk search
        - Helps with query routing to relevant documents
        
        SUMMARY CONTENT:
        - Main topics and themes (2-3 sentences)
        - Key characters or entities mentioned
        - Document type and purpose
        - Time period or context if relevant
        
        EXAMPLE OUTPUT:
        "Moby Dick is a classic American novel about Captain Ahab's obsessive quest 
        to hunt the white whale that destroyed his leg. The story follows Ishmael, 
        a sailor who joins Ahab's crew aboard the Pequod for this dangerous voyage."
        """
        try:
            # Prepare content for summarization
            max_chars = 4000  # Reasonable limit
            if len(content) > max_chars:
                # Use beginning and a sample from middle for better context
                beginning = content[:max_chars//2]
                middle_start = len(content)//2 - max_chars//4
                middle_end = len(content)//2 + max_chars//4
                middle = content[middle_start:middle_end]
                summary_content = beginning + "\n\n" + middle
            else:
                summary_content = content
            
            # Try LLM-based summarization first
            if await self._ensure_models_loaded():
                summary = await self._summarize_with_transformer(summary_content)
                if summary and len(summary) > 20:  # Validate quality
                    return summary
            
            # Fallback to existing LLM service if available
            try:
                response = await self.llm_service.generate_answer(
                    query=f"Summarize this document in 2-3 sentences focusing on main themes and key points:\n\n{summary_content[:1500]}",
                    chunks=[],
                    model_name="google/flan-t5-base",
                    max_length=150,
                    temperature=0.3
                )
                summary = response.get("answer", "").strip()
                if summary and len(summary) > 20:
                    return summary
            except Exception as e:
                print(f"LLM fallback failed: {e}")
            
            # Final fallback: intelligent text extraction
            sentences = summary_content[:1000].split('.')[:3]
            sentences = [s.strip() for s in sentences if s.strip() and len(s) > 20]
            return '. '.join(sentences) + '.' if sentences else f"Document: {filename}"
            
        except Exception as e:
            print(f"Error generating document summary for {filename}: {e}")
            return f"Document: {filename} - Content available"
    
    async def generate_section_summaries(self, content: str, filename: str) -> List[Dict[str, Any]]:
        """
        Generate section-level summaries - Mid-Level Abstraction
        
        WHY SECTION SUMMARIES?
        - Bridges gap between document overview and detailed chunks
        - Enables hierarchical search: document → section → chunk
        - Improves precision for queries about specific topics
        - Reduces search space by filtering irrelevant sections
        
        SECTION DETECTION:
        - Chapter markers (Chapter 1, Chapter I, etc.)
        - Section headers (large text, numbered sections)
        - Natural breaks (multiple line breaks, page breaks)
        - Content length (split very long sections)
        
        SECTION SUMMARY CONTENT:
        - 1-2 sentences per section
        - Key events or topics in that section
        - Important characters or concepts introduced
        - How section relates to overall document
        """
        sections = await self._detect_sections(content, filename)
        section_summaries = []
        
        for i, section in enumerate(sections):
            try:
                section_content = section['content']
                
                # Try transformer summarization first
                summary = None
                if await self._ensure_models_loaded() and len(section_content) > 100:
                    summary = await self._summarize_with_transformer(section_content[:800])
                
                # Fallback to LLM service
                if not summary and len(section_content) > 50:
                    try:
                        response = await self.llm_service.generate_answer(
                            query=f"Summarize this section in 1-2 sentences:\n\n{section_content[:600]}",
                            chunks=[],
                            model_name="google/flan-t5-base",
                            max_length=100,
                            temperature=0.3
                        )
                        summary = response.get("answer", "").strip()
                    except Exception as e:
                        print(f"LLM section summarization failed: {e}")
                
                # Final fallback: smart text extraction
                if not summary or len(summary) < 10:
                    sentences = section_content[:600].split('.')[:2]
                    sentences = [s.strip() for s in sentences if s.strip() and len(s) > 15]
                    summary = '. '.join(sentences) + '.' if sentences else f"Section {i+1} content"
                
                section_summaries.append({
                    'section_number': i + 1,
                    'title': section.get('title', f"Section {i + 1}"),
                    'start_pos': section['start_pos'],
                    'end_pos': section['end_pos'],
                    'content_length': len(section['content']),
                    'summary': summary
                })
                
            except Exception as e:
                print(f"Error summarizing section {i+1}: {e}")
                section_summaries.append({
                    'section_number': i + 1,
                    'title': section.get('title', f"Section {i + 1}"),
                    'start_pos': section['start_pos'],
                    'end_pos': section['end_pos'],
                    'content_length': len(section['content']),
                    'summary': f"Section {i+1} content available"
                })
        
        return section_summaries
    
    async def _detect_sections(self, content: str, filename: str) -> List[Dict[str, Any]]:
        """
        Detect logical sections in document content - Content Structure Analysis
        
        WHY AUTOMATIC SECTION DETECTION?
        - Documents have natural logical divisions
        - Improves summary granularity and search precision
        - Enables section-level retrieval for better context
        - Handles various document formats automatically
        
        DETECTION STRATEGIES:
        1. Chapter markers: "Chapter 1", "Chapter I", "Chapter One"
        2. Numbered sections: "1.", "1.1", "Section 1"
        3. Header patterns: Large gaps, title case text
        4. Length-based splitting: Prevent overly long sections
        """
        sections = []
        
        # Look for chapter/section markers
        chapter_patterns = [
            r'(?i)^chapter\s+\d+',
            r'(?i)^chapter\s+[ivxlcdm]+',
            r'(?i)^section\s+\d+',
            r'^\d+\.\s+[A-Z]',
            r'^[A-Z][A-Z\s]{10,}$'  # All caps titles
        ]
        
        section_breaks = [0]  # Start with beginning of document
        
        lines = content.split('\n')
        current_pos = 0
        
        for i, line in enumerate(lines):
            line_stripped = line.strip()
            
            # Check for chapter/section patterns
            for pattern in chapter_patterns:
                if re.match(pattern, line_stripped):
                    # Calculate character position
                    char_pos = sum(len(l) + 1 for l in lines[:i])
                    if char_pos - current_pos > 500:  # Minimum section length
                        section_breaks.append(char_pos)
                        current_pos = char_pos
                    break
        
        # Add end of document
        section_breaks.append(len(content))
        
        # If no sections found, split by length
        if len(section_breaks) <= 2:
            section_breaks = []
            max_section_length = 3000
            for i in range(0, len(content), max_section_length):
                section_breaks.append(i)
            section_breaks.append(len(content))
        
        # Create section objects
        for i in range(len(section_breaks) - 1):
            start_pos = section_breaks[i]
            end_pos = section_breaks[i + 1]
            section_content = content[start_pos:end_pos].strip()
            
            if len(section_content) > 50:  # Skip tiny sections
                # Try to extract title from first line
                first_line = section_content.split('\n')[0].strip()
                title = first_line if len(first_line) < 100 else f"Section {len(sections) + 1}"
                
                sections.append({
                    'title': title,
                    'start_pos': start_pos,
                    'end_pos': end_pos,
                    'content': section_content
                })
        
        return sections
    
    async def _summarize_with_transformer(self, content: str) -> str:
        """Use transformer model for summarization"""
        try:
            # Truncate if too long for model
            max_tokens = 1024
            tokens = self.tokenizer.encode(content, truncation=True, max_length=max_tokens)
            truncated_content = self.tokenizer.decode(tokens, skip_special_tokens=True)
            
            # Generate summary
            summary_result = self.summarizer(
                truncated_content,
                max_length=100,
                min_length=30,
                do_sample=False
            )
            
            return summary_result[0]['summary_text']
            
        except Exception as e:
            print(f"Transformer summarization failed: {e}")
            # Fallback to simple truncation
            return content[:200] + "..." if len(content) > 200 else content
    
    async def _summarize_with_llm(self, content: str, level: str) -> str:
        """
        Fallback LLM-based summarization - Custom Prompt Engineering
        
        WHY CUSTOM LLM PROMPTS?
        - Tailored summaries for RAG use cases
        - Consistent summary structure across documents
        - Can specify focus areas (characters, themes, events)
        - Better control over summary length and style
        """
        try:
            if level == "document":
                prompt = f"""
                Summarize this document in 2-3 sentences. Focus on:
                - Main themes and topics
                - Key characters or people mentioned
                - Primary events or concepts
                
                Document content:
                {content[:2000]}...
                
                Summary:
                """
            else:  # section
                prompt = f"""
                Summarize this section in 1-2 sentences. Focus on:
                - Key events or topics in this section
                - Important details or concepts
                
                Section content:
                {content[:1000]}...
                
                Summary:
                """
            
            # SIMPLIFIED: Use basic text extraction instead of LLM for now
            # This avoids model loading issues and system hangs
            
            # Extract first few sentences as summary
            sentences = content[:1000].split('.')[:3]  # First 3 sentences, max 1000 chars
            summary = '. '.join([s.strip() for s in sentences if s.strip()]) + '.'
            
            return summary if summary != '.' else "Summary unavailable"
            
        except Exception as e:
            print(f"LLM summarization failed: {e}")
            # Final fallback
            sentences = content.split('.')[:3]
            return '. '.join(sentences) + '.' if sentences else "Summary unavailable"
    
    async def create_chunk_context(self, chunk_text: str, section_summary: str, document_summary: str) -> str:
        """
        Create hierarchical context for individual chunks - Contextual Enhancement
        
        WHY CHUNK CONTEXT?
        - Provides broader context for isolated text chunks
        - Helps embeddings capture relationships to larger themes
        - Improves retrieval by connecting specific details to general concepts
        - Enables better query matching across abstraction levels
        
        CONTEXT COMPONENTS:
        - How chunk relates to its section
        - How section relates to overall document
        - Key themes or characters mentioned in chunk
        - Importance of chunk content within larger narrative
        """
        context = f"""
        Document Context: {document_summary}
        
        Section Context: {section_summary}
        
        Specific Content: {chunk_text}
        """
        
        return context.strip()
    
    async def _ensure_models_loaded(self) -> bool:
        """
        Lazy model loading with proper resource management
        
        WHY LAZY LOADING?
        - Avoids startup hangs
        - Only loads when actually needed
        - Can fail gracefully without breaking the whole system
        """
        if self.summarizer is not None:
            return True
            
        if self._model_loading_attempted:
            return False  # Don't retry if already failed
            
        try:
            print("🔄 Loading summarization model (one-time setup)...")
            self._model_loading_attempted = True
            
            # Use a lighter, more stable model
            model_name = "google/flan-t5-small"  # Much lighter than base/large
            device = -1  # Force CPU to avoid GPU issues
            
            # Set timeout for model loading
            import asyncio
            
            def load_model():
                return pipeline(
                    "text2text-generation",
                    model=model_name,
                    device=device,
                    max_length=100,
                    do_sample=False,
                    model_kwargs={"torch_dtype": torch.float32}  # Explicit dtype
                )
            
            # Load with timeout
            self.summarizer = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, load_model),
                timeout=60.0  # 60 second timeout
            )
            
            print("✅ Summarization model loaded successfully")
            return True
            
        except asyncio.TimeoutError:
            print("⏰ Model loading timed out - using fallback summarization")
            self.summarizer = None
            return False
        except Exception as e:
            print(f"⚠️ Model loading failed: {e} - using fallback summarization")
            self.summarizer = None
            return False

    async def _summarize_with_transformer(self, content: str) -> str:
        """Safe transformer-based summarization with resource limits"""
        try:
            if not self.summarizer:
                return None
                
            # Limit input length
            max_input = 512
            if len(content) > max_input:
                content = content[:max_input]
            
            # Set timeout for inference
            import asyncio
            
            def generate_summary():
                return self.summarizer(
                    f"Summarize: {content}",
                    max_length=80,
                    min_length=20,
                    do_sample=False,
                    num_return_sequences=1
                )
            
            result = await asyncio.wait_for(
                asyncio.get_event_loop().run_in_executor(None, generate_summary),
                timeout=30.0  # 30 second timeout
            )
            
            return result[0]['generated_text'].strip()
            
        except asyncio.TimeoutError:
            print("⏰ Summarization timed out")
            return None
        except Exception as e:
            print(f"⚠️ Transformer summarization failed: {e}")
            return None
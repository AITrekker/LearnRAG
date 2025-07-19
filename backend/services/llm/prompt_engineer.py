"""
Prompt Engineer - Template Management and Context Formatting

Teaching Purpose: This module demonstrates prompt engineering patterns:

1. TEMPLATE SYSTEM: Configurable prompt templates for different use cases
2. CONTEXT ASSEMBLY: Intelligent chunk combination and formatting
3. CONFIDENCE SCORING: Answer quality assessment
4. PROMPT OPTIMIZATION: Template selection and parameter tuning

Core Concepts Illustrated:
- Template-based prompt engineering
- Context window management for large documents
- Answer quality metrics and confidence scoring
- Prompt optimization for different model types
"""

import logging
from typing import List, Dict, Any
from models import SearchResult
from config import PROMPT_TEMPLATES

logger = logging.getLogger(__name__)


class PromptEngineer:
    """
    Manages prompt templates and context formatting for LLM generation
    
    Key responsibilities:
    - Format retrieved chunks into coherent context
    - Apply configurable prompt templates
    - Calculate confidence scores for generated answers
    - Optimize prompts for different model architectures
    """

    def create_prompt(self, query: str, chunks: List[SearchResult], template_id: str = "factual") -> str:
        """
        Create a prompt for answer generation using configurable templates
        
        CONTEXT ASSEMBLY STRATEGY:
        - Combine top chunks into numbered sources
        - Preserve chunk metadata for traceability
        - Format context for optimal model comprehension
        - Apply template for consistent prompt structure
        
        WHY TEMPLATE SYSTEM?
        - Different use cases need different prompt styles
        - Templates encode prompt engineering best practices
        - Configurable without code changes
        - A/B testing different prompt approaches
        """
        # Combine chunks into context with source attribution
        context_parts = []
        for i, chunk in enumerate(chunks[:5]):  # Use top 5 chunks
            # Add source attribution for traceability
            context_parts.append(f"[Source {i+1}] {chunk.chunk_text}")
        
        context = "\n\n".join(context_parts)
        
        # Get template from config, fallback to factual if invalid
        template_config = PROMPT_TEMPLATES.get(template_id, PROMPT_TEMPLATES["factual"])
        template = template_config["template"]
        
        # Format template with context and query
        prompt = template.format(context=context, query=query)
        
        return prompt

    def calculate_confidence(self, answer: str, chunks: List[SearchResult]) -> float:
        """
        Calculate confidence score for the generated answer - Quality Assessment
        
        CONFIDENCE FACTORS:
        1. Answer length (not too short, not too long)
        2. Presence of "don't know" phrases (lower confidence)
        3. Average similarity of source chunks (higher = better context)
        4. Answer completeness and coherence
        
        WHY CONFIDENCE SCORING?
        - Help users assess answer reliability
        - Enable fallback strategies for low-confidence answers
        - Provide feedback for prompt engineering optimization
        - Support A/B testing of different approaches
        """
        if not answer or len(answer.strip()) < 10:
            return 0.1
        
        # Check for "don't know" responses
        dont_know_phrases = [
            "don't have enough information",
            "cannot answer",
            "not enough context",
            "insufficient information"
        ]
        
        answer_lower = answer.lower()
        if any(phrase in answer_lower for phrase in dont_know_phrases):
            return 0.2
        
        # Calculate average similarity of source chunks
        avg_similarity = sum(chunk.similarity for chunk in chunks) / len(chunks) if chunks else 0
        
        # Length-based confidence (optimal range: 50-300 chars)
        length_score = min(len(answer) / 200, 1.0)
        
        # Combine factors
        confidence = (avg_similarity * 0.6) + (length_score * 0.4)
        
        return min(confidence, 0.95)  # Cap at 95%

    def extract_qa_context(self, prompt: str) -> tuple[str, str]:
        """
        Extract question and context from prompt for QA models - QA Optimization
        
        WHY SEPARATE EXTRACTION?
        - QA models (like RoBERTa-SQuAD) expect question and context separately
        - Different from generative models that take full prompts
        - Enables architecture-specific optimization
        - Improves answer quality for extractive QA
        """
        lines = prompt.split('\n')
        question = ""
        context = ""
        
        for i, line in enumerate(lines):
            if line.startswith("Question:"):
                question = line.replace("Question:", "").strip()
            elif "Context:" in line or "Text:" in line:
                # Take everything after Context: until Question:
                context_start = i + 1
                for j in range(context_start, len(lines)):
                    if lines[j].startswith("Question:"):
                        break
                    context += lines[j] + " "
        
        return question.strip(), context.strip()

    def get_available_templates(self) -> List[Dict[str, str]]:
        """
        Get available prompt templates - Configuration Interface
        
        Returns template metadata including:
        - Template IDs for API selection
        - Human-readable names and descriptions
        - Use case recommendations
        """
        templates = []
        for template_id, config in PROMPT_TEMPLATES.items():
            templates.append({
                "id": template_id,
                "name": config["name"],
                "description": config["description"]
            })
        return templates

    def optimize_context_length(self, chunks: List[SearchResult], max_tokens: int = 512) -> List[SearchResult]:
        """
        Optimize context length for model limits - Token Management
        
        OPTIMIZATION STRATEGY:
        - Estimate token count using word count approximation
        - Prioritize highest similarity chunks
        - Truncate or remove chunks to fit limits
        - Preserve most relevant information
        
        WHY TOKEN MANAGEMENT?
        - Models have context window limits (512, 1024, 2048 tokens)
        - Exceeding limits causes truncation or errors
        - Better to select best chunks than truncate randomly
        - Improves answer quality by focusing on relevant content
        """
        # Rough token estimation: 1 token ≈ 0.75 words
        estimated_tokens = 0
        optimized_chunks = []
        
        # Sort chunks by similarity (highest first)
        sorted_chunks = sorted(chunks, key=lambda x: x.similarity, reverse=True)
        
        for chunk in sorted_chunks:
            # Estimate tokens for this chunk (words * 0.75 + some overhead)
            chunk_tokens = len(chunk.chunk_text.split()) * 0.75 + 10
            
            if estimated_tokens + chunk_tokens <= max_tokens:
                optimized_chunks.append(chunk)
                estimated_tokens += chunk_tokens
            else:
                break
        
        logger.info(f"Optimized context from {len(chunks)} to {len(optimized_chunks)} chunks ({estimated_tokens:.0f} tokens)")
        return optimized_chunks
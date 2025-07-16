"""
LLM service for answer generation from retrieved chunks
"""
import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5Tokenizer
import torch
from threading import Lock

# Ensure SentencePiece is available
try:
    import sentencepiece
except ImportError:
    logger = logging.getLogger(__name__)
    logger.warning("SentencePiece not found. Installing...")
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "sentencepiece"])
    import sentencepiece

from models import SearchResult

logger = logging.getLogger(__name__)

class LLMService:
    """Service for generating answers from retrieved chunks using local LLM"""
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_name = None
        self.device = None
        self.load_lock = Lock()
        
    def _detect_device(self) -> str:
        """Detect best available device (GPU first, then CPU)"""
        # Check for NVIDIA CUDA
        if torch.cuda.is_available():
            device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"Using NVIDIA GPU: {gpu_name}")
        # Check for Apple Silicon MPS (Metal Performance Shaders)
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            device = "mps"
            logger.info("Using Apple Silicon GPU (MPS)")
        else:
            device = "cpu"
            logger.info("Using CPU for LLM inference")
        return device
    
    def _load_model(self, model_name: str = "google/flan-t5-base") -> None:
        """Load the LLM model and tokenizer"""
        if self.model is not None and self.model_name == model_name:
            return  # Already loaded
            
        logger.info(f"Loading LLM model: {model_name}")
        start_time = time.time()
        
        try:
            # Detect device
            self.device = self._detect_device()
            
            # Load tokenizer - SentencePiece should now be available
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Load model with appropriate device mapping
            if self.device == "cuda":
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,  # Use full precision to avoid Half/Float issues
                    device_map="auto"
                )
            elif self.device == "mps":
                # Apple Silicon MPS optimization
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,  # MPS works best with float32
                    low_cpu_mem_usage=True
                )
                self.model.to(self.device)
            else:
                # CPU fallback
                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                self.model.to(self.device)
            
            self.model_name = model_name
            
            load_time = time.time() - start_time
            logger.info(f"Model loaded in {load_time:.2f} seconds on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
    
    def _create_prompt(self, query: str, chunks: List[SearchResult]) -> str:
        """Create a prompt for answer generation"""
        # Combine chunks into context
        context_parts = []
        for i, chunk in enumerate(chunks[:5]):  # Use top 5 chunks
            context_parts.append(f"[Source {i+1}] {chunk.chunk_text}")
        
        context = "\n\n".join(context_parts)
        
        # Create instruction prompt for T5
        prompt = f"""Answer the following question based only on the provided context. If the context doesn't contain enough information to answer the question, say "I don't have enough information to answer this question."

Context:
{context}

Question: {query}

Answer:"""
        
        return prompt
    
    def _calculate_confidence(self, answer: str, chunks: List[SearchResult]) -> float:
        """Calculate confidence score for the generated answer"""
        # Simple confidence calculation based on:
        # 1. Answer length (not too short, not too long)
        # 2. Presence of "I don't have enough information" 
        # 3. Average chunk similarity scores
        
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
    
    async def generate_answer(
        self,
        query: str,
        chunks: List[SearchResult],
        model_name: str = "google/flan-t5-base",
        max_length: int = 200,
        temperature: float = 0.3,
        top_p: float = 0.9,
        repetition_penalty: float = 1.1,
        context_chunks: int = 5
    ) -> Dict[str, Any]:
        """
        Generate an answer from retrieved chunks using LLM
        
        Args:
            query: User question
            chunks: Retrieved chunks from vector search
            model_name: LLM model to use
            max_length: Maximum answer length
            
        Returns:
            Dictionary with answer, confidence, and metadata
        """
        start_time = time.time()
        
        try:
            # Load model if needed (thread-safe)
            with self.load_lock:
                self._load_model(model_name)
            
            if not chunks:
                return {
                    "answer": "No relevant information found in the documents.",
                    "confidence": 0.1,
                    "model_used": model_name,
                    "generation_time": time.time() - start_time,
                    "error": None
                }
            
            # Use only the top chunks for context
            top_chunks = chunks[:context_chunks]
            
            # Create prompt
            prompt = self._create_prompt(query, top_chunks)
            logger.info(f"Generated prompt: {prompt[:200]}...")
            
            # Run inference in thread pool to avoid blocking
            answer = await asyncio.get_event_loop().run_in_executor(
                None, self._generate_sync, prompt, max_length, temperature, top_p, repetition_penalty
            )
            
            logger.info(f"Generated answer: {answer}")
            
            # Calculate confidence
            confidence = self._calculate_confidence(answer, chunks)
            
            generation_time = time.time() - start_time
            
            logger.info(f"Generated answer in {generation_time:.2f}s with confidence {confidence:.2f}")
            
            return {
                "answer": answer,
                "confidence": confidence,
                "model_used": model_name,
                "generation_time": generation_time,
                "error": None
            }
            
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return {
                "answer": "Error generating answer. Please try again.",
                "confidence": 0.0,
                "model_used": model_name,
                "generation_time": time.time() - start_time,
                "error": str(e)
            }
    
    def _generate_sync(self, prompt: str, max_length: int, temperature: float = 0.3, top_p: float = 0.9, repetition_penalty: float = 1.1) -> str:
        """Synchronous answer generation (runs in thread pool)"""
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                max_length=512,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate answer with configurable parameters
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_length=max_length,
                    num_return_sequences=1,
                    temperature=temperature,
                    do_sample=True,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                    pad_token_id=self.tokenizer.pad_token_id,
                    early_stopping=True
                )
            
            # Decode answer - T5 models return only the generated part
            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up answer and remove any artifacts
            answer = answer.strip()
            
            # Remove any remaining prompt artifacts
            if "Answer:" in answer:
                answer = answer.split("Answer:")[-1].strip()
            
            # Ensure we have a meaningful answer
            if not answer or len(answer) < 3:
                answer = "I don't have enough information to answer this question."
            
            return answer
            
        except Exception as e:
            logger.error(f"Error in synchronous generation: {e}")
            return "Error generating answer. Please try again."
    
    def _create_prompt(self, query: str, chunks: List[SearchResult]) -> str:
        """
        Create a prompt for the LLM using the query and retrieved chunks
        
        Args:
            query: User's question
            chunks: Retrieved chunks from vector search
            
        Returns:
            Formatted prompt string
        """
        # Build context from chunks
        context_parts = []
        for i, chunk in enumerate(chunks[:5]):  # Limit to top 5 chunks
            context_parts.append(f"Context {i+1}: {chunk.chunk_text}")
        
        context = "\n\n".join(context_parts)
        
        # Create the prompt - T5 models work better with direct instructions
        prompt = f"""Question: {query}

Context: {context}

Based on the context above, provide a direct answer to the question. If the context doesn't contain the answer, say "I don't have enough information."

Answer:"""
        
        return prompt
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available LLM models"""
        return [
            {
                "name": "google/flan-t5-base",
                "description": "Reliable instruction-following (250M) - Proven performer",
                "size": "medium",
                "recommended": True
            },
            {
                "name": "facebook/bart-large-cnn",
                "description": "CNN-trained BART (400M) - Different training approach",
                "size": "large",
                "recommended": True
            },
            {
                "name": "google/flan-t5-large", 
                "description": "Scaling effects demo (780M) - Parameter scaling",
                "size": "large",
                "recommended": False
            },
            {
                "name": "google/t5-base",
                "description": "Original T5 base (220M) - Pre-instruction tuning",
                "size": "medium",
                "recommended": False
            },
            {
                "name": "google/flan-t5-small",
                "description": "Speed/efficiency trade-offs (80M) - Resource demo",
                "size": "small",
                "recommended": False
            }
        ]
    
    def clear_model(self) -> None:
        """Clear loaded model to free memory"""
        if self.model is not None:
            del self.model
            del self.tokenizer
            self.model = None
            self.tokenizer = None
            self.model_name = None
            
            # Clear GPU cache if using CUDA or MPS
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                torch.mps.empty_cache()
                
            logger.info("Model cleared from memory")

# Global instance
llm_service = LLMService()
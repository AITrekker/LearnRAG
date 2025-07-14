"""
LLM service for answer generation from retrieved chunks
"""
import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from threading import Lock

from app.models.responses import SearchResult

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
        if torch.cuda.is_available():
            device = "cuda"
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"Using GPU: {gpu_name}")
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
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Load model with appropriate device mapping
            if self.device == "cuda":
                self.model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,  # Use full precision to avoid Half/Float issues
                    device_map="auto"
                )
            else:
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
        max_length: int = 200
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
            
            # Create prompt
            prompt = self._create_prompt(query, chunks)
            
            # Run inference in thread pool to avoid blocking
            answer = await asyncio.get_event_loop().run_in_executor(
                None, self._generate_sync, prompt, max_length
            )
            
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
    
    def _generate_sync(self, prompt: str, max_length: int) -> str:
        """Synchronous answer generation (runs in thread pool)"""
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                max_length=512,
                truncation=True,
                return_tensors="pt"
            ).to(self.device)
            
            # Generate answer
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs.input_ids,
                    max_length=max_length,
                    num_return_sequences=1,
                    temperature=0.7,
                    do_sample=True,
                    pad_token_id=self.tokenizer.pad_token_id
                )
            
            # Decode answer
            answer = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Clean up answer
            answer = answer.strip()
            
            return answer
            
        except Exception as e:
            logger.error(f"Error in synchronous generation: {e}")
            return "Error generating answer. Please try again."
    
    def get_available_models(self) -> List[Dict[str, Any]]:
        """Get list of available LLM models"""
        return [
            {
                "name": "google/flan-t5-base",
                "description": "Google's T5 model fine-tuned for instructions (250M params)",
                "size": "small",
                "recommended": True
            },
            {
                "name": "google/flan-t5-large", 
                "description": "Larger T5 model for better quality (780M params)",
                "size": "large",
                "recommended": False
            },
            {
                "name": "microsoft/DialoGPT-medium",
                "description": "Microsoft's conversational model (345M params)",
                "size": "medium", 
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
            
            # Clear GPU cache if using CUDA
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                
            logger.info("Model cleared from memory")

# Global instance
llm_service = LLMService()
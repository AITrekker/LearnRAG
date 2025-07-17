"""
LLM Service - Answer Generation from Retrieved Context

Teaching Purpose: This service demonstrates the final step in RAG systems:

1. CONTEXT ASSEMBLY: Combine retrieved chunks into coherent context
2. PROMPT ENGINEERING: Format context and question for optimal LLM performance
3. ANSWER GENERATION: Use local LLM to synthesize human-readable responses
4. FALLBACK HANDLING: Graceful degradation when models fail

Core LLM Concepts Illustrated:
- Local model inference using HuggingFace transformers
- GPU/CPU device optimization for performance
- Prompt templates for different answer styles
- Token management and length constraints
- Model caching and memory management
"""
import asyncio
import logging
import time
from typing import List, Dict, Any, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoModelForQuestionAnswering, T5Tokenizer
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
from config import PROMPT_TEMPLATES

logger = logging.getLogger(__name__)

class LLMService:
    """
    Service for generating answers from retrieved chunks using local LLM
    
    Key concepts demonstrated:
    - Local model inference without external API dependencies
    - Device optimization (GPU/CPU) for performance
    - Prompt engineering for different answer styles
    - Token management and response length control
    - Model caching for efficient memory usage
    """
    
    def __init__(self):
        self.model = None
        self.tokenizer = None
        self.model_name = None
        self.device = None
        self.load_lock = Lock()
        
    def _detect_device(self) -> str:
        """
        Detect best available device for LLM inference - Hardware Optimization
        
        WHY DEVICE DETECTION?
        - GPU inference: 10-50x faster than CPU for transformer models
        - Apple Silicon (MPS): Optimized for M1/M2 chips
        - CPU fallback: Always available but slower
        - Memory constraints: GPU VRAM vs system RAM trade-offs
        
        DEVICE PRIORITY:
        1. NVIDIA CUDA: Best for large models, high throughput
        2. Apple MPS: Optimized for macOS with unified memory
        3. CPU: Universal fallback, slower but reliable
        """
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
    
    def _get_model_type(self, model_name: str) -> str:
        """Determine the model architecture type"""
        model_lower = model_name.lower()
        
        # Question Answering models - more comprehensive detection
        if any(keyword in model_lower for keyword in ["squad", "question-answering", "qa"]):
            return "qa"
        elif any(arch in model_lower for arch in ["roberta", "bert", "distilbert", "albert", "deberta", "electra"]) and "squad" not in model_lower:
            # BERT-family models without explicit SQuAD are usually base models, but we'll try QA first
            return "qa"
        # Seq2seq models (T5, FLAN-T5)
        elif "flan-t5" in model_lower or "t5" in model_lower:
            return "seq2seq"
        # Causal models (GPT-style)
        elif "dialogpt" in model_lower or "gpt" in model_lower:
            return "causal"
        else:
            # Default to qa for BERT-family models
            return "qa"

    def _load_model(self, model_name: str = "deepset/roberta-base-squad2") -> None:
        """Load the LLM model and tokenizer"""
        if self.model is not None and self.model_name == model_name:
            return  # Already loaded
            
        logger.info(f"Loading LLM model: {model_name}")
        start_time = time.time()
        
        try:
            # Detect device
            self.device = self._detect_device()
            
            # Determine model type
            model_type = self._get_model_type(model_name)
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Add pad token if missing (for DialoGPT)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model based on type
            if model_type == "seq2seq":
                model_class = AutoModelForSeq2SeqLM
            elif model_type == "causal":
                model_class = AutoModelForCausalLM
            elif model_type == "qa":
                model_class = AutoModelForQuestionAnswering
            else:
                model_class = AutoModelForSeq2SeqLM
                
            # Load model with appropriate device mapping
            if self.device == "cuda":
                self.model = model_class.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,  # Use full precision to avoid Half/Float issues
                    device_map="auto"
                )
            elif self.device == "mps":
                # Apple Silicon MPS optimization
                self.model = model_class.from_pretrained(
                    model_name,
                    torch_dtype=torch.float32,  # MPS works best with float32
                    low_cpu_mem_usage=True
                )
                self.model.to(self.device)
            else:
                # CPU fallback
                self.model = model_class.from_pretrained(model_name)
                self.model.to(self.device)
            
            self.model_name = model_name
            self.model_type = model_type
            
            load_time = time.time() - start_time
            logger.info(f"Model loaded in {load_time:.2f} seconds on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
    
    def _create_prompt(self, query: str, chunks: List[SearchResult], template_id: str = "factual") -> str:
        """Create a prompt for answer generation using configurable templates"""
        # Combine chunks into context
        context_parts = []
        for i, chunk in enumerate(chunks[:5]):  # Use top 5 chunks
            context_parts.append(f"[Source {i+1}] {chunk.chunk_text}")
        
        context = "\n\n".join(context_parts)
        
        # Get template from config, fallback to factual if invalid
        template_config = PROMPT_TEMPLATES.get(template_id, PROMPT_TEMPLATES["factual"])
        template = template_config["template"]
        
        # Format template with context and query
        prompt = template.format(context=context, query=query)
        
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
        prompt_template: str = "factual",
        max_length: int = 200,
        temperature: float = 0.7,
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
            prompt = self._create_prompt(query, top_chunks, prompt_template)
            
            # Debug: Log actual prompt and chunks being sent to LLM
            logger.info(f"=== LLM DEBUG START ===")
            logger.info(f"Query: {query}")
            logger.info(f"Template: {prompt_template}")  
            logger.info(f"Number of chunks: {len(top_chunks)}")
            for i, chunk in enumerate(top_chunks):
                logger.info(f"Chunk {i+1}: similarity={chunk.similarity:.3f}, text={chunk.chunk_text[:150]}...")
            logger.info(f"Full prompt being sent to LLM:\n{prompt}")
            logger.info(f"=== LLM DEBUG END ===")
            
            # Run inference in thread pool to avoid blocking
            answer = await asyncio.get_event_loop().run_in_executor(
                None, self._generate_sync, prompt, max_length, temperature, top_p, repetition_penalty
            )
            
            logger.info(f"=== LLM ANSWER DEBUG: {answer} ===")
            
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
            # Handle different model types
            if self.model_type == "qa":
                # For QA models like RoBERTa-SQuAD
                return self._generate_qa_answer(prompt, max_length)
            elif self.model_type == "causal":
                # For causal models like DialoGPT
                return self._generate_causal_answer(prompt, max_length, temperature, top_p, repetition_penalty)
            else:
                # For seq2seq models like T5/FLAN-T5
                return self._generate_seq2seq_answer(prompt, max_length, temperature, top_p, repetition_penalty)
                
        except Exception as e:
            logger.error(f"Error in synchronous generation: {e}")
            return "Error generating answer. Please try again."
    
    def _generate_qa_answer(self, prompt: str, max_length: int) -> str:
        """Generate answer using QA model (RoBERTa-SQuAD)"""
        # Extract question and context from prompt
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
        
        if not question or not context:
            return "I don't have enough information to answer this question."
        
        # Use QA pipeline
        inputs = self.tokenizer(question, context, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Get answer span
        start_scores = outputs.start_logits
        end_scores = outputs.end_logits
        
        start_idx = torch.argmax(start_scores)
        end_idx = torch.argmax(end_scores)
        
        if end_idx < start_idx:
            return "I don't have enough information to answer this question."
        
        # Extract answer
        answer_tokens = inputs['input_ids'][0][start_idx:end_idx+1]
        answer = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)
        
        return answer.strip() if answer.strip() else "I don't have enough information to answer this question."
    
    def _generate_causal_answer(self, prompt: str, max_length: int, temperature: float, top_p: float, repetition_penalty: float) -> str:
        """Generate answer using causal model (DialoGPT)"""
        # Tokenize input
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)
        
        # Generate answer
        with torch.no_grad():
            outputs = self.model.generate(
                inputs.input_ids,
                max_length=inputs.input_ids.shape[1] + max_length,
                temperature=temperature,
                do_sample=True,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1
            )
        
        # Extract only the generated part (after the prompt)
        generated_text = self.tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        return generated_text.strip() if generated_text.strip() else "I don't have enough information to answer this question."
    
    def _generate_seq2seq_answer(self, prompt: str, max_length: int, temperature: float, top_p: float, repetition_penalty: float) -> str:
        """Generate answer using seq2seq model (T5/FLAN-T5)"""
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
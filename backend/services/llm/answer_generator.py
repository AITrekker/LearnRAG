"""
Answer Generator - Core LLM Inference Logic

Teaching Purpose: This module demonstrates LLM inference patterns:

1. MULTI-ARCHITECTURE SUPPORT: QA, Seq2Seq, and Causal model handling
2. ASYNC INFERENCE: Non-blocking generation with thread pools
3. ERROR HANDLING: Graceful fallbacks and error recovery
4. INFERENCE OPTIMIZATION: Model-specific generation strategies

Core Concepts Illustrated:
- Architecture-specific inference methods
- Async/await patterns for blocking operations
- Error handling and fallback strategies
- Generation parameter optimization
"""

import asyncio
import logging
import time
import torch
from typing import List, Dict, Any
from models import SearchResult

logger = logging.getLogger(__name__)


class AnswerGenerator:
    """
    Handles core LLM inference with support for multiple model architectures
    
    Key responsibilities:
    - Execute inference for different model types (QA, Seq2Seq, Causal)
    - Handle async operations and threading
    - Manage generation parameters and optimization
    - Provide error handling and fallback strategies
    """
    
    def __init__(self, model_manager, prompt_engineer):
        """Initialize answer generator with dependencies"""
        self.model_manager = model_manager
        self.prompt_engineer = prompt_engineer

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
        Generate an answer from retrieved chunks using LLM - Main Generation Logic
        
        GENERATION PIPELINE:
        1. Validate inputs and handle edge cases
        2. Load model and get architecture type
        3. Create and optimize prompt
        4. Execute inference in thread pool (non-blocking)
        5. Calculate confidence and return results
        
        WHY ASYNC INFERENCE?
        - LLM inference can take 1-10+ seconds
        - Async prevents blocking other requests
        - Thread pool isolates blocking operations
        - Enables concurrent request handling
        """
        start_time = time.time()
        
        try:
            # Get model and tokenizer (thread-safe)
            model, tokenizer, model_type, device = self.model_manager.get_model_and_tokenizer(model_name)
            
            # Handle edge case: no chunks
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
            
            # Optimize context length for model limits
            optimized_chunks = self.prompt_engineer.optimize_context_length(top_chunks)
            
            # Create prompt
            prompt = self.prompt_engineer.create_prompt(query, optimized_chunks, prompt_template)
            
            # Debug logging for development and troubleshooting
            logger.info(f"=== LLM DEBUG START ===")
            logger.info(f"Query: {query}")
            logger.info(f"Template: {prompt_template}")
            logger.info(f"Model: {model_name} (type: {model_type})")
            logger.info(f"Number of chunks: {len(optimized_chunks)}")
            for i, chunk in enumerate(optimized_chunks):
                logger.info(f"Chunk {i+1}: similarity={chunk.similarity:.3f}, text={chunk.chunk_text[:150]}...")
            logger.info(f"Full prompt being sent to LLM:\n{prompt}")
            logger.info(f"=== LLM DEBUG END ===")
            
            # Run inference in thread pool to avoid blocking
            answer = await asyncio.get_event_loop().run_in_executor(
                None, self._generate_sync, prompt, model, tokenizer, model_type, device,
                max_length, temperature, top_p, repetition_penalty
            )
            
            logger.info(f"=== LLM ANSWER DEBUG: {answer} ===")
            
            # Calculate confidence
            confidence = self.prompt_engineer.calculate_confidence(answer, chunks)
            
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

    def _generate_sync(self, prompt: str, model, tokenizer, model_type: str, device: str,
                      max_length: int, temperature: float = 0.3, top_p: float = 0.9, 
                      repetition_penalty: float = 1.1) -> str:
        """
        Synchronous answer generation (runs in thread pool) - Core Inference
        
        WHY SYNCHRONOUS WRAPPER?
        - PyTorch operations are synchronous and blocking
        - Thread pool execution prevents blocking main event loop
        - Enables concurrent request handling
        - Isolates GPU operations from async code
        """
        try:
            # Handle different model types with appropriate inference methods
            if model_type == "qa":
                # For QA models like RoBERTa-SQuAD
                return self._generate_qa_answer(prompt, model, tokenizer, device, max_length)
            elif model_type == "causal":
                # For causal models like DialoGPT
                return self._generate_causal_answer(prompt, model, tokenizer, device, 
                                                 max_length, temperature, top_p, repetition_penalty)
            else:
                # For seq2seq models like T5/FLAN-T5
                return self._generate_seq2seq_answer(prompt, model, tokenizer, device,
                                                   max_length, temperature, top_p, repetition_penalty)
                
        except Exception as e:
            logger.error(f"Error in synchronous generation: {e}")
            return "Error generating answer. Please try again."

    def _generate_qa_answer(self, prompt: str, model, tokenizer, device: str, max_length: int) -> str:
        """
        Generate answer using QA model (RoBERTa-SQuAD) - Extractive QA
        
        QA MODEL STRATEGY:
        - Extract question and context from prompt
        - Use model to predict answer span in context
        - Return extracted text as answer
        
        WHY EXTRACTIVE QA?
        - High accuracy for factual questions
        - Answers are grounded in source text
        - Good for when exact quotes are needed
        - Fast inference compared to generative models
        """
        # Extract question and context from prompt
        question, context = self.prompt_engineer.extract_qa_context(prompt)
        
        if not question or not context:
            return "I don't have enough information to answer this question."
        
        # Use QA pipeline
        inputs = tokenizer(question, context, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs)
        
        # Get answer span
        start_scores = outputs.start_logits
        end_scores = outputs.end_logits
        
        start_idx = torch.argmax(start_scores)
        end_idx = torch.argmax(end_scores)
        
        if end_idx < start_idx:
            return "I don't have enough information to answer this question."
        
        # Extract answer
        answer_tokens = inputs['input_ids'][0][start_idx:end_idx+1]
        answer = tokenizer.decode(answer_tokens, skip_special_tokens=True)
        
        return answer.strip() if answer.strip() else "I don't have enough information to answer this question."

    def _generate_causal_answer(self, prompt: str, model, tokenizer, device: str,
                              max_length: int, temperature: float, top_p: float, 
                              repetition_penalty: float) -> str:
        """
        Generate answer using causal model (DialoGPT) - Autoregressive Generation
        
        CAUSAL MODEL STRATEGY:
        - Continue the prompt with autoregressive generation
        - Use sampling for more natural responses
        - Extract only the generated portion
        
        WHY CAUSAL MODELS?
        - More conversational and natural responses
        - Can generate creative answers beyond source text
        - Good for dialogue and chat-like interactions
        - Flexible response length and style
        """
        # Tokenize input
        inputs = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(device)
        
        # Generate answer
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=inputs.input_ids.shape[1] + max_length,
                temperature=temperature,
                do_sample=True,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
                num_return_sequences=1
            )
        
        # Extract only the generated part (after the prompt)
        generated_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        return generated_text.strip() if generated_text.strip() else "I don't have enough information to answer this question."

    def _generate_seq2seq_answer(self, prompt: str, model, tokenizer, device: str,
                               max_length: int, temperature: float, top_p: float, 
                               repetition_penalty: float) -> str:
        """
        Generate answer using seq2seq model (T5/FLAN-T5) - Instruction Following
        
        SEQ2SEQ STRATEGY:
        - Encode prompt as input sequence
        - Decode answer as output sequence
        - Use instruction-following capabilities
        
        WHY SEQ2SEQ MODELS?
        - Designed for instruction following and QA
        - Good balance of accuracy and creativity
        - Handles complex multi-step reasoning
        - Optimized for text-to-text tasks
        """
        # Tokenize input
        inputs = tokenizer(
            prompt,
            max_length=512,
            truncation=True,
            return_tensors="pt"
        ).to(device)
        
        # Generate answer with configurable parameters
        with torch.no_grad():
            outputs = model.generate(
                inputs.input_ids,
                max_length=max_length,
                num_return_sequences=1,
                temperature=temperature,
                do_sample=True,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
                pad_token_id=tokenizer.pad_token_id,
                early_stopping=True
            )
        
        # Decode answer - T5 models return only the generated part
        answer = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Clean up answer and remove any artifacts
        answer = answer.strip()
        
        # Remove any remaining prompt artifacts
        if "Answer:" in answer:
            answer = answer.split("Answer:")[-1].strip()
        
        # Ensure we have a meaningful answer
        if not answer or len(answer) < 3:
            answer = "I don't have enough information to answer this question."
        
        return answer
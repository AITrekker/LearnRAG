"""
Model Manager - LLM Model Loading and Device Management

Teaching Purpose: This module demonstrates LLM infrastructure patterns:

1. DEVICE DETECTION: Automatic GPU/CPU optimization
2. MODEL LOADING: Architecture-aware loading with error handling
3. MEMORY MANAGEMENT: Efficient model storage and cleanup
4. THREADING: Thread-safe model loading with locks

Core Concepts Illustrated:
- Hardware optimization for different devices (CUDA, MPS, CPU)
- Model architecture detection (QA, Seq2Seq, Causal)
- Memory management for large models
- Thread safety for concurrent requests
"""

import logging
import time
import torch
from typing import Dict, Any, List, Optional
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForCausalLM, AutoModelForQuestionAnswering
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

from config import AVAILABLE_LLM_MODELS

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Manages LLM model loading, device optimization, and memory management
    
    Key responsibilities:
    - Detect optimal device (CUDA, MPS, CPU) for inference
    - Load models with appropriate architecture classes
    - Manage model lifecycle and memory cleanup
    - Provide thread-safe model access
    """
    
    def __init__(self):
        """Initialize model manager with device detection"""
        self.model = None
        self.tokenizer = None
        self.model_name = None
        self.model_type = None
        self.device = None
        self.load_lock = Lock()
        
    def detect_device(self) -> str:
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
    
    def get_model_type(self, model_name: str) -> str:
        """
        Determine the model architecture type - Architecture Detection
        
        WHY ARCHITECTURE DETECTION?
        - Different model types require different loading classes
        - QA models: Optimized for extractive question answering
        - Seq2Seq models: Good for generative tasks like T5/FLAN-T5
        - Causal models: For autoregressive generation like GPT
        
        DETECTION STRATEGY:
        - Use model name patterns to infer architecture
        - Default to QA for BERT-family models
        - Handle special cases like FLAN-T5, DialoGPT
        """
        model_lower = model_name.lower()
        
        # Question Answering models - comprehensive detection
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

    def load_model(self, model_name: str = "deepset/roberta-base-squad2") -> None:
        """
        Load the LLM model and tokenizer - Core Model Loading
        
        LOADING STRATEGY:
        1. Check if model is already loaded (avoid redundant loading)
        2. Detect optimal device for inference
        3. Determine model architecture type
        4. Load tokenizer with proper configuration
        5. Load model with device-specific optimizations
        
        DEVICE OPTIMIZATIONS:
        - CUDA: Use device_map="auto" for multi-GPU
        - MPS: Apple Silicon optimizations with float32
        - CPU: Standard loading with explicit device placement
        """
        if self.model is not None and self.model_name == model_name:
            return  # Already loaded
            
        logger.info(f"Loading LLM model: {model_name}")
        start_time = time.time()
        
        try:
            # Detect device
            self.device = self.detect_device()
            
            # Determine model type
            self.model_type = self.get_model_type(model_name)
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Add pad token if missing (for DialoGPT)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # Load model based on type
            if self.model_type == "seq2seq":
                model_class = AutoModelForSeq2SeqLM
            elif self.model_type == "causal":
                model_class = AutoModelForCausalLM
            elif self.model_type == "qa":
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
            
            load_time = time.time() - start_time
            logger.info(f"Model loaded in {load_time:.2f} seconds on {self.device}")
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise

    def get_model_and_tokenizer(self, model_name: str):
        """
        Get model and tokenizer with thread-safe loading - Thread-Safe Access
        
        WHY THREAD SAFETY?
        - Multiple requests may try to load models simultaneously
        - Prevents race conditions during model loading
        - Ensures model consistency across concurrent requests
        """
        with self.load_lock:
            self.load_model(model_name)
            return self.model, self.tokenizer, self.model_type, self.device

    def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Get list of available LLM models - Configuration Interface
        
        Returns model metadata from configuration including:
        - Model descriptions and use cases
        - Size and performance characteristics
        - Recommendations for different scenarios
        """
        return AVAILABLE_LLM_MODELS
    
    def clear_model(self) -> None:
        """
        Clear loaded model to free memory - Memory Management
        
        WHY EXPLICIT CLEANUP?
        - LLM models can be very large (1GB+ each)
        - Free GPU memory for other models or larger batch sizes
        - Prevent memory leaks in long-running services
        - Enable dynamic model switching
        """
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
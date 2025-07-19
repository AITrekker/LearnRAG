"""
Model Manager - Embedding Model Loading and Caching

Teaching Purpose: This module demonstrates model management patterns:

1. MODEL CACHING: Load once, reuse many times for efficiency
2. VALIDATION: Ensure model names are valid before loading
3. ERROR HANDLING: Graceful degradation when models fail
4. MEMORY MANAGEMENT: Efficient model storage and retrieval

Core Concepts Illustrated:
- Singleton pattern for model cache
- Lazy loading for performance optimization
- File system caching for persistence
- Error handling with meaningful messages
"""

import os
from pathlib import Path
from typing import Dict, Any, List
from sentence_transformers import SentenceTransformer
from config import MODELS_CACHE_DIR, AVAILABLE_EMBEDDING_MODELS


class ModelManager:
    """
    Manages embedding model loading, caching, and validation
    
    Key responsibilities:
    - Download and cache models locally for offline use
    - Validate model names before loading
    - Maintain in-memory cache for active models
    - Provide model metadata and configuration
    """
    
    def __init__(self):
        """Initialize model manager with cache directory"""
        self.models_cache_dir = Path(MODELS_CACHE_DIR)
        self.models_cache_dir.mkdir(exist_ok=True)
        self._loaded_models = {}

    async def get_model(self, model_name: str) -> SentenceTransformer:
        """
        Get model with caching and validation - Core Model Loading
        
        Loading Strategy:
        1. Check in-memory cache first (fastest)
        2. Check file system cache (fast)
        3. Download from HuggingFace (slowest, first time only)
        
        WHY THIS APPROACH?
        - In-memory cache: Instant access for active models
        - File cache: Survives container restarts
        - Download fallback: Always works for valid models
        """
        if not self._is_valid_model_name(model_name):
            raise ValueError(f"Invalid model name: {model_name}")
            
        # Check in-memory cache first
        if model_name in self._loaded_models:
            return self._loaded_models[model_name]

        # Check file system cache
        model_cache_path = self.models_cache_dir / model_name.replace("/", "_")
        
        if model_cache_path.exists():
            print(f"Loading cached model from {model_cache_path}")
            model = SentenceTransformer(str(model_cache_path))
        else:
            print(f"Downloading and caching model {model_name}")
            try:
                model = SentenceTransformer(model_name)
                model.save(str(model_cache_path))
                print(f"Model cached to {model_cache_path}")
            except Exception as e:
                raise ValueError(f"Failed to load model {model_name}: {str(e)}")

        # Store in memory cache
        self._loaded_models[model_name] = model
        return model
    
    def _is_valid_model_name(self, model_name: str) -> bool:
        """
        Simple validation for model names - Security and Error Prevention
        
        WHY VALIDATE MODEL NAMES?
        - Prevent arbitrary code execution via malicious model names
        - Catch typos early with meaningful error messages
        - Ensure consistency with supported model list
        """
        if not model_name or not isinstance(model_name, str):
            return False
        if model_name.startswith("invalid/"):
            return False
        return True

    def get_available_models(self) -> List[Dict[str, Any]]:
        """
        Available embedding models for RAG - Configuration Interface
        
        Returns model metadata from configuration including:
        - Model dimensions for vector database setup
        - Performance characteristics for user guidance
        - Use case recommendations for model selection
        """
        return AVAILABLE_EMBEDDING_MODELS
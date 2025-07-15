"""
Centralized logging configuration for LearnRAG

Teaching Purpose: Demonstrates proper logging setup for production RAG systems:
- Structured logging for debugging and monitoring
- Different log levels for development vs production
- Context-aware logging for tracing RAG operations
"""

import logging
import sys
from datetime import datetime
from typing import Any, Dict


class RAGFormatter(logging.Formatter):
    """Custom formatter for RAG-specific logging with context"""
    
    def format(self, record):
        # Add timestamp and operation context
        record.timestamp = datetime.utcnow().isoformat()
        
        # Color coding for console output
        color_codes = {
            'DEBUG': '\033[36m',    # Cyan
            'INFO': '\033[32m',     # Green  
            'WARNING': '\033[33m',  # Yellow
            'ERROR': '\033[31m',    # Red
            'CRITICAL': '\033[35m'  # Magenta
        }
        reset_code = '\033[0m'
        
        color = color_codes.get(record.levelname, '')
        record.colored_level = f"{color}{record.levelname}{reset_code}"
        
        return super().format(record)


def setup_logging(level: str = "INFO", structured: bool = False) -> logging.Logger:
    """
    Setup centralized logging for LearnRAG application
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        structured: Whether to use structured (JSON) logging
    """
    
    # Create logger
    logger = logging.getLogger("learnrag")
    logger.setLevel(getattr(logging, level.upper()))
    
    # Avoid duplicate handlers
    if logger.handlers:
        return logger
    
    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    
    if structured:
        # JSON format for production
        formatter = logging.Formatter(
            '{"timestamp": "%(asctime)s", "level": "%(levelname)s", '
            '"module": "%(name)s", "message": "%(message)s"}'
        )
    else:
        # Human-readable format for development
        formatter = RAGFormatter(
            '%(timestamp)s | %(colored_level)-8s | %(name)-20s | %(message)s'
        )
    
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    return logger


def log_rag_operation(
    operation: str, 
    context: Dict[str, Any], 
    level: str = "INFO"
) -> None:
    """
    Log RAG-specific operations with structured context
    
    Teaching Purpose: Shows how to add context to logs for debugging RAG pipelines
    """
    logger = logging.getLogger("learnrag.rag")
    
    message = f"RAG Operation: {operation}"
    if context:
        context_str = " | ".join([f"{k}={v}" for k, v in context.items()])
        message += f" | Context: {context_str}"
    
    getattr(logger, level.lower())(message)


# Setup default logger
default_logger = setup_logging()
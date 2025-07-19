"""
Database Configuration - Async SQLAlchemy with Connection Pooling

This module demonstrates production-ready database patterns:

1. ASYNC OPERATIONS: Non-blocking database operations using asyncpg
2. CONNECTION POOLING: Efficient connection reuse and management
3. SESSION MANAGEMENT: Proper session lifecycle and cleanup
4. TABLE CREATION: Automatic schema initialization
5. DEPENDENCY INJECTION: FastAPI dependency pattern for database access

Core Database Concepts Illustrated:
- Async SQLAlchemy for high-performance database operations
- Connection pooling to handle concurrent requests efficiently
- Session management with proper resource cleanup
- Dependency injection for clean separation of concerns
- Production-ready configuration with monitoring and health checks
"""

import os
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase
from config import DATABASE_URL, DATABASE_POOL_SIZE, DATABASE_MAX_OVERFLOW, DATABASE_POOL_RECYCLE

# Convert to async URL
ASYNC_DATABASE_URL = DATABASE_URL
if ASYNC_DATABASE_URL.startswith("postgresql://"):
    ASYNC_DATABASE_URL = ASYNC_DATABASE_URL.replace("postgresql://", "postgresql+asyncpg://")

engine = create_async_engine(
    ASYNC_DATABASE_URL, 
    echo=False,
    pool_size=DATABASE_POOL_SIZE,
    max_overflow=DATABASE_MAX_OVERFLOW,
    pool_pre_ping=True,  # Verify connections before use
    pool_recycle=DATABASE_POOL_RECYCLE,   # Recycle connections every hour
)
AsyncSessionLocal = async_sessionmaker(
    engine, 
    class_=AsyncSession, 
    expire_on_commit=False,
    autoflush=True,
    autocommit=False
)


class Base(DeclarativeBase):
    """
    Base class for all database models - SQLAlchemy Declarative Base
    
    WHY DECLARATIVE BASE?
    - Provides common functionality for all model classes
    - Enables automatic table creation and schema management
    - Supports relationship mapping and foreign key constraints
    - Facilitates migration and schema evolution
    """
    pass


async def init_db():
    """
    Initialize database tables - Schema Creation
    
    WHY AUTOMATIC INITIALIZATION?
    - Ensures database schema matches model definitions
    - Simplifies deployment and development setup
    - Creates tables, indexes, and constraints automatically
    - Handles schema changes through model evolution
    
    INITIALIZATION PROCESS:
    1. Import all models to register them with SQLAlchemy
    2. Create async database connection
    3. Run table creation synchronously within async context
    4. Confirm successful table creation
    """
    # Import all models to ensure they're registered with SQLAlchemy metadata
    from models import (
        Tenant, File, Embedding, DocumentSummary, SectionSummary, 
        TenantEmbeddingSettings, RagSession
    )
    
    # Verify models are loaded
    print(f"📋 Registering models: {[cls.__name__ for cls in Base.registry._class_registry.values() if hasattr(cls, '__tablename__')]}")
    
    async with engine.begin() as conn:
        # Debug: Show what tables will be created
        print("🔍 Tables to be created:")
        for table_name, table in Base.metadata.tables.items():
            columns = [f"{col.name}({col.type})" for col in table.columns]
            print(f"   {table_name}: {columns}")
        
        # Create all tables
        await conn.run_sync(Base.metadata.create_all)
        print("📊 Database tables created successfully")


async def get_db():
    """
    Database session dependency - Connection Management
    
    WHY DEPENDENCY INJECTION?
    - Provides clean separation between business logic and data access
    - Ensures proper session lifecycle management
    - Enables easy testing with mock databases
    - Supports transaction management and rollback
    
    SESSION LIFECYCLE:
    1. Create new session from connection pool
    2. Yield session to request handler
    3. Automatically close session after request
    4. Handle exceptions and cleanup resources
    
    PRODUCTION BENEFITS:
    - Connection pooling reduces overhead
    - Proper cleanup prevents connection leaks
    - Exception handling ensures system stability
    - Async operations enable high concurrency
    """
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
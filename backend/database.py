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
    pass


async def init_db():
    # Import all models to ensure they're registered
    import models
    
    async with engine.begin() as conn:
        # Create all tables
        await conn.run_sync(Base.metadata.create_all)
        print("📊 Database tables created successfully")


async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
        finally:
            await session.close()
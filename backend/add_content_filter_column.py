"""
Add enable_content_filter column to tenant_embedding_settings table
"""
import asyncio
import asyncpg
from sqlalchemy.ext.asyncio import create_async_engine

async def add_content_filter_column():
    """Add the enable_content_filter column to existing database"""
    
    # Connect to database
    engine = create_async_engine("postgresql+asyncpg://postgres:postgres@localhost:5432/learnrag")
    
    async with engine.begin() as conn:
        # Check if column exists
        result = await conn.execute(
            """
            SELECT column_name 
            FROM information_schema.columns 
            WHERE table_name = 'tenant_embedding_settings' 
            AND column_name = 'enable_content_filter'
            """
        )
        
        if not result.fetchone():
            # Add the column with default value
            await conn.execute(
                """
                ALTER TABLE tenant_embedding_settings 
                ADD COLUMN enable_content_filter BOOLEAN DEFAULT TRUE
                """
            )
            print("✅ Added enable_content_filter column")
        else:
            print("✅ Column already exists")
    
    await engine.dispose()

if __name__ == "__main__":
    asyncio.run(add_content_filter_column())
"""
File Processor - File-Level Embedding Generation and Delta Sync

Teaching Purpose: This module demonstrates file processing patterns:

1. DELTA SYNC: Only re-embed when files or settings change
2. BATCH PROCESSING: Efficient database operations and memory usage
3. HIERARCHICAL CONTEXT: Integration with summary system
4. ERROR HANDLING: Graceful degradation for individual files

Core Processing Concepts Illustrated:
- Delta sync for efficiency and cost savings
- Batch database operations for performance
- Integration patterns between services
- Metrics collection for monitoring and optimization
"""

import time
from pathlib import Path
from typing import List, Dict, Any, Optional
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, delete

from models import File, Embedding, DocumentSummary, SectionSummary, Tenant
from services.file_processor import FileProcessor as BaseFileProcessor
from services.metrics_service import MetricsService
from services.summary_service import SummaryService


class FileProcessor:
    """
    Handles file-level embedding generation with delta sync and hierarchical context
    
    Key responsibilities:
    - Delta sync: Only re-embed when needed
    - Batch processing: Efficient database operations
    - Summary integration: Hierarchical context generation
    - Metrics collection: Performance monitoring
    """
    
    def __init__(self, model_manager, chunking_strategies):
        """Initialize file processor with dependencies"""
        self.model_manager = model_manager
        self.chunking_strategies = chunking_strategies
        self.base_file_processor = BaseFileProcessor()
        self.summary_service = SummaryService()
        self.metrics_service = MetricsService()

    async def generate_embeddings_for_files(
        self, 
        files: List[File], 
        embedding_model: str, 
        chunking_strategy: str, 
        db: AsyncSession,
        chunk_size: int = 512,
        chunk_overlap: int = 50,
        tenant_name: str = "Unknown"
    ):
        """
        Generate vector embeddings for multiple files - Main Processing Loop
        
        DELTA SYNC STRATEGY:
        - Check if embeddings already exist for file + model + strategy combination
        - Skip processing if embeddings are current
        - Delete old embeddings before generating new ones
        
        PERFORMANCE OPTIMIZATIONS:
        - Load model once for all files
        - Batch database commits for efficiency
        - Collect metrics for monitoring
        """
        # Start metrics session
        session_id = self.metrics_service.start_session(
            tenant_name, embedding_model, chunking_strategy, 
            chunk_size if chunking_strategy == "fixed_size" else None, 
            chunk_overlap, len(files)
        )
        
        # Get model once for all files
        model = await self.model_manager.get_model(embedding_model)
        
        try:
            for file in files:
                await self._generate_embeddings_for_file(
                    file, model, embedding_model, chunking_strategy, 
                    chunk_size, chunk_overlap, db
                )
        finally:
            self.metrics_service.end_session()

    async def _generate_embeddings_for_file(
        self,
        file: File,
        model,
        embedding_model: str,
        chunking_strategy: str,
        chunk_size: int,
        chunk_overlap: int,
        db: AsyncSession
    ):
        """
        Generate embeddings for a single file with delta sync - Core Processing
        
        DELTA SYNC IMPLEMENTATION:
        1. Check for existing embeddings with same parameters
        2. Skip if already processed (delta sync optimization)
        3. Delete old embeddings if regenerating
        4. Process file and generate new embeddings
        """
        # Check for existing embeddings (delta sync)
        existing_embeddings = await db.execute(
            select(Embedding).where(
                Embedding.file_id == file.id,
                Embedding.embedding_model == embedding_model,
                Embedding.chunking_strategy == chunking_strategy
            )
        )
        existing = existing_embeddings.scalars().all()

        if existing:
            print(f"Embeddings already exist for file {file.filename} with model {embedding_model}")
            return

        # Delete old embeddings if they exist (cleanup)
        if existing:
            await db.execute(
                delete(Embedding).where(
                    Embedding.file_id == file.id,
                    Embedding.embedding_model == embedding_model,
                    Embedding.chunking_strategy == chunking_strategy
                )
            )

        file_start_time = time.time()
        
        # Get tenant information
        tenant_result = await db.execute(select(Tenant).where(Tenant.id == file.tenant_id))
        tenant = tenant_result.scalar_one()
        
        # Extract text content
        file_path = Path("/app/internal_files") / tenant.slug / file.file_path
        text_content = await self.base_file_processor.extract_text(file_path, file.content_type)

        if not text_content.strip():
            print(f"No text content extracted from {file.filename}")
            return

        # Generate hierarchical summaries first
        document_summary, section_summaries = await self._generate_and_store_summaries(
            file, text_content, db
        )

        # Chunk the text
        chunks = await self.chunking_strategies.chunk_text(
            text_content, chunking_strategy, chunk_size, chunk_overlap
        )
        
        # Collect metrics
        file_size = file_path.stat().st_size if file_path.exists() else 0
        tokens_processed = sum(len(chunk.split()) for chunk in chunks)
        chunk_distribution = [len(chunk.split()) for chunk in chunks]

        # Process chunks in batches
        batch_size = 32
        for i, chunk in enumerate(chunks):
            try:
                # Determine which section this chunk belongs to
                chunk_section = self._find_chunk_section(chunk, text_content, section_summaries)
                
                # Generate hierarchical context for this chunk
                chunk_context = None
                if document_summary and chunk_section:
                    chunk_context = await self.summary_service.create_chunk_context(
                        chunk, chunk_section['summary'], document_summary.summary_text
                    )
                
                # Generate embedding for chunk (with or without context)
                embedding_text = chunk_context if chunk_context else chunk
                embedding_vector = model.encode([embedding_text], convert_to_tensor=False)[0]
                
                # Create embedding record
                embedding = Embedding(
                    file_id=file.id,
                    chunk_index=i,
                    chunk_text=chunk,
                    chunk_context=chunk_context,
                    section_id=chunk_section['id'] if chunk_section else None,
                    embedding_model=embedding_model,
                    chunking_strategy=chunking_strategy,
                    embedding=embedding_vector.tolist(),
                    chunk_metadata={
                        "chunk_length": len(chunk),
                        "chunk_words": len(chunk.split()),
                        "has_context": chunk_context is not None
                    }
                )
                db.add(embedding)

                # Batch commit for performance
                if (i + 1) % batch_size == 0:
                    await db.commit()

            except Exception as e:
                print(f"Error generating embedding for chunk {i} of file {file.filename}: {e}")
                continue

        await db.commit()
        
        # Log metrics
        file_processing_time = time.time() - file_start_time
        self.metrics_service.log_file_processed(
            file.id, file.filename, file_size,
            len(chunks), tokens_processed, file_processing_time, chunk_distribution
        )
        
        print(f"Generated {len(chunks)} embeddings for {file.filename}")

    async def _generate_and_store_summaries(self, file: File, text_content: str, db: AsyncSession):
        """
        Generate and store hierarchical summaries - Summary Integration
        
        WHY GENERATE SUMMARIES DURING EMBEDDING?
        - Ensures summaries are available when embeddings are created
        - Enables immediate hierarchical context generation
        - Maintains consistency between embeddings and summaries
        - Optimizes single-pass processing for efficiency
        """
        try:
            # Check if summaries already exist
            existing_summary = await db.execute(
                select(DocumentSummary).where(DocumentSummary.file_id == file.id)
            )
            existing = existing_summary.scalar_one_or_none()
            
            if existing:
                print(f"Document summaries already exist for {file.filename}")
                # Load existing section summaries
                section_result = await db.execute(
                    select(SectionSummary).where(SectionSummary.document_summary_id == existing.id)
                )
                sections = section_result.scalars().all()
                return existing, [{'id': s.id, 'summary': s.summary_text, 'start': s.start_position, 'end': s.end_position} for s in sections]
            
            print(f"📝 Generating hierarchical summaries for {file.filename}")
            
            # Generate document summary
            doc_summary_text = await self.summary_service.generate_document_summary(
                text_content, file.filename
            )
            
            # Generate section summaries
            section_summaries = await self.summary_service.generate_section_summaries(
                text_content, file.filename
            )
            
            # Store document summary
            document_summary = DocumentSummary(
                file_id=file.id,
                summary_text=doc_summary_text,
                summary_type="document"
            )
            db.add(document_summary)
            await db.flush()  # Get the ID
            
            # Store section summaries
            stored_sections = []
            for section in section_summaries:
                section_summary = SectionSummary(
                    document_summary_id=document_summary.id,
                    section_number=section['section_number'],
                    title=section['title'],
                    summary_text=section['summary'],
                    start_position=section['start_pos'],
                    end_position=section['end_pos'],
                    content_length=section['content_length']
                )
                db.add(section_summary)
                await db.flush()  # Get the ID
                
                stored_sections.append({
                    'id': section_summary.id,
                    'summary': section['summary'],
                    'start': section['start_pos'],
                    'end': section['end_pos']
                })
            
            await db.commit()
            print(f"✅ Generated document summary + {len(stored_sections)} section summaries")
            
            return document_summary, stored_sections
            
        except Exception as e:
            print(f"⚠️ Error generating summaries for {file.filename}: {e}")
            await db.rollback()
            return None, []

    def _find_chunk_section(self, chunk_text: str, full_text: str, section_summaries: List[Dict]) -> Optional[Dict]:
        """
        Map chunk to its containing section - Hierarchical Chunk Mapping
        
        WHY MAP CHUNKS TO SECTIONS?
        - Enables section-aware chunk retrieval
        - Provides hierarchical context for better embeddings
        - Improves query routing from general to specific
        - Maintains document structure in vector space
        
        MAPPING STRATEGY:
        - Find chunk position in original document
        - Match position to section boundaries
        - Return section info for context generation
        """
        if not section_summaries:
            return None
            
        try:
            # Find the chunk's position in the full text
            chunk_start = full_text.find(chunk_text[:100])  # Use first part of chunk for matching
            if chunk_start == -1:
                return None
            
            # Find which section contains this position
            for section in section_summaries:
                if section['start'] <= chunk_start <= section['end']:
                    return section
                    
            return None
            
        except Exception as e:
            print(f"Error mapping chunk to section: {e}")
            return None
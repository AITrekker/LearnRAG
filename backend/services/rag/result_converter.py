"""
Result Converter - Standardized Result Format Conversion

This module demonstrates clean data transformation patterns:

WHY RESULT CONVERSION?
- Different search methods return different result formats
- API consistency requires standardized response formats
- Enables seamless switching between search techniques
- Simplifies result processing and display logic

CONVERSION PATTERNS:
- SearchResult ↔ HierarchicalSearchResult conversion
- Format standardization across different search methods
- Metadata preservation during conversion
- Fallback handling for missing data fields
"""

from models import SearchResult, HierarchicalSearchResult


class ResultConverter:
    """
    Convert between different result formats for API consistency
    
    This demonstrates clean data transformation patterns in RAG systems
    """
    
    def to_hierarchical_result(self, search_result: SearchResult) -> HierarchicalSearchResult:
        """
        Convert regular SearchResult to HierarchicalSearchResult
        
        WHY CONVERT RESULT FORMATS?
        - Enables fallback from hierarchical to regular search
        - Maintains consistent API response format
        - Allows mixing of different search techniques
        - Preserves all available metadata
        
        CONVERSION PROCESS:
        - Copy core fields (chunk_text, similarity, file info)
        - Set hierarchical fields to None/default when not available
        - Preserve metadata for debugging and analysis
        """
        return HierarchicalSearchResult(
            chunk_text=search_result.chunk_text,
            chunk_context=None,  # Regular search doesn't provide context
            similarity=search_result.similarity,
            file_name=search_result.file_name,
            file_path=search_result.file_path,
            chunk_index=search_result.chunk_index,
            section_title=None,  # Regular search doesn't have section info
            document_summary=None,  # Regular search doesn't have document summary
            chunk_metadata=search_result.chunk_metadata
        )
    
    def to_search_result(self, hierarchical_result: HierarchicalSearchResult) -> SearchResult:
        """
        Convert HierarchicalSearchResult to regular SearchResult
        
        WHY DOWNGRADE RESULTS?
        - Some APIs expect regular SearchResult format
        - Enables backward compatibility
        - Simplifies result processing when hierarchy isn't needed
        
        CONVERSION PROCESS:
        - Extract core search fields
        - Merge hierarchical context into metadata if present
        - Preserve similarity scores and file information
        """
        # Merge hierarchical context into metadata
        enhanced_metadata = hierarchical_result.chunk_metadata.copy() if hierarchical_result.chunk_metadata else {}
        
        if hierarchical_result.chunk_context:
            enhanced_metadata["chunk_context"] = hierarchical_result.chunk_context
        if hierarchical_result.section_title:
            enhanced_metadata["section_title"] = hierarchical_result.section_title
        if hierarchical_result.document_summary:
            enhanced_metadata["document_summary"] = hierarchical_result.document_summary
        
        return SearchResult(
            chunk_text=hierarchical_result.chunk_text,
            similarity=hierarchical_result.similarity,
            file_name=hierarchical_result.file_name,
            file_path=hierarchical_result.file_path,
            chunk_index=hierarchical_result.chunk_index,
            chunk_metadata=enhanced_metadata
        )
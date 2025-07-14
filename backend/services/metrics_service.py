import json
import time
import uuid
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

class MetricsService:
    def __init__(self):
        self.metrics_file = Path("/app/metrics/embedding_metrics.jsonl")
        self.metrics_file.parent.mkdir(exist_ok=True, parents=True)
    
    def start_session(self, tenant_name: str, embedding_model: str, chunking_strategy: str, 
                     chunk_size: Optional[int], chunk_overlap: int, total_files: int) -> str:
        """Start a new embedding generation session"""
        session_id = str(uuid.uuid4())
        self.session_data = {
            "session_id": session_id,
            "tenant": tenant_name,
            "model": embedding_model,
            "strategy": chunking_strategy,
            "chunk_size": chunk_size,
            "overlap": chunk_overlap,
            "session_start": datetime.now(timezone.utc).isoformat(),
            "session_end": None,
            "total_files": total_files,
            "total_chunks": 0,
            "total_tokens": 0,
            "total_time_sec": 0,
            "files": []
        }
        return session_id
    
    def log_file_processed(self, file_id: int, filename: str, file_size_bytes: int,
                          chunks_generated: int, tokens_processed: int, 
                          processing_time_sec: float, chunk_distribution: List[int]):
        """Log metrics for a processed file"""
        file_metrics = {
            "file_id": file_id,
            "filename": filename,
            "file_size_bytes": file_size_bytes,
            "chunks_generated": chunks_generated,
            "tokens_processed": tokens_processed,
            "processing_time_sec": round(processing_time_sec, 3),
            "avg_chunk_length": round(tokens_processed / chunks_generated, 1) if chunks_generated > 0 else 0,
            "chunk_distribution": chunk_distribution
        }
        
        self.session_data["files"].append(file_metrics)
        self.session_data["total_chunks"] += chunks_generated
        self.session_data["total_tokens"] += tokens_processed
        self.session_data["total_time_sec"] += processing_time_sec
    
    def end_session(self):
        """End the session and write metrics to file"""
        self.session_data["session_end"] = datetime.now(timezone.utc).isoformat()
        self.session_data["total_time_sec"] = round(self.session_data["total_time_sec"], 3)
        
        # Append to JSONL file
        with open(self.metrics_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(self.session_data) + '\n')
        
        print(f"📊 Metrics logged to {self.metrics_file}")
        print(f"📈 Session summary: {self.session_data['total_files']} files, "
              f"{self.session_data['total_chunks']} chunks, "
              f"{self.session_data['total_time_sec']:.1f}s")
    
    def get_session_summary(self) -> Dict[str, Any]:
        """Get current session summary for real-time display"""
        return {
            "session_id": self.session_data.get("session_id"),
            "files_processed": len(self.session_data["files"]),
            "total_files": self.session_data["total_files"],
            "total_chunks": self.session_data["total_chunks"],
            "total_tokens": self.session_data["total_tokens"],
            "elapsed_time": self.session_data["total_time_sec"],
            "files": self.session_data["files"]
        }
from dataclasses import dataclass
from typing import Optional, Dict, List, Any

@dataclass
class Chunk:
    text: str
    source_path: str
    document_name: str
    summary: Optional[str] = None
    score: Optional[float] = None
    relevance_note: Optional[str] = None
    context: Optional[str] = None
    embedding: Optional[List[float]] = None 

    def to_dict(self) -> Dict[str, Any]:
        # Persistent key fields needed at query time
        return {
            "text": self.text,
            "document_name": self.document_name,
            "source_path": self.source_path,
            "summary": self.summary,          #  persist summary
            "embedding": self.embedding,
            
            "score": self.score,
            "relevance_note": self.relevance_note,
            "context": self.context,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Chunk":
        return cls(
            text=data.get("text", ""),
            document_name=data.get("document_name", ""),
            source_path=data.get("source_path", ""),
            summary=data.get("summary"),               
            score=data.get("score"),
            relevance_note=data.get("relevance_note"),
            context=data.get("context"),
            embedding=data.get("embedding"),         
        )

@dataclass
class SearchResult:
    text: str
    source_path: str
    document_name: str
    score: float
    chunk: Optional[Chunk] = None
    rank: Optional[int] = None
    mode: Optional[str] = None
    detail: Optional[str] = None      
    meta: Optional[Dict[str, Any]] = None

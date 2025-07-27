from dataclasses import dataclass
from typing import Optional, Dict, List

@dataclass
class Chunk:
    text: str
    source_path: str
    document_name: str
    summary: Optional[str] = None
    score: Optional[float] = None
    relevance_note: Optional[str] = None
    context: str = None
    embedding: Optional[List[float]] = None,
    

    def to_dict(self):
        return {
            "text": self.text,
            "document_name": self.document_name,
            "source_path": self.source_path,
            "embedding": self.embedding
        }

    @classmethod
    def from_dict(cls, data: dict):
        return cls(
            text=data.get("text", ""),
            document_name=data.get("document_name", ""),
            source_path=data.get("source_path", ""),
            embedding=data.get("embedding")  # can be None or a list of floats
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
    meta: Optional[Dict] = None

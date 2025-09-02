from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, List, Any, Tuple


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

    ordinal: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        
        return {
            "text": self.text,
            "document_name": self.document_name,
            "source_path": self.source_path,
            "summary": self.summary,
            "embedding": self.embedding,
            "score": self.score,
            "relevance_note": self.relevance_note,
            "context": self.context,
            "ordinal": self.ordinal,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Chunk":
        return cls(
            text=data.get("text", ""),
            document_name=data.get("document_name", ""),
            source_path=data.get("source_path", ""),
            summary=data.get("summary"),
            score=data.get("score"),
            relevance_note=data.get("relevance_note"),
            context=data.get("context"),
            embedding=data.get("embedding"),
            ordinal=data.get("ordinal"),
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
    meta: Optional[Dict[str, Any]] = field(default_factory=dict)


    def to_dict(self) -> Dict[str, Any]:
        
        d = asdict(self)
        if self.chunk is not None:
            d["chunk"] = self.chunk.to_dict()
        return d

    @classmethod
    def from_chunk(
        cls,
        chunk: Chunk,
        score: float,
        rank: Optional[int] = None,
        mode: Optional[str] = None,
        detail: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> "SearchResult":
        return cls(
            text=chunk.text,
            source_path=chunk.source_path,
            document_name=chunk.document_name,
            score=float(score),
            chunk=chunk,
            rank=rank,
            mode=mode,
            detail=detail,
            meta=meta or {},
        )

    @classmethod
    def from_pair(
        cls,
        pair: Tuple[Chunk, float],
        rank: Optional[int] = None,
        mode: Optional[str] = None,
        detail: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> "SearchResult":
        
        ch, s = pair
        return cls.from_chunk(ch, s, rank=rank, mode=mode, detail=detail, meta=meta or {})

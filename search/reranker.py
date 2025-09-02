from typing import List, Optional
import numpy as np

from embedding.ollama_embedder import OllamaEmbedder
from models.data_models import SearchResult


class Reranker:
    
    def __init__(self, model_name: str = "mxbai-embed-large"):
        self.embedder = OllamaEmbedder(model_name=model_name)

    def rerank(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        if not results:
            return results

        # Embed query once
        q_vec = np.array(self.embedder.get_embedding(query), dtype=float)
        if q_vec.ndim != 1 or np.linalg.norm(q_vec) == 0:
            # If something odd happens, just return the original ordering
            return results

        for r in results:
            chunk_text: Optional[str] = getattr(getattr(r, "chunk", None), "text", None)
            if not chunk_text:
                continue

            c_vec = np.array(self.embedder.get_embedding(chunk_text), dtype=float)
            score = self._cosine_similarity(q_vec, c_vec)

            
            if r.chunk is not None:
                r.chunk.score = score
            
            if r.meta is None:
                r.meta = {}
            r.meta["rerank_score"] = float(score)

           
            suffix = f"Rerank Score: {score:.4f}"
            r.detail = (f"{r.detail} | {suffix}") if r.detail else suffix

        # Sort by rerank score if available
        results.sort(
            key=lambda it: (
                float(getattr(getattr(it, "chunk", None), "score", float("-inf")))
            ),
            reverse=True,
        )

        # reassign ranks to reflect new order
        for i, r in enumerate(results, start=1):
            r.rank = i

        return results

    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        if v1.ndim != 1 or v2.ndim != 1:
            return 0.0
        denom = float(np.linalg.norm(v1) * np.linalg.norm(v2))
        if denom == 0.0:
            return 0.0
        return float(np.dot(v1, v2) / denom)


from embedding.ollama_embedder import OllamaEmbedder
from models.data_models import SearchResult
import numpy as np
from typing import List

class Reranker:
    def __init__(self):
        self.embedder = OllamaEmbedder(model_name="mxbai-embed-large")

    def rerank(self, query: str, results: List[SearchResult]) -> List[SearchResult]:
        query_vec = self.embedder.get_embedding(query)

        for result in results:
            chunk_text = result.chunk.text
            chunk_vec = self.embedder.get_embedding(chunk_text)
            score = self._cosine_similarity(query_vec, np.array(chunk_vec))
            result.chunk.score = score
            result.detail = (result.detail + f" | Rerank Score: {score:.4f}") if result.detail else f"Rerank Score: {score:.4f}"


        results.sort(key=lambda r: r.chunk.score, reverse=True)
        for i, result in enumerate(results):
            result.rank = i + 1
        return results

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))


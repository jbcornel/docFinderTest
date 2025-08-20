

from typing import List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed
import numpy as np

from models.data_models import Chunk, SearchResult
from embedding.embedding_manager import EmbeddingManager
from summarization.summarizer import Summarizer 



SUMMARY_MODEL_MAP = {
    "llama3": "llama3:latest",  
    "phi3": "phi3:mini",
    "tinyllama": "tinyllama:1.1b",
}


class SemanticSearchEngine:
    """
    - Ranks chunks by cosine similarity (unchanged)
    - Summarizes ONLY the first N results at query time using your Summarizer
      (N = summarize_topk, 0..5). Parallelized up to max_workers.
    """

    def __init__(self, summary_model: str = "llama3", max_workers: int = 5):
        self.embedder = EmbeddingManager.get_instance()
    
        self.summary_model = SUMMARY_MODEL_MAP.get(summary_model.lower(), summary_model)
        self.max_workers = max(1, int(max_workers))

    def _cosine(self, v1: np.ndarray, v2: np.ndarray) -> float:
        denom = np.linalg.norm(v1) * np.linalg.norm(v2)
        if denom == 0:
            return 0.0
        return float(np.dot(v1, v2) / denom)

    def _summarize_batch(self, texts: List[str], workers: int) -> List[str]:
       
        if not texts:
            return []
        summarizer = Summarizer(
            model_name=self.summary_model,
            workers=min(workers, len(texts)),
            max_tokens=96,
            temperature=0.0,
        )
        return summarizer.batch(texts)

    def search(
        self,
        query: str,
        chunks: List[Chunk],
        top_k: int = 5,
        summarize_topk: int = 2,   # how many of the top results to summarize (0..5)
    ) -> List[SearchResult]:
        # 1 Embed query
        try:
            q = np.array(self.embedder.get_single_embedding(query))
        except Exception as e:
            print(f"[Error] Failed to embed query: {e}")
            return []

        # 2 Score chunks by cosine similarity
        scored: List[Tuple[Chunk, float]] = []
        for ch in chunks:
            if not ch.text or not getattr(ch, "embedding", None):
                continue
            vec = np.array(ch.embedding)
            if vec.ndim != 1 or vec.shape != q.shape or np.linalg.norm(vec) == 0:
                continue
            s = self._cosine(q, vec)
            ch.score = float(s)
            scored.append((ch, s))

        if not scored:
            return []

        # 3 Take top_k
        top = sorted(scored, key=lambda x: x[1], reverse=True)[:max(1, int(top_k))]

        # 4 Summaries for only the first N (clamped 0..5 and <= len(top))
        n_to_summarize = max(0, min(5, int(summarize_topk)))
        n_to_summarize = min(n_to_summarize, len(top))

        summaries = [""] * len(top)
        if n_to_summarize > 0:
            texts = [top[i][0].text for i in range(n_to_summarize)]
            try:
                outs = self._summarize_batch(texts, workers=self.max_workers)
            except Exception as e:
                print(f"[Warn] Query-time summarization failed: {e}")
                outs = ["[Summary unavailable]"] * len(texts)

            for i, s in enumerate(outs):
                summaries[i] = (s or "").strip() or "[Summary unavailable]"

        # 5 Build SearchResult list (detail = summary for first N, empty after)
        results: List[SearchResult] = []
        for idx, (ch, score) in enumerate(top, start=1):
            results.append(SearchResult(
                text=ch.text,
                source_path=ch.source_path,
                document_name=ch.document_name,
                score=score,
                mode="semantic",
                detail=summaries[idx - 1],  # blank if idx-1 >= n_to_summarize
                chunk=ch,
                rank=idx
            ))
        return results

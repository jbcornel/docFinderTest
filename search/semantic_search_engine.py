from typing import List, Tuple
import numpy as np

from models.data_models import Chunk, SearchResult
from embedding.embedding_manager import EmbeddingManager
from summarization.summarizer import Summarizer
from vector_store.chroma_store import ChromaVectorStore


SUMMARY_MODEL_MAP = {
    "llama3": "llama3:latest",
    "phi3": "phi3:mini",
    "tinyllama": "tinyllama:1.1b",
}


class SemanticSearchEngine:
   

    def __init__(
        self,
        summary_model: str = "llama3",
        max_workers: int = 5,
        persist_dir: str = "./chroma_store/semantic",
        collection_name: str = "docfinder_semantic",
    ):
        self.embedder = EmbeddingManager.get_instance()
     
        self.summary_model = SUMMARY_MODEL_MAP.get(str(summary_model).lower(), summary_model)
        self.max_workers = max(1, int(max_workers))
        self.persist_dir = persist_dir
        self.collection_name = collection_name

        self._vs = None  

  

    def _cosine(self, v1: np.ndarray, v2: np.ndarray) -> float:
        denom = np.linalg.norm(v1) * np.linalg.norm(v2)
        if denom == 0:
            return 0.0
        return float(np.dot(v1, v2) / denom)

    def _as_results(self, pairs: List[Tuple[Chunk, float]]) -> List[SearchResult]:
        out: List[SearchResult] = []
        for idx, (ch, score) in enumerate(pairs, start=1):
            out.append(SearchResult(
                text=ch.text,
                source_path=ch.source_path,
                document_name=ch.document_name,
                score=score,
                detail="",
                mode="semantic",
                chunk=ch,
                rank=idx
            ))
        return out

    def _summarize_top(self, results: List[SearchResult], n: int) -> None:
        n = max(0, min(5, int(n)))
        if n == 0 or not results:
            return

        texts = [r.text for r in results[:n]]
        try:
            summarizer = Summarizer(
                model_name=self.summary_model,
                workers=min(self.max_workers, len(texts)),
                max_tokens=96,
                temperature=0.0,
            )
            outs = summarizer.batch(texts)
        except Exception as e:
            print(f"[Warn] Query-time summarization failed: {e}")
            outs = ["[Summary unavailable]"] * len(texts)

        for i, s in enumerate(outs):
            summary = (s or "").strip() or "[Summary unavailable]"
            r = results[i]
            r.detail = summary
            if getattr(r, "chunk", None):
                r.chunk.summary = summary
                if self._vs is not None:
                    try:
                        self._vs.update_summary(r.chunk, summary)
                    except Exception as e:
                        print(f"[Warn] Failed to update summary in vector store: {e}")


    def search_store(
        self,
        query: str,
        top_k: int = 5,
        summarize_topk: int = 0,
    ) -> List[SearchResult]:
        
        if self._vs is None:
            self._vs = ChromaVectorStore(
                persist_dir=self.persist_dir,
                collection_name=self.collection_name,
            )
        try:
            pairs: List[Tuple[Chunk, float]] = self._vs.query(query, top_k=max(1, int(top_k)))
        except Exception as e:
            print(f"[Error] Vector store query failed: {e}")
            return []

        results = self._as_results(pairs)
        self._summarize_top(results, summarize_topk)
        return results

    def search_memory(
        self,
        query: str,
        chunks: List[Chunk],
        top_k: int = 5,
        summarize_topk: int = 0,
    ) -> List[SearchResult]:
        
        try:
            q = np.array(self.embedder.get_single_embedding(query))
        except Exception as e:
            print(f"[Error] Failed to embed query: {e}")
            return []

        scored: List[Tuple[Chunk, float]] = []
        for ch in chunks:
            vec = getattr(ch, "embedding", None)
            if not ch.text or vec is None:
                continue
            vec = np.array(vec)
            if vec.ndim != 1 or vec.shape != q.shape or np.linalg.norm(vec) == 0:
                continue
            s = self._cosine(q, vec)
            ch.score = float(s)
            scored.append((ch, s))

        if not scored:
            return []

        top = sorted(scored, key=lambda x: x[1], reverse=True)[:max(1, int(top_k))]
        results = self._as_results(top)
        self._summarize_top(results, summarize_topk)
        return results

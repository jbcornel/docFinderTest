
from __future__ import annotations
from typing import List, Optional, Callable
import re
import math

import numpy as np

from .base_chunker import BaseChunker
from models.data_models import Chunk
from embedding.embedding_manager import EmbeddingManager


def _default_sentence_split(text: str) -> List[str]:
    """
    Lightweight sentence splitter:
    - Splits on end punctuation followed by space/case change
    - Also respects newlines as soft boundaries
    """
    if not isinstance(text, str) or not text.strip():
        return []
    
    t = re.sub(r"[ \t]+", " ", text.strip())
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])|(?<=\n)\s*(?=\S)", t)
    out: List[str] = []
    for p in parts:
        p = p.strip()
        if p:
           
            out.extend([seg.strip() for seg in p.split("\n") if seg.strip()])
    return out


def _approx_tokens(s: str) -> int:
    """Very rough token estimate (~4 chars/token)."""
    if not isinstance(s, str):
        return 0
    return max(1, len(s) // 4)


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    na = np.linalg.norm(a)
    nb = np.linalg.norm(b)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))




class SemanticClusterChunker(BaseChunker):
    """
    Cluster adjacent sentences by semantic similarity (embedding-based) with a token budget.

    Design goals:
      - **Only** produce chunks; do **not** attach `chunk.embedding` (main.py handles embeddings).
      - Single batch of sentence embeddings per file.
      - Greedy growth: keep appending sentences while similarity to the running centroid
        stays above a threshold and the token budget allows; otherwise start a new chunk.

    Parameters
    ----------
    similarity_threshold : float
        Minimum cosine similarity with the current cluster centroid to append the next sentence.
        Increase (e.g., 0.68–0.72) for tighter topicality, decrease (e.g., 0.55–0.60) for longer chunks.
    max_tokens : int
        Rough token cap per chunk (default 1100).
    min_tokens : int
        Minimum target tokens before we allow topic-based splits or flush due to budget (default 350).
    min_sentences : int
        Minimum sentences per chunk (default 1).
    sentence_splitter : Optional[Callable[[str], List[str]]]
        Custom sentence splitter; defaults to a simple regex splitter.
    debug : bool
        If True, prints progress (chunks formed, flush reasons, etc.).
    """

    def __init__(
        self,
        similarity_threshold: float = 0.58,
        max_tokens: int = 1100,
        min_tokens: int = 350,
        min_sentences: int = 1,
        sentence_splitter: Optional[Callable[[str], List[str]]] = None,
        debug: bool = False,
    ):
        self.similarity_threshold = float(similarity_threshold)
        self.max_tokens = int(max_tokens)
        self.min_tokens = int(min_tokens)
        self.min_sentences = max(1, int(min_sentences))
        self.split_sentences = sentence_splitter or _default_sentence_split
        self.debug = bool(debug)


        self.embedder = EmbeddingManager.get_instance()


    def chunk(self, text: str, source_path: str, document_name: str) -> List[Chunk]:
       
        sents = self.split_sentences(text)
        sents = [s for s in sents if isinstance(s, str) and s.strip()]
        if self.debug:
            print(f"[SemCluster] file='{document_name}' sentences={len(sents)}")

        if not sents:
            return []

       
        try:
            sent_vecs: List[List[float]] = self.embedder.get_embedding(sents)
        except Exception as e:
            if self.debug:
                print(f"[SemCluster] embedding error: {e}. Falling back to size-only grouping.")

            sent_vecs = [[] for _ in sents]

        
        chunks_text: List[str] = []

        cur_sents: List[str] = []
        cur_vecs: List[np.ndarray] = []
        cur_tokens = 0
        cur_centroid: Optional[np.ndarray] = None

        def _flush(reason: str):
            nonlocal cur_sents, cur_vecs, cur_tokens, cur_centroid
            if not cur_sents:
                return
            body = " ".join(cur_sents).strip()
            if body:
                chunks_text.append(body)
                if self.debug:
                    print(f"[SemCluster] flush: {reason}; chunk_len={len(body)} chars, sents={len(cur_sents)}, tokens~{cur_tokens}")
            cur_sents = []
            cur_vecs = []
            cur_tokens = 0
            cur_centroid = None

        for i, (s, v) in enumerate(zip(sents, sent_vecs)):
            tok = _approx_tokens(s)
            v_np = np.asarray(v, dtype=float) if isinstance(v, list) and v else None

            if not cur_sents:
            
                cur_sents.append(s)
                if v_np is not None:
                    cur_vecs.append(v_np)
                    cur_centroid = v_np
                cur_tokens = tok
                continue



            if cur_tokens + tok > self.max_tokens and len(cur_sents) >= self.min_sentences and cur_tokens >= self.min_tokens:
                _flush("token budget")

                cur_sents.append(s)
                if v_np is not None:
                    cur_vecs.append(v_np)
                    cur_centroid = v_np
                else:
                    cur_centroid = None
                cur_tokens = tok
                continue


            keep = True
            if cur_centroid is not None and v_np is not None:
                sim = _cos(cur_centroid, v_np)
                keep = (sim >= self.similarity_threshold)
                if self.debug:
                    print(f"[SemCluster] sent#{i} sim={sim:.3f} th={self.similarity_threshold} -> {'keep' if keep else 'split'}")


            if not keep and cur_tokens < self.min_tokens:
                keep = True  

            if keep or len(cur_sents) < self.min_sentences:
                cur_sents.append(s)
                if v_np is not None:
                    cur_vecs.append(v_np)

                    cur_centroid = np.mean(np.stack(cur_vecs, axis=0), axis=0)
                cur_tokens += tok
            else:
                _flush("topic shift")
                cur_sents.append(s)
                if v_np is not None:
                    cur_vecs.append(v_np)
                    cur_centroid = v_np
                else:
                    cur_centroid = None
                cur_tokens = tok


        _flush("end")

        if self.debug:
            print(f"[SemCluster] produced chunks={len(chunks_text)}")


        if len(chunks_text) >= 2:
            last_tokens = _approx_tokens(chunks_text[-1])
            if last_tokens < max(20, int(self.min_tokens * 0.5)):

                chunks_text[-2] = (chunks_text[-2].rstrip() + "\n\n" + chunks_text[-1].lstrip()).strip()
                chunks_text.pop()


        out: List[Chunk] = []
        for idx, body in enumerate(chunks_text):
            ch = Chunk(
                text=body,
                source_path=source_path,
                document_name=document_name,
            )
            setattr(ch, "ordinal", idx)
            out.append(ch)

        return out

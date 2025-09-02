from typing import List, Tuple
import re
import math

from models.data_models import Chunk
from embedding.embedding_manager import EmbeddingManager

def _approx_tokens(s: str) -> int:

    return max(1, math.ceil(len(s) / 4))

def _sentences(text: str) -> List[str]:
    """
    Lightweight sentence splitter (no external deps).
    Keeps delimiters, trims whitespace, drops empties.
    """
    if not isinstance(text, str) or not text.strip():
        return []

    parts = re.split(r'([.!?])', text)
    sents, cur = [], ""
    for p in parts:
        if p in (".", "!", "?"):
            cur += p
            sents.append(cur.strip())
            cur = ""
        else:
            cur += p
    if cur.strip():
        sents.append(cur.strip())

    merged: List[str] = []
    for s in sents:
        if merged and len(s) < 8:
            merged[-1] = (merged[-1] + " " + s).strip()
        else:
            merged.append(s)
    return [s for s in merged if s.strip()]

def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    sa = sum(x*x for x in a)
    sb = sum(y*y for y in b)
    if sa == 0.0 or sb == 0.0:
        return 0.0
    dot = sum(x*y for x, y in zip(a, b))
    return dot / (sa**0.5 * sb**0.5)

class SemanticChunker:
    """
    Semantic chunker that groups adjacent sentences until
    (a) we hit a target size, and
    (b) similarity to the boundary sentence drops below a threshold.

    Produces List[Chunk] identical to your current chunker output,
    so the rest of your system (ingest, Chroma, exact/semantic search) just works.
    """

    def __init__(
        self,
        target_chunk_tokens: int = 350,  
        max_chunk_tokens: int = 500,      
        min_chunk_tokens: int = 120,      
        boundary_sim_threshold: float = 0.60,  
        smoothing: int = 1,            
    ):
        self.target = int(target_chunk_tokens)
        self.max_tokens = int(max_chunk_tokens)
        self.min_tokens = int(min_chunk_tokens)
        self.threshold = float(boundary_sim_threshold)
        self.smoothing = max(0, int(smoothing))
        self.embedder = EmbeddingManager.get_instance()

    def chunk(self, text: str, source_path: str, document_name: str) -> List[Chunk]:
        sents = _sentences(text)
        if not sents:
            return []

        # Embed sentences
        sent_embs = self.embedder.get_embedding(sents)  

        chunks: List[Chunk] = []
        cur_texts: List[str] = []
        cur_tokens = 0

        def _flush():
            nonlocal cur_texts, cur_tokens
            if not cur_texts:
                return
            body = " ".join(cur_texts).strip()
            if not body:
                cur_texts, cur_tokens = [], 0
                return
            ch = Chunk(
                text=body,
                source_path=source_path,
                document_name=document_name,
            )
            chunks.append(ch)
            cur_texts, cur_tokens = [], 0

      
        def _boundary_sim(last_idx: int, next_idx: int) -> float:
            if last_idx < 0 or next_idx < 0:
                return 0.0
            # average last_k of left and first_k of right
            left_idxs = list(range(max(0, last_idx - self.smoothing), last_idx + 1))
            right_idxs = list(range(next_idx, min(len(sent_embs), next_idx + 1 + self.smoothing)))
            if not left_idxs or not right_idxs:
                return 0.0
            # average vectors
            def avg_vec(idxs: List[int]) -> List[float]:
                if not idxs:
                    return []
                dim = len(sent_embs[idxs[0]]) if sent_embs[idxs[0]] else 0
                acc = [0.0] * dim
                for i in idxs:
                    v = sent_embs[i]
                    if not v:
                        continue
                    for d in range(dim):
                        acc[d] += v[d]
                n = max(1, len(idxs))
                return [x / n for x in acc]
            return _cosine(avg_vec(left_idxs), avg_vec(right_idxs))

        # Walk sentences and build chunks
        for i, s in enumerate(sents):
            t = _approx_tokens(s)
            # if adding this sentence exceeds absolute max, flush first
            if cur_tokens > 0 and (cur_tokens + t) > self.max_tokens:
                _flush()

            cur_texts.append(s)
            cur_tokens += t

            # if we're comfortably below minimum size, keep adding
            if cur_tokens < self.min_tokens:
                continue

            # if we're above target, consider boundary; otherwise keep collecting
            if cur_tokens >= self.target:
                sim = _boundary_sim(i, i + 1) if i + 1 < len(sents) else 0.0
                if sim < self.threshold:
                    _flush()

        # flush any remainder
        _flush()

        # assign ordinals (per-file stable order) — helpful for Chroma IDs
        for idx, ch in enumerate(chunks):
            setattr(ch, "ordinal", idx)

        return chunks

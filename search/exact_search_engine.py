
import math
from collections import Counter, defaultdict
from typing import List
from models.data_models import Chunk, SearchResult
import re

class ExactSearchEngine:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1
        self.b = b

    def _tokenize(self, text: str) -> List[str]:
        if not isinstance(text, str):
            raise ValueError(f"Expected a string for tokenization, but got {type(text)}")
        return text.lower().split()

    def _highlight_matches(self, text: str, query_tokens: List[str]) -> str:
        # Highlight query tokens using ANSI escape codes (or you can adapt for HTML/UI later)
        for token in query_tokens:
            pattern = re.compile(re.escape(token), re.IGNORECASE)
            text = pattern.sub(lambda m: f"\033[93m{m.group(0)}\033[0m", text)  # Yellow highlight
        return text

    def _compute_idf(self, chunks: List[Chunk], query_tokens: List[str]) -> dict:
        N = len(chunks)
        df = defaultdict(int)
        for chunk in chunks:
            seen = set()
            if not hasattr(chunk, 'text') or not isinstance(chunk.text, str):
                raise ValueError("Each chunk must have a 'text' attribute of type str")
            tokens = set(self._tokenize(chunk.text))
            for token in query_tokens:
                if token in tokens and token not in seen:
                    df[token] += 1
                    seen.add(token)
        idf = {}
        for token in query_tokens:
            df_token = df[token] if df[token] else 1
            idf[token] = math.log((N - df_token + 0.5) / (df_token + 0.5) + 1)
        return idf

    def _count_exact_matches(self, tokens: List[str], query_tokens: List[str]) -> int:
        return sum(1 for i in range(len(tokens) - len(query_tokens) + 1)
                   if tokens[i:i+len(query_tokens)] == query_tokens)

    def search(self, query: str, chunks: List[Chunk]) -> List[SearchResult]:
        query_tokens = self._tokenize(query)
        idf = self._compute_idf(chunks, query_tokens)
        avgdl = sum(len(self._tokenize(c.text)) for c in chunks if isinstance(c.text, str)) / len(chunks)

        scored_chunks = []
        for chunk in chunks:
            if not isinstance(chunk.text, str):
                continue  # skip malformed chunks
            tokens = self._tokenize(chunk.text)
            tf = Counter(tokens)
            doc_len = len(tokens)
            score = 0
            for token in query_tokens:
                f = tf[token]
                numer = f * (self.k1 + 1)
                denom = f + self.k1 * (1 - self.b + self.b * doc_len / avgdl)
                score += idf[token] * (numer / denom)

            # Exact phrase match bonus
            exact_count = self._count_exact_matches(tokens, query_tokens)
            if exact_count > 0:
                score += 10.0 * exact_count  # Boost factor for full phrase matches

            if score > 0:
                highlighted_text = self._highlight_matches(chunk.text, query_tokens)
                scored_chunks.append(SearchResult(
                    text=highlighted_text,
                    source_path=chunk.source_path,
                    document_name=chunk.document_name,
                    score=score,
                    mode="exact",
                    detail=f"{score:.4f}",
                    meta={"bm25_score": round(score, 4), "exact_match_count": exact_count},
                    chunk=chunk
                ))

        # Assign ranks explicitly
        sorted_results = sorted(scored_chunks, key=lambda x: x.score, reverse=True)
        for i, result in enumerate(sorted_results):
            result.rank = i + 1

        return sorted_results[:10]

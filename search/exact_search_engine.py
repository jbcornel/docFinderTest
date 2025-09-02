

import re
import math
from typing import List, Dict, Tuple
from collections import Counter, defaultdict
from difflib import SequenceMatcher

from models.data_models import Chunk, SearchResult


class ExactSearchEngine:
   
    def __init__(
        self,
        k1: float = 1.5,
        b: float = 0.75,
        exact_weight: float = 10.0,
        partial_weight: float = 2.0,
        fuzzy_threshold: float = 0.84, 
    ):
        self.k1 = k1
        self.b = b
        self.exact_weight = exact_weight
        self.partial_weight = partial_weight
        self.fuzzy_threshold = fuzzy_threshold

   
    def _tokenize(self, text: str) -> List[str]:
        
        return re.findall(r"\w+", (text or "").lower())

    def _idf(self, N: int, df: int) -> float:
        return math.log((N - df + 0.5) / (df + 0.5) + 1)

    def _similar(self, a: str, b: str) -> float:
        return SequenceMatcher(None, a, b).ratio()

  
    def _compute_idf_map(self, chunks: List[Chunk], query_tokens: List[str]) -> Dict[str, float]:
        N = len(chunks)
        df_map: Dict[str, int] = defaultdict(int)
        for ch in chunks:
            toks = set(self._tokenize(ch.text))
            for qt in query_tokens:
                if qt in toks:
                    df_map[qt] += 1
        return {qt: self._idf(N, df_map[qt]) for qt in query_tokens}

    def _bm25_with_fuzzy(
        self,
        tokens: List[str],
        tf: Counter,
        query_tokens: List[str],
        idf_map: Dict[str, float],
        fuzzy: bool,
    ) -> Tuple[float, Dict]:
       
        score = 0.0
        doc_len = len(tokens) or 1
        avgdl = doc_len

        per_token_rows = []
        fuzzy_match_count_total = 0

        for qt in query_tokens:
            f_exact = tf[qt]

    
            f_fuzzy = 0
            if fuzzy:
                for tok, c in tf.items():
                    if tok == qt:
                        continue
                    if self._similar(qt, tok) >= self.fuzzy_threshold:
                        f_fuzzy += c

            f_eff = f_exact + f_fuzzy
            fuzzy_match_count_total += (1 if f_fuzzy > 0 else 0)

            if f_eff == 0:
            
                per_token_rows.append({
                    "token": qt, "idf": idf_map.get(qt, 0.0),
                    "df": None, "f_exact": f_exact, "f_fuzzy": f_fuzzy,
                    "f_effective": f_eff, "bm25_component": 0.0
                })
                continue

            idf_val = idf_map.get(qt, 0.0)
            numer = f_eff * (self.k1 + 1)
            denom = f_eff + self.k1 * (1 - self.b + self.b * doc_len / (avgdl or 1))
            comp = idf_val * (numer / (denom or 1.0))
            score += comp

            per_token_rows.append({
                "token": qt, "idf": idf_val,
                "df": None,  
                "f_exact": f_exact, "f_fuzzy": f_fuzzy,
                "f_effective": f_eff, "bm25_component": comp
            })

        diagnostics = {
            "k1": self.k1, "b": self.b,
            "avgdl": float(avgdl), "doc_len": int(doc_len),
            "per_token": per_token_rows,
            "fuzzy_match_count": fuzzy_match_count_total,
            "fuzzy_threshold": int(self.fuzzy_threshold * 100), 
        }
        return score, diagnostics

    def _exact_partial_counts(self, tokens: List[str], query_tokens: List[str]) -> Tuple[int, int]:
        exact_cnt = 0
        partial_cnt = 0
        token_set = set(tokens)

        for qt in query_tokens:
            if qt in token_set:
                exact_cnt += 1
            else:
               
                for tok in token_set:
                    if tok != qt and (qt in tok or tok in qt):
                        partial_cnt += 1
                        break
        return exact_cnt, partial_cnt

  
    def search(
        self,
        query: str,
        chunks: List[Chunk],
        top_k: int = 5,
        fuzzy: bool = True,
    ) -> List[SearchResult]:
        query_tokens = self._tokenize(query)
        if not query_tokens:
            return []

        idf_map = self._compute_idf_map(chunks, query_tokens)

        scored: List[Tuple[float, dict, Chunk]] = []
        for ch in chunks:
            tokens = self._tokenize(ch.text)
            tf = Counter(tokens)

            
            bm25, bm25_diag = self._bm25_with_fuzzy(tokens, tf, query_tokens, idf_map, fuzzy=fuzzy)

          
            exact_cnt, partial_cnt = self._exact_partial_counts(tokens, query_tokens)
            exact_bonus = self.exact_weight * exact_cnt
            partial_bonus = self.partial_weight * partial_cnt

            total = bm25 + exact_bonus + partial_bonus

            has_signal = (exact_cnt > 0) or (partial_cnt > 0) or (fuzzy and bm25_diag.get("fuzzy_match_count", 0) > 0)
            if not has_signal:
                continue

            meta = {
                "bm25_score": bm25,
                "exact_match_count": exact_cnt,
                "partial_match_count": partial_cnt,
                "fuzzy_match_count": bm25_diag.get("fuzzy_match_count", 0),
                "fuzzy_threshold": bm25_diag.get("fuzzy_threshold"),
                "bm25_details": {
                    "k1": self.k1, "b": self.b,
                    "N": len(chunks),
                    "avgdl": bm25_diag.get("avgdl"),
                    "doc_len": bm25_diag.get("doc_len"),
                    "per_token": bm25_diag.get("per_token"),
                },
            }

            scored.append((total, {
                "bm25": bm25,
                "exact_bonus": exact_bonus,
                "partial_bonus": partial_bonus,
                "meta": meta,
                "total": total
            }, ch))

      
        scored.sort(key=lambda t: t[0], reverse=True)
        results: List[SearchResult] = []

        for rank_idx, (_, parts, ch) in enumerate(scored[:max(1, top_k)], start=1):
            bm25 = parts["bm25"]
            exact_bonus = parts["exact_bonus"]
            partial_bonus = parts["partial_bonus"]
            total = parts["total"]
            meta = parts["meta"]

            res = SearchResult(
                text=ch.text,
                source_path=ch.source_path,
                document_name=ch.document_name,
                score=total,
                detail=f"BM25 base={bm25:.4f} | Exact={exact_bonus:.4f} | Partial={partial_bonus:.4f} | Total={total:.4f}",
            )
          
            setattr(res, "mode", "exact+fuzzy" if fuzzy else "exact")
            setattr(res, "rank", rank_idx)
            setattr(res, "meta", meta)
            setattr(res, "chunk", ch)  # so formatter can read chunk.summary if present

            results.append(res)

        return results

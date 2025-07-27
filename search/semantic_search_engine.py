
# from models.data_models import Chunk, SearchResult
# from embedding.ollama_embedder import OllamaEmbedder
# from typing import List
# import numpy as np
# import ollama

# class SemanticSearchEngine:
#     def __init__(self):
#         self.embedder = OllamaEmbedder(model_name="mxbai-embed-large")

#     def search(self, query: str, chunks: List[Chunk], top_k: int = 5) -> List[SearchResult]:
#         query_vec = self.embedder.get_embedding(query)
#         scored = []

#         for chunk in chunks:
#             if not chunk.text:
#                 continue
#             vec = self.embedder.get_embedding(chunk.text)
#             score = self._cosine_similarity(np.array(query_vec), np.array(vec))
#             chunk.score = float(score)
#             scored.append((chunk, score))

#         top_chunks = sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]

#         results = []
#         for i, (chunk, score) in enumerate(top_chunks):
#             explanation = self._explain_semantic_match(query, chunk.text)
#             result = SearchResult(
#                 text=chunk.text,
#                 source_path=chunk.source_path,
#                 document_name=chunk.document_name,
#                 score=score,
#                 mode="semantic",
#                 detail=explanation,
#                 chunk=chunk,
#                 rank=i + 1
#             )
#             results.append(result)

#         return results

#     def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
#         return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

#     def _explain_semantic_match(self, query: str, chunk_text: str) -> str:
#         try:
#             prompt = (
#                 "Explain in 3 sentences how the following passage is semantically related to the query.\n\n"
#                 f"Query: {query}\n\n"
#                 f"Passage: {chunk_text}\n\n"
#                 "Explanation:"
#             )
#             response = ollama.generate(model="llama3", prompt=prompt)
#             return response['response'].strip()
#         except Exception as e:
#             return f"[Explanation unavailable: {str(e)}]"


# from models.data_models import Chunk, SearchResult
# from embedding.ollama_embedder import OllamaEmbedder
# from typing import List
# import numpy as np
# import ollama

# class SemanticSearchEngine:
#     def __init__(self):
#         self.embedder = OllamaEmbedder(model_name="mxbai-embed-large")

#     def search(self, query: str, chunks: List[Chunk], top_k: int = 5) -> List[SearchResult]:
#         query_vec = self.embedder.get_embedding(query)
#         scored = []

#         for chunk in chunks:
#             # Skip if missing required fields
#             if not chunk.text or not hasattr(chunk, "embedding") or not isinstance(chunk.embedding, list):
#                 print(f"[WARN] Skipping chunk due to missing embedding: {chunk.document_name}")
#                 continue

#             vec = np.array(chunk.embedding)  # ✅ Use precomputed embedding from ingestion only
#             score = self._cosine_similarity(np.array(query_vec), vec)
#             chunk.score = float(score)
#             scored.append((chunk, score))

#         top_chunks = sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]

#         results = []
#         for i, (chunk, score) in enumerate(top_chunks):
#             explanation = self._explain_semantic_match(query, chunk.text)
#             result = SearchResult(
#                 text=chunk.text,
#                 source_path=chunk.source_path,
#                 document_name=chunk.document_name,
#                 score=score,
#                 mode="semantic",
#                 detail=explanation,
#                 chunk=chunk,
#                 rank=i + 1
#             )
#             results.append(result)

#         return results

#     def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
#         return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

#     def _explain_semantic_match(self, query: str, chunk_text: str) -> str:
#         try:
#             prompt = (
#                 "Explain in 3 sentences how the following passage is semantically related to the query.\n\n"
#                 f"Query: {query}\n\n"
#                 f"Passage: {chunk_text}\n\n"
#                 "Explanation:"
#             )
#             response = ollama.generate(model="llama3", prompt=prompt)
#             return response['response'].strip()
#         except Exception as e:
#             return f"[Explanation unavailable: {str(e)}]"
# from models.data_models import Chunk, SearchResult
# from embedding.ollama_embedder import OllamaEmbedder
# from typing import List
# import numpy as np
# import ollama

# class SemanticSearchEngine:
#     def __init__(self):
#         self.embedder = OllamaEmbedder(model_name="mxbai-embed-large")

#     def search(self, query: str, chunks: List[Chunk], top_k: int = 5) -> List[SearchResult]:
#         # Fix: ensure query is passed as a single string, not list
#         if isinstance(query, list):
#             query = query[0]

#         # Ensure query_vec is unwrapped if returned as list
#         query_embedding = self.embedder.get_embedding(query)
#         if isinstance(query_embedding, list) and len(query_embedding) == 1:
#             query_vec = np.array(query_embedding[0])
#         elif isinstance(query_embedding, list) and isinstance(query_embedding[0], list):
#             query_vec = np.array(query_embedding[0])
#         else:
#             query_vec = np.array(query_embedding)

#         if query_vec.ndim != 1:
#             raise ValueError(f"Query embedding must be a 1D vector, got shape {query_vec.shape}")

#         scored = []

#         for chunk in chunks:
#             if not chunk.text or not hasattr(chunk, "embedding") or not chunk.embedding:
#                 continue

#             vec = np.array(chunk.embedding)
#             if vec.ndim != 1:
#                 print(f"[Warning] Skipping chunk due to invalid embedding shape: {vec.shape}")
#                 continue
#             if vec.shape != query_vec.shape:
#                 print(f"[Warning] Skipping chunk due to embedding shape mismatch: {vec.shape} vs {query_vec.shape}")
#                 continue

#             score = self._cosine_similarity(query_vec, vec)
#             chunk.score = float(score)
#             scored.append((chunk, score))

#         top_chunks = sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]

#         results = []
#         for i, (chunk, score) in enumerate(top_chunks):
#             explanation = self._explain_semantic_match(query, chunk.text)
#             result = SearchResult(
#                 text=chunk.text,
#                 source_path=chunk.source_path,
#                 document_name=chunk.document_name,
#                 score=score,
#                 mode="semantic",
#                 detail=explanation,
#                 chunk=chunk,
#                 rank=i + 1
#             )
#             results.append(result)

#         return results

#     def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
#         return float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

#     def _explain_semantic_match(self, query: str, chunk_text: str) -> str:
#         try:
#             prompt = (
#                 "Explain in 3 sentences how the following passage is semantically related to the query.\n\n"
#                 f"Query: {query}\n\n"
#                 f"Passage: {chunk_text}\n\n"
#                 "Explanation:"
#             )
#             response = ollama.generate(model="llama3", prompt=prompt)
#             return response['response'].strip()
#         except Exception as e:
#             return f"[Explanation unavailable: {str(e)}]"
# from models.data_models import Chunk, SearchResult
# from typing import List
# import numpy as np
# import ollama

# class SemanticSearchEngine:
#     def __init__(self):
#         self.embedder = None
#         self.query_vec = None

#     def set_query_embedding(self, query_embedding):
#         # Ensure the embedding is a 1D vector
#         if isinstance(query_embedding, list) and isinstance(query_embedding[0], list):
#             self.query_vec = np.array(query_embedding[0])
#         elif isinstance(query_embedding, list):
#             self.query_vec = np.array(query_embedding)
#         else:
#             raise ValueError("Invalid query embedding format")

#         if self.query_vec.ndim != 1:
#             raise ValueError(f"Query embedding must be a 1D vector, got shape {self.query_vec.shape}")

#         # Prevent division by zero in cosine similarity
#         if np.linalg.norm(self.query_vec) == 0:
#             raise ValueError("Query embedding has zero magnitude, cannot compute cosine similarity")

#     def search(self, query: str, chunks: List[Chunk], top_k: int = 5) -> List[SearchResult]:
#         if self.query_vec is None:
#             raise RuntimeError("Query embedding must be set using set_query_embedding() before calling search")

#         scored = []

#         for chunk in chunks:
#             if not chunk.text or not hasattr(chunk, "embedding") or not chunk.embedding:
#                 continue

#             vec = np.array(chunk.embedding)
#             if vec.ndim != 1:
#                 print(f"[Warning] Skipping chunk due to invalid embedding shape: {vec.shape}")
#                 continue
#             if vec.shape != self.query_vec.shape:
#                 print(f"[Warning] Skipping chunk due to embedding shape mismatch: {vec.shape} vs {self.query_vec.shape}")
#                 continue
#             if np.linalg.norm(vec) == 0:
#                 print("[Warning] Skipping chunk due to zero magnitude embedding")
#                 continue

#             score = self._cosine_similarity(self.query_vec, vec)
#             chunk.score = float(score)
#             scored.append((chunk, score))

#         top_chunks = sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]

#         results = []
#         for i, (chunk, score) in enumerate(top_chunks):
#             explanation = self._explain_semantic_match(query, chunk.text)
#             result = SearchResult(
#                 text=chunk.text,
#                 source_path=chunk.source_path,
#                 document_name=chunk.document_name,
#                 score=score,
#                 mode="semantic",
#                 detail=explanation,
#                 chunk=chunk,
#                 rank=i + 1
#             )
#             results.append(result)

#         return results

#     def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
#         denominator = np.linalg.norm(vec1) * np.linalg.norm(vec2)
#         if denominator == 0:
#             return 0.0
#         return float(np.dot(vec1, vec2) / denominator)

#     def _explain_semantic_match(self, query: str, chunk_text: str) -> str:
#         try:
#             prompt = (
#                 "Explain in 3 sentences how the following passage is semantically related to the query.\n\n"
#                 f"Query: {query}\n\n"
#                 f"Passage: {chunk_text}\n\n"
#                 "Explanation:"
#             )
#             response = ollama.generate(model="llama3", prompt=prompt)
#             return response['response'].strip()
#         except Exception as e:
#             return f"[Explanation unavailable: {str(e)}]"


# from models.data_models import Chunk, SearchResult
# from typing import List, Union
# import numpy as np
# import ollama

# class SemanticSearchEngine:
#     def __init__(self):
#         self.embedder = None
#         self.query_vec = None

#     def set_query_embedding(self, query_embedding: Union[str, List[float], List[List[float]]]):
#         """
#         Accepts either a raw query string or a precomputed embedding (list or nested list).
#         Automatically handles and sets the query vector for similarity computation.
#         """
#         if isinstance(query_embedding, str):
#             # Handle raw string query — embed it
#             if not self.embedder:
#                 raise RuntimeError("Embedder must be set before using string query")
#             response = self.embedder.get_embedding(query_embedding)
#             if isinstance(response, list) and len(response) == 1:
#                 query_embedding = response[0]
#             else:
#                 query_embedding = response

#         # Ensure the embedding is a 1D float array
#         if isinstance(query_embedding, list):
#             if len(query_embedding) == 1 and isinstance(query_embedding[0], list):
#                 query_embedding = query_embedding[0]
#             self.query_vec = np.array(query_embedding, dtype=np.float32)
#         elif isinstance(query_embedding, np.ndarray):
#             self.query_vec = query_embedding.astype(np.float32)
#         else:
#             raise ValueError("Invalid query embedding format")

#         if self.query_vec.ndim != 1:
#             raise ValueError(f"Query embedding must be a 1D vector, got shape {self.query_vec.shape}")

#         norm = np.linalg.norm(self.query_vec)
#         if norm == 0.0:
#             raise ValueError("Query embedding has zero magnitude, cannot compute cosine similarity")

#     # def search(self, query: str, chunks: List[Chunk], top_k: int = 5) -> List[SearchResult]:
#     #     if self.query_vec is None:
#     #         raise RuntimeError("Query embedding must be set using set_query_embedding() before calling search")

#     #     scored = []

#     #     for chunk in chunks:
#     #         if not chunk.text or not hasattr(chunk, "embedding") or not chunk.embedding:
#     #             continue

#     #         vec = np.array(chunk.embedding)
#     #         if vec.ndim != 1:
#     #             print(f"[Warning] Skipping chunk due to invalid embedding shape: {vec.shape}")
#     #             continue
#     #         if vec.shape != self.query_vec.shape:
#     #             print(f"[Warning] Skipping chunk due to embedding shape mismatch: {vec.shape} vs {self.query_vec.shape}")
#     #             continue
#     #         if np.linalg.norm(vec) == 0:
#     #             print("[Warning] Skipping chunk due to zero magnitude embedding")
#     #             continue

#     #         score = self._cosine_similarity(self.query_vec, vec)
#     #         chunk.score = float(score)
#     #         scored.append((chunk, score))

#     #     top_chunks = sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]

#     #     results = []
#     #     for i, (chunk, score) in enumerate(top_chunks):
#     #         explanation = self._explain_semantic_match(query, chunk.text)
#     #         result = SearchResult(
#     #             text=chunk.text,
#     #             source_path=chunk.source_path,
#     #             document_name=chunk.document_name,
#     #             score=score,
#     #             mode="semantic",
#     #             detail=explanation,
#     #             chunk=chunk,
#     #             rank=i + 1
#     #         )
#     #         results.append(result)

#     #     return results



#     def search(self, query: str, chunks: List[Chunk], top_k: int = 5) -> List[SearchResult]:
#         try:
#             query_vec = np.array(self.embedder.get_single_embedding(query))
#         except Exception as e:
#             print(f"[Error] Failed to embed query: {e}")
#             return []

#         scored = []

#         for chunk in chunks:
#             if not chunk.text or not hasattr(chunk, "embedding") or not chunk.embedding:
#                 continue

#             vec = np.array(chunk.embedding)
#             if vec.ndim != 1:
#                 print(f"[Warning] Skipping chunk due to invalid embedding shape: {vec.shape}")
#                 continue
#             if vec.shape != query_vec.shape:
#                 print(f"[Warning] Skipping chunk due to embedding shape mismatch: {vec.shape} vs {query_vec.shape}")
#                 continue
#             if np.linalg.norm(vec) == 0:
#                 print("[Warning] Skipping chunk due to zero magnitude embedding")
#                 continue

#             score = self._cosine_similarity(query_vec, vec)
#             chunk.score = float(score)
#             scored.append((chunk, score))

#         top_chunks = sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]

#         results = []
#         for i, (chunk, score) in enumerate(top_chunks):
#             explanation = self._explain_semantic_match(query, chunk.text)
#             results.append(SearchResult(
#                 text=chunk.text,
#                 source_path=chunk.source_path,
#                 document_name=chunk.document_name,
#                 score=score,
#                 mode="semantic",
#                 detail=explanation,
#                 chunk=chunk,
#                 rank=i + 1
#             ))

#         return results

#     def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
#         denominator = np.linalg.norm(vec1) * np.linalg.norm(vec2)
#         if denominator == 0:
#             return 0.0
#         return float(np.dot(vec1, vec2) / denominator)

#     def _explain_semantic_match(self, query: str, chunk_text: str) -> str:
#         try:
#             prompt = (
#                 "Explain in 3 sentences how the following passage is semantically related to the query.\n\n"
#                 f"Query: {query}\n\n"
#                 f"Passage: {chunk_text}\n\n"
#                 "Explanation:"
#             )
#             response = ollama.generate(model="llama3", prompt=prompt)
#             return response['response'].strip()
#         except Exception as e:
#             return f"[Explanation unavailable: {str(e)}]"


# from models.data_models import Chunk, SearchResult
# from typing import List, Union
# from embedding.embedding_manager import EmbeddingManager
# import numpy as np
# import ollama

# class SemanticSearchEngine:
#     def __init__(self):
#         self.embedder = EmbeddingManager.get_instance()

#     def search(self, query: str, chunks: List[Chunk], top_k: int = 5) -> List[SearchResult]:
#         try:
#             query_vec = np.array(self.embedder.get_single_embedding(query))
#         except Exception as e:
#             print(f"[Error] Failed to embed query: {e}")
#             return []

#         scored = []

#         for chunk in chunks:
#             if not chunk.text or not hasattr(chunk, "embedding") or not chunk.embedding:
#                 continue

#             vec = np.array(chunk.embedding)
#             if vec.ndim != 1:
#                 print(f"[Warning] Skipping chunk due to invalid embedding shape: {vec.shape}")
#                 continue
#             if vec.shape != query_vec.shape:
#                 print(f"[Warning] Skipping chunk due to embedding shape mismatch: {vec.shape} vs {query_vec.shape}")
#                 continue
#             if np.linalg.norm(vec) == 0:
#                 print("[Warning] Skipping chunk due to zero magnitude embedding")
#                 continue

#             score = self._cosine_similarity(query_vec, vec)
#             chunk.score = float(score)
#             scored.append((chunk, score))

#         top_chunks = sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]

#         results = []
#         for i, (chunk, score) in enumerate(top_chunks):
#             explanation = self._explain_semantic_match(query, chunk.text)
#             results.append(SearchResult(
#                 text=chunk.text,
#                 source_path=chunk.source_path,
#                 document_name=chunk.document_name,
#                 score=score,
#                 mode="semantic",
#                 detail=explanation,
#                 chunk=chunk,
#                 rank=i + 1
#             ))

#         return results

#     def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
#         denominator = np.linalg.norm(vec1) * np.linalg.norm(vec2)
#         if denominator == 0:
#             return 0.0
#         return float(np.dot(vec1, vec2) / denominator)

#     def _explain_semantic_match(self, query: str, chunk_text: str) -> str:
#         try:
#             prompt = (
#                 "Explain in 3 sentences how the following passage is semantically related to the query.\n\n"
#                 f"Query: {query}\n\n"
#                 f"Passage: {chunk_text}\n\n"
#                 "Explanation:"
#             )
#             response = ollama.generate(model="mistral", prompt=prompt)
#             return response['response'].strip()
#         except Exception as e:
#             return f"[Explanation unavailable: {str(e)}]"


from concurrent.futures import ThreadPoolExecutor
from typing import List
import numpy as np
from models.data_models import Chunk, SearchResult
from embedding.embedding_manager import EmbeddingManager
import ollama

class SemanticSearchEngine:
    def __init__(self):
        self.embedder = EmbeddingManager.get_instance()

    def _cosine_similarity(self, vec1: np.ndarray, vec2: np.ndarray) -> float:
        denominator = np.linalg.norm(vec1) * np.linalg.norm(vec2)
        if denominator == 0:
            return 0.0
        return float(np.dot(vec1, vec2) / denominator)

    def _explain_chunk(self, query: str, chunk_text: str) -> str:
        try:
            preview = chunk_text[:500]
            prompt = (
                "Explain in 3 sentences how the following passage is semantically related to the query.\n\n"
                f"Query: {query}\n\n"
                f"Passage: {preview}\n\n"
                "Explanation:"
            )
            response = ollama.generate(model="llama3", prompt=prompt)
            return response['response'].strip()
        except Exception as e:
            return f"[Explanation unavailable: {str(e)}]"

    def search(self, query: str, chunks: List[Chunk], top_k: int = 5) -> List[SearchResult]:
        try:
            query_vec = np.array(self.embedder.get_single_embedding(query))
        except Exception as e:
            print(f"[Error] Failed to embed query: {e}")
            return []

        scored = []
        for chunk in chunks:
            if not chunk.text or not hasattr(chunk, "embedding") or not chunk.embedding:
                continue

            vec = np.array(chunk.embedding)
            if vec.ndim != 1:
                print(f"[Warning] Skipping chunk due to invalid embedding shape: {vec.shape}")
                continue
            if vec.shape != query_vec.shape:
                print(f"[Warning] Skipping chunk due to embedding shape mismatch: {vec.shape} vs {query_vec.shape}")
                continue
            if np.linalg.norm(vec) == 0:
                print("[Warning] Skipping chunk due to zero magnitude embedding")
                continue

            score = self._cosine_similarity(query_vec, vec)
            chunk.score = float(score)
            scored.append((chunk, score))

        top_chunks = sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]

        # === Multithreaded Explanation Generation ===
        explanations = []
        with ThreadPoolExecutor(max_workers=top_k) as executor:
            futures = [executor.submit(self._explain_chunk, query, chunk.text) for chunk, _ in top_chunks]
            for future in futures:
                explanations.append(future.result())

        results = []
        for i, ((chunk, score), explanation) in enumerate(zip(top_chunks, explanations)):
            results.append(SearchResult(
                text=chunk.text,
                source_path=chunk.source_path,
                document_name=chunk.document_name,
                score=score,
                mode="semantic",
                detail=explanation,
                chunk=chunk,
                rank=i + 1
            ))

        return results

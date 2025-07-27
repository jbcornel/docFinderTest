# # semantic_cluster_chunker.py

# from transformers import AutoTokenizer
# from sklearn.cluster import KMeans
# from nltk.tokenize import sent_tokenize
# from typing import List
# import numpy as np
# import nltk

# from .base_chunker import BaseChunker
# from embedding.ollama_embedder import OllamaEmbedder

# nltk.download('punkt')

# class SemanticClusterChunker(BaseChunker):
#     def __init__(self, chunk_size=300, n_clusters=5):
#         self.embedder = OllamaEmbedder()
#         self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2", local_files_only=True)
#         self.chunk_size = chunk_size
#         self.n_clusters = n_clusters
#         self.max_token_length = 512  # Hard limit for transformer models

#     def chunk(self, text: str) -> List[str]:
#         sentences = sent_tokenize(text)

#         if len(sentences) == 0:
#             return []

#         # Truncate long sentences for safety
#         safe_sentences = [
#             self._truncate_if_needed(s) for s in sentences
#         ]

#         # Get embeddings for each sentence using Ollama
#         embeddings = self.embedder.embed(safe_sentences)
#         embedding_matrix = np.vstack(embeddings)

#         # Adjust cluster count if fewer sentences than clusters
#         n_clusters = min(self.n_clusters, max(1, len(safe_sentences)))

#         # Cluster sentence embeddings
#         kmeans = KMeans(n_clusters=n_clusters, random_state=42)
#         labels = kmeans.fit_predict(embedding_matrix)

#         # Group sentences by cluster
#         clustered = {}
#         for idx, label in enumerate(labels):
#             clustered.setdefault(label, []).append(safe_sentences[idx])

#         # Reconstruct chunks
#         chunks = []
#         for group in clustered.values():
#             chunk = " ".join(group)
#             chunk = self._truncate_if_needed(chunk)

#             token_len = len(self.tokenizer.encode(chunk, add_special_tokens=False))
#             if token_len <= self.max_token_length:
#                 chunks.append(chunk)
#             else:
#                 truncated = self._split_long_text(chunk)
#                 # Make sure each split chunk also obeys the hard limit
#                 safe_chunks = [self._truncate_if_needed(c) for c in truncated]
#                 chunks.extend(safe_chunks)


#         return chunks

#     def _split_long_text(self, text: str) -> List[str]:
#         tokens = self.tokenizer.encode(text, add_special_tokens=False)
#         token_chunks = [tokens[i:i+self.max_token_length] for i in range(0, len(tokens), self.max_token_length)]

#         return [self.tokenizer.decode(chunk, skip_special_tokens=True) for chunk in token_chunks]

#     def _truncate_if_needed(self, text: str) -> str:
#         tokens = self.tokenizer.encode(text, add_special_tokens=False)
#         if len(tokens) > self.max_token_length:
#             tokens = tokens[:self.max_token_length]
#             return self.tokenizer.decode(tokens, skip_special_tokens=True)
#         return text


# from transformers import AutoTokenizer
# from sklearn.cluster import KMeans
# from nltk.tokenize import sent_tokenize
# from typing import List, Tuple
# import numpy as np
# import nltk
# import ollama

# from .base_chunker import BaseChunker
# from embedding.ollama_embedder import OllamaEmbedder

# nltk.download('punkt')

# class SemanticClusterChunker(BaseChunker):
#     def __init__(self, chunk_size=300, n_clusters=5):
#         self.embedder = OllamaEmbedder()
#         self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2", local_files_only=True)
#         self.chunk_size = chunk_size
#         self.n_clusters = n_clusters
#         self.max_token_length = 512  # Hard limit for transformer models

#     def chunk(self, text: str) -> List[Tuple[str, str]]:
#         sentences = sent_tokenize(text)

#         if len(sentences) == 0:
#             return []

#         # Truncate long sentences for safety
#         safe_sentences = [self._truncate_if_needed(s) for s in sentences]

#         # Get embeddings for each sentence using Ollama
#         embeddings = self.embedder.embed(safe_sentences)
#         embedding_matrix = np.vstack(embeddings)

#         # Adjust cluster count if fewer sentences than clusters
#         n_clusters = min(self.n_clusters, max(1, len(safe_sentences)))

#         # Cluster sentence embeddings
#         kmeans = KMeans(n_clusters=n_clusters, random_state=42)
#         labels = kmeans.fit_predict(embedding_matrix)

#         # Group sentences by cluster
#         clustered = {}
#         for idx, label in enumerate(labels):
#             clustered.setdefault(label, []).append(safe_sentences[idx])

#         # Reconstruct chunks
#         chunks_with_context = []
#         for group in clustered.values():
#             chunk = " ".join(group)
#             token_len = len(self.tokenizer.encode(chunk, add_special_tokens=False))
#             if token_len <= self.max_token_length:
#                 context = self._generate_contextual_description(chunk, text)
#                 chunks_with_context.append((chunk, context))
#             else:
#                 truncated_chunks = self._split_long_text(chunk)
#                 for c in truncated_chunks:
#                     context = self._generate_contextual_description(c, text)
#                     chunks_with_context.append((c, context))

#         return chunks_with_context

#     def _split_long_text(self, text: str) -> List[str]:
#         tokens = self.tokenizer.encode(text, add_special_tokens=False)
#         token_chunks = [tokens[i:i+self.max_token_length] for i in range(0, len(tokens), self.max_token_length)]

#         decoded_chunks = []
#         for chunk in token_chunks:
#             decoded = self.tokenizer.decode(chunk, skip_special_tokens=True)
#             safe_decoded = self._truncate_if_needed(decoded)
#             decoded_chunks.append(safe_decoded)
#         return decoded_chunks

#     def _truncate_if_needed(self, text: str) -> str:
#         tokens = self.tokenizer.encode(text, add_special_tokens=False)
#         if len(tokens) > self.max_token_length:
#             print("chunk truncated")
#             tokens = tokens[:self.max_token_length]
#             decoded = self.tokenizer.decode(tokens, skip_special_tokens=True)
#             recheck = self.tokenizer.encode(decoded, add_special_tokens=False)
#             if len(recheck) > self.max_token_length:
#                 decoded = self.tokenizer.decode(recheck[:self.max_token_length - 5], skip_special_tokens=True)
#             return decoded
#         return text

#     def _generate_contextual_description(self, chunk_text: str, full_document: str) -> str:
#         try:
#             prompt = (
#                 "<document>\n"
#                 f"{full_document}\n"
#                 "</document>\n"
#                 "Here is the chunk we want to situate within the whole document\n"
#                 "<chunk>\n"
#                 f"{chunk_text}\n"
#                 "</chunk>\n"
#                 "Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."
#             )
#             response = ollama.generate(model="llama3", prompt=prompt)
#             return response['response'].strip()
#         except Exception as e:
#             return f"[Explanation unavailable: {str(e)}]"


# from transformers import AutoTokenizer
# from sklearn.cluster import KMeans
# from nltk.tokenize import sent_tokenize
# from typing import List, Tuple
# import numpy as np
# import nltk
# import ollama

# from .base_chunker import BaseChunker
# from embedding.ollama_embedder import OllamaEmbedder

# nltk.download('punkt')

# class SemanticClusterChunker(BaseChunker):
#     def __init__(self, chunk_size=300, n_clusters=5):
#         self.embedder = OllamaEmbedder()
#         self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2", local_files_only=True)
#         self.chunk_size = chunk_size
#         self.n_clusters = n_clusters
#         self.max_token_length = 512  # Hard limit for transformer models

#     def chunk(self, text: str, source_path: str = None, document_name: str = None) -> List[Tuple[str, str]]:

#         sentences = sent_tokenize(text)

#         if len(sentences) == 0:
#             return []

#         # Do not truncate sentences — preserve full content
#         safe_sentences = sentences

#         # Get embeddings for each sentence using Ollama
#         embeddings = self.embedder.get_embedding(safe_sentences)
#         embedding_matrix = np.vstack(embeddings)

#         # Adjust cluster count if fewer sentences than clusters
#         n_clusters = min(self.n_clusters, max(1, len(safe_sentences)))

#         # Cluster sentence embeddings
#         kmeans = KMeans(n_clusters=n_clusters, random_state=42)
#         labels = kmeans.fit_predict(embedding_matrix)

#         # Group sentences by cluster
#         clustered = {}
#         for idx, label in enumerate(labels):
#             clustered.setdefault(label, []).append(safe_sentences[idx])

#         # Reconstruct chunks
#         chunks_with_context = []
#         for group in clustered.values():
#             initial_chunk = " ".join(group)
#             split_chunks = self._recursive_split_if_needed(initial_chunk)
#             for chunk in split_chunks:
#                 context = self._generate_contextual_description(chunk, text)
#                 chunks_with_context.append((chunk, context))

#         return chunks_with_context

#     def _recursive_split_if_needed(self, text: str) -> List[str]:
#         tokens = self.tokenizer.encode(text, add_special_tokens=False)
#         if len(tokens) <= self.max_token_length:
#             return [text]

#         # Split tokens into max-token chunks
#         token_chunks = [tokens[i:i+self.max_token_length] for i in range(0, len(tokens), self.max_token_length)]
#         chunks = []
#         for token_chunk in token_chunks:
#             decoded = self.tokenizer.decode(token_chunk, skip_special_tokens=True)
#             # Recheck and split recursively if still too long
#             if len(self.tokenizer.encode(decoded, add_special_tokens=False)) > self.max_token_length:
#                 chunks.extend(self._recursive_split_if_needed(decoded))
#             else:
#                 chunks.append(decoded)
#         return chunks

#     def _generate_contextual_description(self, chunk_text: str, full_document: str) -> str:
#         try:
#             prompt = (
#                 "<document>\n"
#                 f"{full_document}\n"
#                 "</document>\n"
#                 "Here is the chunk we want to situate within the whole document\n"
#                 "<chunk>\n"
#                 f"{chunk_text}\n"
#                 "</chunk>\n"
#                 "Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."
#             )
#             response = ollama.generate(model="llama3", prompt=prompt)
#             return response['response'].strip()
#         except Exception as e:
#             return f"[Explanation unavailable: {str(e)}]"


# from transformers import AutoTokenizer
# from sklearn.cluster import KMeans
# from nltk.tokenize import sent_tokenize
# from typing import List, Tuple
# import numpy as np
# import nltk
# import ollama

# from .base_chunker import BaseChunker
# from embedding.ollama_embedder import OllamaEmbedder

# nltk.download('punkt')

# class SemanticClusterChunker(BaseChunker):
#     def __init__(self, chunk_size=300, n_clusters=5):
#         self.embedder = OllamaEmbedder()
#         self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2", local_files_only=True)
#         self.chunk_size = chunk_size
#         self.n_clusters = n_clusters
#         self.max_token_length = 512  # Hard limit for transformer models

#     def chunk(self, text: str, source_path: str = None, document_name: str = None) -> List[Tuple[str, str]]:
#         sentences = sent_tokenize(text)

#         if len(sentences) == 0:
#             return []

#         safe_sentences = sentences

#         # Get embeddings for each sentence using Ollama
#         embeddings = self.embedder.get_embedding(safe_sentences)
#         embedding_matrix = np.vstack(embeddings)

#         # Adjust cluster count if fewer sentences than clusters
#         n_clusters = min(self.n_clusters, max(1, len(safe_sentences)))

#         # Cluster sentence embeddings
#         kmeans = KMeans(n_clusters=n_clusters, random_state=42)
#         labels = kmeans.fit_predict(embedding_matrix)

#         # Group sentences by cluster
#         clustered = {}
#         for idx, label in enumerate(labels):
#             clustered.setdefault(label, []).append(safe_sentences[idx])

#         # Reconstruct chunks
#         chunks_without_context = []
#         for group in clustered.values():
#             initial_chunk = " ".join(group)
#             split_chunks = self._recursive_split_if_needed(initial_chunk)
#             for chunk in split_chunks:
#                 chunks_without_context.append((chunk, None))

#         return chunks_without_context

#     def _recursive_split_if_needed(self, text: str) -> List[str]:
#         tokens = self.tokenizer.encode(text, add_special_tokens=False)
#         if len(tokens) <= self.max_token_length:
#             return [text]

#         # Split tokens into max-token chunks
#         token_chunks = [tokens[i:i+self.max_token_length] for i in range(0, len(tokens), self.max_token_length)]
#         chunks = []
#         for token_chunk in token_chunks:
#             decoded = self.tokenizer.decode(token_chunk, skip_special_tokens=True)
#             if len(self.tokenizer.encode(decoded, add_special_tokens=False)) > self.max_token_length:
#                 chunks.extend(self._recursive_split_if_needed(decoded))
#             else:
#                 chunks.append(decoded)
#         return chunks

#     # Original context generation method preserved but not used
#     def _generate_contextual_description(self, chunk_text: str, full_document: str) -> str:
#         try:
#             prompt = (
#                 "<document>\n"
#                 f"{full_document}\n"
#                 "</document>\n"
#                 "Here is the chunk we want to situate within the whole document\n"
#                 "<chunk>\n"
#                 f"{chunk_text}\n"
#                 "</chunk>\n"
#                 "Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."
#             )
#             response = ollama.generate(model="llama3", prompt=prompt)
#             return response['response'].strip()
#         except Exception as e:
#             return f"[Explanation unavailable: {str(e)}]"


# from transformers import AutoTokenizer
# from sklearn.cluster import KMeans
# from nltk.tokenize import sent_tokenize
# from typing import List, Tuple
# import numpy as np
# import nltk
# import ollama

# from .base_chunker import BaseChunker
# from embedding.ollama_embedder import OllamaEmbedder

# nltk.download('punkt')

# class SemanticClusterChunker(BaseChunker):
#     def __init__(self, chunk_size=300, n_clusters=5):
#         self.embedder = OllamaEmbedder()
#         self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2", local_files_only=True)
#         self.chunk_size = chunk_size
#         self.n_clusters = n_clusters
#         self.max_token_length = 512  # Hard limit for transformer models

#     def chunk(self, text: str, source_path: str = None, document_name: str = None) -> List[Tuple[str, str]]:
#         sentences = sent_tokenize(text)

#         if len(sentences) == 0:
#             return []

#         safe_sentences = sentences  # Preserve full content for embeddings

#         embeddings = self.embedder.get_embedding(safe_sentences)
#         embedding_matrix = np.vstack(embeddings)

#         n_clusters = min(self.n_clusters, max(1, len(safe_sentences)))

#         kmeans = KMeans(n_clusters=n_clusters, random_state=42)
#         labels = kmeans.fit_predict(embedding_matrix)

#         clustered = {}
#         for idx, label in enumerate(labels):
#             clustered.setdefault(label, []).append(safe_sentences[idx])

#         chunks_with_context = []
#         chunk_counter = 0

#         for group_id, group in clustered.items():
#             initial_chunk = " ".join(group)
#             split_chunks = self._recursive_split_if_needed(initial_chunk)
#             for chunk in split_chunks:
#                 chunk_counter += 1
#                 token_count = len(self.tokenizer.encode(chunk, add_special_tokens=False))
#                 print(f"[DEBUG] Cluster {group_id} → Chunk #{chunk_counter}")
#                 print(f"↳ Tokens: {token_count}")
#                 print(f"↳ Preview: {chunk[:100]}")
#                 chunks_with_context.append((chunk, None))  # No context to keep it lightweight

#         return chunks_with_context

#     def _recursive_split_if_needed(self, text: str) -> List[str]:
#         tokens = self.tokenizer.encode(text, add_special_tokens=False)
#         if len(tokens) <= self.max_token_length:
#             return [text]

#         token_chunks = [tokens[i:i+self.max_token_length] for i in range(0, len(tokens), self.max_token_length)]
#         chunks = []
#         for token_chunk in token_chunks:
#             decoded = self.tokenizer.decode(token_chunk, skip_special_tokens=True)
#             if len(self.tokenizer.encode(decoded, add_special_tokens=False)) > self.max_token_length:
#                 chunks.extend(self._recursive_split_if_needed(decoded))
#             else:
#                 chunks.append(decoded)
#         return chunks

#     # Original context generator (disabled for cost/testing ease)
#     # def _generate_contextual_description(self, chunk_text: str, full_document: str) -> str:
#     #     try:
#     #         prompt = (
#     #             "<document>\n"
#     #             f"{full_document}\n"
#     #             "</document>\n"
#     #             "Here is the chunk we want to situate within the whole document\n"
#     #             "<chunk>\n"
#     #             f"{chunk_text}\n"
#     #             "</chunk>\n"
#     #             "Please give a short succinct context to situate this chunk within the overall document for the purposes of improving search retrieval of the chunk. Answer only with the succinct context and nothing else."
#     #         )
#     #         response = ollama.generate(model="llama3", prompt=prompt)
#     #         return response['response'].strip()
#     #     except Exception as e:
#     #         return f"[Explanation unavailable: {str(e)}]"
# chunking/recursive_token_chunker.py

from transformers import AutoTokenizer
from models.data_models import Chunk
from typing import List
import nltk

nltk.download("punkt")

class RecursiveTokenChunker:
    def __init__(self, chunk_size=300, max_token_length=512):
        self.tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2",
            local_files_only=True
        )
        self.chunk_size = chunk_size
        self.max_token_length = max_token_length

    def chunk(self, text: str, source_path: str = "", document_name: str = "") -> List[Chunk]:
        tokens = self.tokenizer.encode(text, add_special_tokens=False)
        chunks = []
        i = 0

        while i < len(tokens):
            end = min(i + self.chunk_size, len(tokens))
            token_slice = tokens[i:end]

            if len(token_slice) == 0:
                break

            # Print and skip if somehow still oversized
            if len(token_slice) > self.max_token_length:
                print(f"[❌ Oversized Slice Detected] Tokens: {len(token_slice)} at index {i}")
                print(f"[Tokens]: {token_slice[:30]}...")
                i += self.chunk_size
                continue

            decoded = self.tokenizer.decode(token_slice, skip_special_tokens=True)

            chunks.append(Chunk(
                text=decoded,
                source_path=source_path,
                document_name=document_name,
                embedding=None,
                context=""
            ))

            print(f"[✅ Chunk Created] Tokens: {len(token_slice)} | Preview: {decoded[:80]}...")
            i += self.chunk_size

        print(f"[🏁 Chunking Done] Total: {len(chunks)}")
        return chunks

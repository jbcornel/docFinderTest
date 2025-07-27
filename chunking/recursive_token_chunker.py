
# from transformers import AutoTokenizer
# from typing import List
# from .base_chunker import BaseChunker
# from models.data_models import Chunk  

# class RecursiveTokenChunker(BaseChunker):
#     def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", chunk_size: int = 300):
#         self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
#         self.max_token_length = self.tokenizer.model_max_length
#         self.chunk_size = min(chunk_size, self.max_token_length)
#         self.separators = ["\n\n", "\n", ".", "!", "?", ",", " "]

#     def chunk(self, text: str, source_path: str = "", document_name: str = "") -> List[Chunk]:
#         chunks = self._recursive_split(text, self.separators)
#         validated_chunks = []
#         for chunk_text in chunks:
#             for token_limited in self._enforce_max_token_limit(chunk_text):
#                 validated_chunks.append(Chunk(
#                     text=token_limited,
#                     source_path=source_path,
#                     document_name=document_name
#                 ))
#         return validated_chunks

#     def _safe_encode(self, text: str) -> List[int]:
#         return self.tokenizer.encode(
#             text,
#             add_special_tokens=False,
#             truncation=True,
#             max_length=self.max_token_length
#         )

#     def _enforce_max_token_limit(self, text: str) -> List[str]:
#         tokens = self._safe_encode(text)
#         if len(tokens) <= self.chunk_size:
#             return [text]
#         token_chunks = [tokens[i:i + self.chunk_size] for i in range(0, len(tokens), self.chunk_size)]
#         return [self.tokenizer.decode(chunk, skip_special_tokens=True) for chunk in token_chunks]

#     def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
#         if not separators:
#             return self._enforce_max_token_limit(text)

#         sep = separators[0]
#         splits = text.split(sep)

#         chunks = []
#         current_chunk = ""

#         for part in splits:
#             trial_chunk = current_chunk + part + sep
#             num_tokens = len(self._safe_encode(trial_chunk))

#             if num_tokens <= self.chunk_size:
#                 current_chunk = trial_chunk
#             else:
#                 if current_chunk.strip():
#                     chunks.append(current_chunk.strip())
#                 current_chunk = part + sep

#         if current_chunk.strip():
#             chunks.append(current_chunk.strip())

#         refined_chunks = []
#         for chunk in chunks:
#             token_len = len(self._safe_encode(chunk))
#             if token_len > self.chunk_size:
#                 refined_chunks.extend(self._recursive_split(chunk, separators[1:]))
#             else:
#                 refined_chunks.append(chunk)

#         return refined_chunks
from transformers import AutoTokenizer
from typing import List
from models.data_models import Chunk
from .base_chunker import BaseChunker
#cl100k_base
class RecursiveTokenChunker(BaseChunker):
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", chunk_size: int = 300):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, local_files_only=True)
        self.max_token_length = self.tokenizer.model_max_length
        self.chunk_size = min(chunk_size, self.max_token_length)
        self.separators = ["\n\n", "\n", ".", "!", "?", ",", " "]

    def chunk(self, text: str, source_path: str = "", document_name: str = "") -> List[Chunk]:
        chunks = self._recursive_split(text, self.separators)
        validated_chunks = []
        for idx, chunk_text in enumerate(chunks):
            for token_limited in self._enforce_max_token_limit(chunk_text):
                token_count = len(self._safe_encode(token_limited))
               # print(f"[DEBUG] Recursive Chunk #{idx + 1} — Tokens: {token_count}")
               # print(f"↳ Preview: {token_limited[:100]}")
                validated_chunks.append(Chunk(
                    text=token_limited,
                    source_path=source_path,
                    document_name=document_name
                ))
        return validated_chunks

    def _safe_encode(self, text: str) -> List[int]:
        return self.tokenizer.encode(
            text,
            add_special_tokens=False,
            truncation=True,
            max_length=self.max_token_length
        )

    def _enforce_max_token_limit(self, text: str) -> List[str]:
        tokens = self._safe_encode(text)
        if len(tokens) <= self.chunk_size:
            return [text]
        token_chunks = [tokens[i:i + self.chunk_size] for i in range(0, len(tokens), self.chunk_size)]
        return [self.tokenizer.decode(chunk, skip_special_tokens=True) for chunk in token_chunks]

    def _recursive_split(self, text: str, separators: List[str]) -> List[str]:
        if not separators:
            return self._enforce_max_token_limit(text)

        sep = separators[0]
        splits = text.split(sep)

        chunks = []
        current_chunk = ""

        for part in splits:
            trial_chunk = current_chunk + part + sep
            num_tokens = len(self._safe_encode(trial_chunk))

            if num_tokens <= self.chunk_size:
                current_chunk = trial_chunk
            else:
                if current_chunk.strip():
                    chunks.append(current_chunk.strip())
                current_chunk = part + sep

        if current_chunk.strip():
            chunks.append(current_chunk.strip())

        refined_chunks = []
        for chunk in chunks:
            token_len = len(self._safe_encode(chunk))
            if token_len > self.chunk_size:
                refined_chunks.extend(self._recursive_split(chunk, separators[1:]))
            else:
                refined_chunks.append(chunk)

        return refined_chunks

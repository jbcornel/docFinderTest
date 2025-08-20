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

            
            if len(token_slice) > self.max_token_length:
                print(f"[ Oversized Slice Detected] Tokens: {len(token_slice)} at index {i}")
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

            print(f"[ Chunk Created] Tokens: {len(token_slice)} | Preview: {decoded[:80]}...")
            i += self.chunk_size

        print(f"[Chunking Done] Total: {len(chunks)}")
        return chunks

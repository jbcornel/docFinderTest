
import os
os.environ["OLLAMA_ORCH_ACCEL"] = "1"
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoTokenizer
import numpy as np
import ollama
from typing import List, Union

MAX_TOKENS = 512
THREAD_BATCH_SIZE = 8

class OllamaEmbedder:
    def __init__(self, model_name: str = "nomic-embed-text"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2",
            local_files_only=True
        )

    def get_embedding(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
        if isinstance(texts, str):
            texts = [texts]

        def embed_text(text: str):
            try:
                start = time.time()
                response = ollama.embeddings(model=self.model_name, prompt=text)
                duration = time.time() - start
                emb = response.get("embedding")
                if emb is None:
                    raise ValueError("No embedding returned")
                return emb, duration
            except Exception as e:
                print(f"[Error] Embedding failed for text: {text[:30]}... → {e}")
                return [0.0] * 1024, 0.0

        embeddings = []
        total_time = 0.0
        count = 0

        print(f"[Embedding] Starting with {len(texts)} items using {THREAD_BATCH_SIZE} threads...")

        with ThreadPoolExecutor(max_workers=THREAD_BATCH_SIZE) as executor:
            futures = {executor.submit(embed_text, text): text for text in texts}
            for idx, future in enumerate(as_completed(futures), 1):
                result, duration = future.result()
                embeddings.append(result)
                total_time += duration
                count += 1
                print(f"  → [{count}/{len(texts)}] Done in {duration:.2f}s | Avg speed: {count / total_time:.2f} embeds/sec")

        print(f"[ Complete] {count} embeddings in {total_time:.2f}s → {count / total_time:.2f} embeds/sec")
        return embeddings

    def get_single_embedding(self, text: str) -> List[float]:
        try:
            response = ollama.embeddings(model=self.model_name, prompt=text)
            emb = response["embedding"]
            print(f"[OllamaEmbedder] Single embedding length: {len(emb)}")
            return emb
        except Exception as e:
            print(f"[Error] Single embedding failed: {e}")
            return [0.0] * 1024

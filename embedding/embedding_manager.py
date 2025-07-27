# # embedding/embedding_manager.py
# from embedding.ollama_embedder import OllamaEmbedder

# class EmbeddingManager:
#     _instance = None

#     def __init__(self, model_name: str = "nomic-embed-text"):
#         if EmbeddingManager._instance is not None:
#             raise RuntimeError("EmbeddingManager is a singleton! Use get_instance()")
#         self.embedder = OllamaEmbedder(model_name=model_name)
#         self.model_name = model_name
#         EmbeddingManager._instance = self
#         print(f"[EmbeddingManager] Initialized with model: {model_name}")
# #mxbai-embed-large
#     @classmethod
#     def get_instance(cls, model_name: str = "nomic-embed-text"):
#         if cls._instance is None:
#             cls(model_name)
#         return cls._instance

#     def get_single_embedding(self, text: str):
#         print(f"[EmbeddingManager] Embedding query: '{text[:30]}...' with model: {self.model_name}")
#         return self.embedder.get_single_embedding(text)

#     def get_embedding(self, texts):
#         print(f"[EmbeddingManager] Embedding batch of {len(texts)} texts with model: {self.model_name}")
#         return self.embedder.get_embedding(texts)


# embedding/embedding_manager.py
import time
from embedding.ollama_embedder import OllamaEmbedder

class EmbeddingManager:
    _instance = None
#all-minilm
#nomic-embed-text
#mxbai-embed-large
    def __init__(self, model_name: str = "all-minilm"):
        if EmbeddingManager._instance is not None:
            raise RuntimeError("EmbeddingManager is a singleton! Use get_instance()")
        self.embedder = OllamaEmbedder(model_name=model_name)
        self.model_name = model_name
        EmbeddingManager._instance = self
        print(f"[EmbeddingManager] Initialized with model: {model_name}")

    @classmethod
    def get_instance(cls, model_name: str = "all-minilm"):
        if cls._instance is None:
            cls(model_name)
        return cls._instance

    def get_single_embedding(self, text: str):
        print(f"[EmbeddingManager] Embedding query: '{text[:30]}...' with model: {self.model_name}")
        return self.embedder.get_single_embedding(text)


    #batch embed 
    def get_embedding(self, texts):
        print(f"[EmbeddingManager] Embedding {len(texts)} texts in parallel using model: {self.model_name}")
        return self.embedder.get_embedding(texts)

    # def get_embedding(self, texts):
    #     print(f"[EmbeddingManager] Embedding batch of {len(texts)} texts with model: {self.model_name}")
    #     embeddings = []

    #     for idx, text in enumerate(texts):
    #         try:
    #             start = time.time()
    #             embedding = self.embedder.get_single_embedding(text)
    #             elapsed = time.time() - start
    #             print(f"[{idx + 1}/{len(texts)}] ✅ Embedded ({elapsed:.2f}s) | Preview: {text[:80]}...")
    #             embeddings.append(embedding)
    #         except Exception as e:
    #             print(f"[{idx + 1}/{len(texts)}] ❌ Failed to embed text: {text[:100]}")
    #             print(f"   ↳ Error: {str(e)}\n")
    #             embeddings.append([])  # Optional: Append empty embedding or skip

    #     return embeddings
#normal embed 



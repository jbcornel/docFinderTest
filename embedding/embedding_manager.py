import time
from embedding.ollama_embedder import OllamaEmbedder

class EmbeddingManager:
    _instance = None

    def __init__(self, model_name: str = "nomic-embed-text"):
        if EmbeddingManager._instance is not None:
            raise RuntimeError("EmbeddingManager is a singleton! Use get_instance()")
        self.embedder = OllamaEmbedder(model_name=model_name)
        self.model_name = model_name
        EmbeddingManager._instance = self
        print(f"[EmbeddingManager] Initialized with model: {model_name}")

    @classmethod
    def get_instance(cls, model_name: str = "nomic-embed-text"):
        if cls._instance is None:
            cls(model_name)
        return cls._instance

    def get_single_embedding(self, text: str):
        print(f"[EmbeddingManager] Embedding query: '{text[:30]}...' with model: {self.model_name}")
        return self.embedder.get_single_embedding(text)



    def get_embedding(self, texts):
        print(f"[EmbeddingManager] Embedding {len(texts)} texts in parallel using model: {self.model_name}")
        return self.embedder.get_embedding(texts)


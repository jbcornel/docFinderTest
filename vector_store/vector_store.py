import chromadb
from chromadb.config import Settings

class VectorStoreManager:
    _instances = {}

    def __init__(self, persist_dir, collection_name):
        self.client = chromadb.Client(Settings(
            chroma_db_impl="duckdb+parquet",
            persist_directory=persist_dir
        ))
        self.collection = self.client.get_or_create_collection(name=collection_name)
        print(f"[VectorStoreManager] Initialized collection '{collection_name}' at '{persist_dir}'")

    @classmethod
    def get_instance(cls, mode):
        if mode not in cls._instances:
            if mode == "semantic":
                cls._instances[mode] = cls("./chroma_store/semantic", "semantic_chunks")
            elif mode == "exact":
                cls._instances[mode] = cls("./chroma_store/exact", "exact_chunks")
            else:
                raise ValueError(f"Unknown mode: {mode}")
        return cls._instances[mode]

    def get_collection(self):
        return self.collection

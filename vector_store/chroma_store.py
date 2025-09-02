import os
import hashlib
from typing import List, Dict, Any, Tuple, Optional

import chromadb
from chromadb.config import Settings

from models.data_models import Chunk
from embedding.embedding_manager import EmbeddingManager


def _chunk_id(chunk: "Chunk", tie_breaker: Optional[str] = None) -> str:
  
    h = hashlib.sha256()
    h.update((chunk.source_path or "").encode("utf-8", errors="ignore"))
    h.update(b"||")
    h.update((chunk.document_name or "").encode("utf-8", errors="ignore"))
    h.update(b"||")
    # full text hash that prevents same-prefix collisions
    h.update(hashlib.sha256((chunk.text or "").encode("utf-8", errors="ignore")).digest())
  
    ordinal = getattr(chunk, "ordinal", None)
    if ordinal is not None:
        h.update(b"||")
        h.update(str(int(ordinal)).encode("utf-8"))
        # if a collision occurs
    if tie_breaker:
        h.update(b"||")
        h.update(tie_breaker.encode("utf-8"))
    return h.hexdigest()[:32]


class ChromaVectorStore:
    
    def __init__(self, persist_dir: str = "./chroma_store/semantic", collection_name: str = "docfinder_semantic"):
        os.makedirs(persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=persist_dir, settings=Settings(allow_reset=False))
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        self.embedder = EmbeddingManager.get_instance()

    def upsert_chunks(self, chunks: List[Chunk]) -> int:
        if not chunks:
            return 0

        ids: List[str] = []
        docs: List[str] = []
        metas: List[Dict[str, Any]] = []
        embs: List[Optional[List[float]]] = []

        
        seen_ids = set()

        # Track indices that still require embeddings
        to_embed_idx: List[int] = []
        to_embed_texts: List[str] = []

        for i, ch in enumerate(chunks):
            # skip empty/whitespace-only chunks
            if not isinstance(ch.text, str) or not ch.text.strip():
                continue


            cid = _chunk_id(ch)
            if cid in seen_ids:
                
                cid = _chunk_id(ch, tie_breaker=str(i))
                if cid in seen_ids:
                   
                    continue
            seen_ids.add(cid)

            ids.append(cid)
            docs.append(ch.text)
            metas.append({
                "source_path": ch.source_path,
                "document_name": ch.document_name,
                "summary": getattr(ch, "summary", "") or "",
            })

            if getattr(ch, "embedding", None):
                embs.append(ch.embedding)
            else:
                embs.append(None)
                
                to_embed_idx.append(len(embs) - 1)
                to_embed_texts.append(ch.text)

    
        if to_embed_texts:
            new_embs = self.embedder.get_embedding(to_embed_texts)
            for slot, vec in zip(to_embed_idx, new_embs):
                embs[slot] = vec

        # Filter out any items that still lack an embedding if occurs, edge case
        prepared = [(i, d, m, e) for i, d, m, e in zip(ids, docs, metas, embs) if isinstance(e, list)]
        if not prepared:
            return 0

        #batch upserts to avoid Chroma's max per-call limit (≈5461) ---
        MAX_BATCH = 4000  
        total = 0
        for start in range(0, len(prepared), MAX_BATCH):
            chunk_batch = prepared[start:start + MAX_BATCH]

            # remove duplicate ID if they occur within this batch, last one wins
            by_id = {}
            for _id, _doc, _meta, _emb in chunk_batch:
                by_id[_id] = (_doc, _meta, _emb)

            batch_ids = list(by_id.keys())
            batch_docs = [by_id[k][0] for k in batch_ids]
            batch_metas = [by_id[k][1] for k in batch_ids]
            batch_embs = [by_id[k][2] for k in batch_ids]

            self.collection.upsert(
                ids=batch_ids,
                documents=batch_docs,
                metadatas=batch_metas,
                embeddings=batch_embs,
            )
            total += len(batch_ids)

        return total


    def query(self, query_text: str, top_k: int = 5) -> List[Tuple[Chunk, float]]:
        if not query_text or not query_text.strip():
            return []
        q_emb = self.embedder.get_embedding([query_text])[0]
        res = self.collection.query(
            query_embeddings=[q_emb],
            n_results=max(1, int(top_k)),
            include=["documents", "metadatas", "distances", "embeddings"],
        )
        docs = (res.get("documents") or [[]])[0]
        metas = (res.get("metadatas") or [[]])[0]
        dists = (res.get("distances") or [[]])[0]
        embs = (res.get("embeddings") or [[]])[0]

        out: List[Tuple[Chunk, float]] = []
        for doc, meta, dist, emb in zip(docs, metas, dists, embs):
            if not isinstance(doc, str):
                continue
            ch = Chunk(
                text=doc,
                source_path=(meta or {}).get("source_path", ""),
                document_name=(meta or {}).get("document_name", ""),
                embedding=emb,
                summary=(meta or {}).get("summary", ""),
            )
            sim = 1.0 - float(dist) if dist is not None else 0.0
            out.append((ch, sim))
        return out

    def update_summary(self, chunk: Chunk, new_summary: str) -> None:
        try:
            self.collection.update(
                ids=[_chunk_id(chunk)],
                metadatas=[{
                    "source_path": chunk.source_path,
                    "document_name": chunk.document_name,
                    "summary": new_summary or "",
                }],
            )
        except Exception:
            pass

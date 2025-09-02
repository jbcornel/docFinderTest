from output.result_formatter import ResultFormatter
from vector_store.chroma_store import ChromaVectorStore
from models.data_models import Chunk, SearchResult
from chunking.recursive_token_chunker import RecursiveTokenChunker
from search.semantic_search_engine import SemanticSearchEngine
from search.exact_search_engine import ExactSearchEngine
from embedding.embedding_manager import EmbeddingManager
from summarization.summarizer import Summarizer
from chunking.semantic_chunker import SemanticChunker 
from chunking.llm_semantic_chunker_ollama import LLMSemanticChunkerOllama
from chunking.semantic_cluster_chunker import SemanticClusterChunker
from search.reranker import Reranker  # optional reranker

from typing import List
import argparse
import os
import json
import shutil


# summary model choices (alias -> concrete model id)
SUMMARY_MODEL_MAP = {
    "phi3": "phi3:mini",
    "llama3": "llama3:latest",
}

CHUNKER_STATE = {"name": "recursive"}


def ingest_phase(directory: str, mode: str = "semantic", append: bool = False):
    print("[Phase 1: Ingesting, Chunking, and Embedding Documents]")

    # choose chunker
    if CHUNKER_STATE.get("name") == "recursive":
        chunker = RecursiveTokenChunker()
    elif CHUNKER_STATE.get("name") == "semantic":
        chunker = SemanticChunker()
    elif CHUNKER_STATE.get("name") == "semantic-llm":
        chunker = LLMSemanticChunkerOllama(
            model_name="phi3:mini",
            temperature=0.2,
            base_chunk_size=50,
            base_chunk_overlap=0,
            debug=True,
            window_token_budget=800,
            max_retries=3,
        )
    elif CHUNKER_STATE.get("name") == "semantic-cluster":
        chunker = SemanticClusterChunker(
            similarity_threshold=0.62,
            max_tokens=600,
            min_sentences=1,
            debug=False,
        )
    else:
        raise ValueError(f"Unknown chunker option: {CHUNKER_STATE.get('name')}")

    all_chunks: List[Chunk] = []

    filepaths = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith(".txt") or f.endswith(".md")
    ]

    if not filepaths:
        print(f"[WARN] No .txt/.md files found in: {os.path.abspath(directory)}")

    for filepath in filepaths:
        print(f"Processing file: {os.path.basename(filepath)}")
        with open(filepath, encoding="utf-8", errors="ignore") as f:
            text = f.read()

        # create array of chunks using chosen chunker
        chunks = chunker.chunk(
            text,
            source_path=filepath,
            document_name=os.path.basename(filepath),
        )

        # assign id per chunk
        for idx, ch in enumerate(chunks):
            setattr(ch, "ordinal", idx)

        all_chunks.extend(chunks)

    # Embeddings done through singleton instance
    embedder = EmbeddingManager.get_instance()
    texts = [chunk.text for chunk in all_chunks if chunk.text and isinstance(chunk.text, str)]
    text_chunk_map = [chunk for chunk in all_chunks if chunk.text and isinstance(chunk.text, str)]

    if texts:
        try:
            embeddings = embedder.get_embedding(texts)
            if not isinstance(embeddings, list) or not all(isinstance(e, list) for e in embeddings):
                raise ValueError("Invalid embedding format")
            if len(embeddings) != len(text_chunk_map):
                raise ValueError("Mismatch between embeddings and chunks")
            for chunk, embedding in zip(text_chunk_map, embeddings):
                chunk.embedding = embedding
        except Exception as e:
            print(f"[Error] Batch embedding failed: {e}")

    if mode == "semantic":
        vs = ChromaVectorStore(persist_dir="./chroma_store/semantic", collection_name="docfinder_semantic")
        added = vs.upsert_chunks(all_chunks)
        print(f"[INFO] Upserted {added} semantic chunks into Chroma at ./chroma_store/semantic")
    else:
        existing_chunks: List[Chunk] = []
        path = "vector_store_exact.json"
        if append and os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                existing_chunks = [Chunk.from_dict(data) for data in json.load(f)]

        combined_chunks = existing_chunks + all_chunks

        with open(path, "w", encoding="utf-8") as f:
            json.dump([chunk.to_dict() for chunk in combined_chunks], f, ensure_ascii=False, indent=2)

        print(f"[INFO] Saved {len(combined_chunks)} total chunks with embeddings to {path}")


def _summarize_results_in_place(
    results: List[SearchResult],
    n: int,
    model_choice: str,
    workers: int,
    persist_semantic: bool,
):
   
    n = max(0, min(5, int(n)))
    if n <= 0 or not results:
        return

    texts = [r.text for r in results[:n]]
    resolved_model = SUMMARY_MODEL_MAP.get(model_choice, model_choice)
    print(f"[Summarizer] Model: {resolved_model} | items: {len(texts)} | workers: {min(workers, len(texts))}")

    summarizer = Summarizer(
        model_name=resolved_model,
        workers=min(workers, len(texts)),
        max_tokens=96,
        temperature=0.0,
    )
    try:
        outs = summarizer.batch(texts)
    except Exception as e:
        print(f"[Warn] Query-time summarization failed: {e}")
        outs = ["[Summary unavailable]"] * len(texts)

    # write summaries into results 
    vs = None
    if persist_semantic:
        vs = ChromaVectorStore(persist_dir="./chroma_store/semantic", collection_name="docfinder_semantic")

    for i, s in enumerate(outs):
        summary = (s or "").strip() or "[Summary unavailable]"
        r = results[i]
        r.detail = summary
        if r.chunk is not None:
            r.chunk.summary = summary
        if vs is not None and r.chunk is not None:
            try:
                vs.update_summary(r.chunk, summary)
            except Exception as e:
                print(f"[Warn] Failed to update summary in vector store: {e}")

    
    for r in results[:n]:
        if r.detail and r.detail.strip():
            if r.meta is None:
                r.meta = {}
            r.meta["_had_summary"] = True


def query_phase(
    query: str,
    mode: str = "semantic",
    fuzzy: bool = False,
    summarize_at_query: bool = False,  # if False, we won't generate summaries at all
    summarize_workers: int = 5,
    summaries: int = 2,
    top_k: int = 5,
    summary_model_choice: str = "llama3",
    use_reranker: bool = False,
):
    print("\n[Phase 2: Performing Query/Retrieval]")

    
    if mode == "semantic":  
        resolved_model = SUMMARY_MODEL_MAP.get(summary_model_choice, summary_model_choice)
        engine = SemanticSearchEngine(
            summary_model=resolved_model,            
            max_workers=min(5, max(1, int(summarize_workers))),
            persist_dir="./chroma_store/semantic",
            collection_name="docfinder_semantic",
        )
       
        results: List[SearchResult] = engine.search_store(
            query=query,
            top_k=max(1, int(top_k)),
            summarize_topk=0,
        )

    else:  # if exact search
        path = "vector_store_exact.json"
        if not os.path.exists(path):
            print(f"[Error] Vector store not found at {path}. Run --ingest first.")
            return
        with open(path, "r", encoding="utf-8") as f:
            chunks_data = json.load(f)
        chunks: List[Chunk] = [Chunk.from_dict(data) for data in chunks_data]
        if not chunks:
            print("[Error] No chunks available.")
            return

        engine = ExactSearchEngine()
        results: List[SearchResult] = engine.search(query, chunks, fuzzy=fuzzy)

    # optional reranking 
    if use_reranker and results:
        try:
            reranker = Reranker() 
            results = reranker.rerank(query, results)
            print("[INFO] Reranker applied.")
        except Exception as e:
            print(f"[WARN] Reranker failed: {e}")

    if not results:
        print("\n[No results to display]")
        return

   
    if summarize_at_query:
        _summarize_results_in_place(
            results=results,
            n=summaries,
            model_choice=summary_model_choice,
            workers=summarize_workers,
            persist_semantic=(mode == "semantic"),
        )


    summarized_n = max(0, min(5, int(summaries))) if summarize_at_query else 0
    for i, r in enumerate(results):
        d = (r.detail or "").strip()
        if i < summarized_n and summarize_at_query:
            if not d:
                r.detail = "[No explanation required]"
        else:
            if d:
                r.detail = f"[No explanation required] | {d}"
            else:
                r.detail = "[No explanation required]"

    print("\nTop Results:")
    ResultFormatter.print_results(results)


def purge_vector_store():
    # purge exact search vector json store
    for file in ["vector_store_exact.json"]:
        if os.path.exists(file):
            os.remove(file)
            print(f"[INFO] {file} purged.")
        else:
            print(f"[INFO] {file} not found.")

    # purge entire chroma vector store 
    chroma_dir = "./chroma_store/"
    chroma_abs = os.path.abspath(chroma_dir)
    if os.path.isdir(chroma_dir):
        try:
            shutil.rmtree(chroma_dir)
            print(f"[INFO] Chroma store purged at {chroma_abs}.", flush=True)
        except Exception as e:
            print(f"[WARN] Could not purge Chroma store at {chroma_abs}: {e}", flush=True)
    else:
        if os.path.exists(chroma_dir):
            try:
                os.remove(chroma_dir)
                print(f"[INFO] Removed non-directory path at {chroma_abs}.", flush=True)
            except Exception as e:
                print(f"[WARN] Path exists but could not be removed at {chroma_abs}: {e}", flush=True)
        else:
            print(f"[INFO] Chroma store not found at {chroma_abs}.", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Local Document Search Tool",
        add_help=True,  # -h/--help prints all options
        epilog="Examples:\n"
               "  Ingest (semantic store)        : python3 main.py --ingest --chunker recursive\n"
               "  Ingest (exact store)           : python3 main.py --ingest --mode exact\n"
               "  Query semantic (no rerank)     : python3 main.py --query \"foo\" --mode semantic --topk 5 --summarize-at-query --summaries 2\n"
               "  Query semantic with rerank     : python3 main.py --query \"foo\" --mode semantic --rerank --summarize-at-query --summaries 2\n"
               "  Query exact (with fuzzy)       : python3 main.py --query \"foo\" --mode exact --fuzzy --summarize-at-query --summaries 2\n",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--ingest", action="store_true", help="Run ingestion phase")
    parser.add_argument("--purge", action="store_true", help="Purge the chunk cache")
    parser.add_argument("--dir", help="Directory of .txt/.md files to ingest", default="./docs")
    parser.add_argument("--query", help="Text query to perform search")
    parser.add_argument("--mode", help="Choose search mode: 'semantic' or 'exact'", default="semantic")
    parser.add_argument("--append", action="store_true", help="Append new chunks to existing vector store")
    parser.add_argument("--fuzzy", action="store_true", help="Enable fuzzy matching (only valid with --query and --mode exact)")
    parser.add_argument("--topk", type=int, default=5, help="How many results to retrieve overall (default 5)")
    parser.add_argument(
        "--summarize-at-query", action="store_true",
        help="Generate summaries for the first N results (summaries are computed after optional reranking)."
    )
    parser.add_argument(
        "--summaries", type=int, default=2,
        help="How many of the top results to summarize (1–5). Default: 2"
    )
    parser.add_argument(
        "--summarize-workers", type=int, default=5,
        help="Parallel workers for query-time summaries (cap at 5)"
    )
    parser.add_argument(
        "--summary-model",
        choices=["phi3", "llama3"],
        default="llama3",
        help="Model to use for summaries: | phi3 | llama3",
    )
    parser.add_argument(
        "--chunker",
        choices=["recursive", "semantic", "semantic-llm", "semantic-cluster"],
        default="recursive",
        help="Chunking strategy used during ingestion"
    )
    parser.add_argument(
        "--rerank", action="store_true",
        help="Apply embedding-based reranker after initial results"
    )

    args = parser.parse_args()
    CHUNKER_STATE["name"] = getattr(args, "chunker", "recursive")

    
    if args.fuzzy:
        if not args.query:
            parser.error("--fuzzy can only be used with --query.")
        if args.mode.lower() != "exact":
            parser.error("--fuzzy is only supported with --mode exact.")

    
    if args.purge:
        purge_vector_store()
    elif args.ingest:
        ingest_phase(args.dir, args.mode, append=args.append)
    elif args.query:
        summarize_at_query = bool(getattr(args, "summarize_at_query", False))
        summaries_n = max(1, min(5, int(getattr(args, "summaries", 2)))) if summarize_at_query else 0
        workers = max(1, min(5, int(getattr(args, "summarize_workers", 5))))
        top_k = max
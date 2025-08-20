from output.result_formatter import ResultFormatter
from vector_store.vector_store import VectorStoreManager 
from models.data_models import Chunk, SearchResult
from chunking.recursive_token_chunker import RecursiveTokenChunker
from search.semantic_search_engine import SemanticSearchEngine
from search.exact_search_engine import ExactSearchEngine
from embedding.embedding_manager import EmbeddingManager
from summarization.summarizer import Summarizer  

import argparse
import os
import json

#summary model choices
SUMMARY_MODEL_MAP = {
    "phi3": "phi3:mini",
    "llama3": "llama3:latest",
}



def ingest_phase(directory: str, mode: str = "semantic", append: bool = False):
    print("[Phase 1: Ingesting, Chunking, and Embedding Documents]")

    chunker = RecursiveTokenChunker()
    all_chunks = []

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

        chunks = chunker.chunk(
            text,
            source_path=filepath,
            document_name=os.path.basename(filepath),
        )
        all_chunks.extend(chunks)

    #Embeddings (once at ingestion) done through singleton instance
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

#persistent vector store 
    existing_chunks = []
    path = "vector_store_semantic.json" if mode == "semantic" else "vector_store_exact.json"
    if append and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            existing_chunks = [Chunk.from_dict(data) for data in json.load(f)]

    combined_chunks = existing_chunks + all_chunks

    with open(path, "w", encoding="utf-8") as f:
        json.dump([chunk.to_dict() for chunk in combined_chunks], f, ensure_ascii=False, indent=2)

    print(f"[INFO] Saved {len(combined_chunks)} total chunks with embeddings to {path}")



def query_phase(
    query: str,
    mode: str = "semantic",
    fuzzy: bool = False,
    summarize_at_query: bool = False,
    summarize_workers: int = 5,
    summaries: int = 2,
    top_k: int = 5,
    summary_model_choice: str = "llama3",
):
    """
    Query the vector store. If summarize_at_query=True,
    generate summaries for top-N results only, in parallel (up to 5 workers).
    """
    print("\n[Phase 2: Performing Query/Retrieval]")

    path = "vector_store_semantic.json" if mode == "semantic" else "vector_store_exact.json"
    if not os.path.exists(path):
        print(f"[Error] Vector store not found at {path}. Run --ingest first.")
        return

    with open(path, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)

    
    chunks = [Chunk.from_dict(data) for data in chunks_data if mode == "exact" or data.get("embedding")]

    if not chunks:
        print("[Error] No chunks available.")
        return

    if mode == "semantic":
        if fuzzy: 
            print("[INFO] --fuzzy is ignored in semantic mode (only applies to exact search).")

        engine = SemanticSearchEngine()

        summarize_n = max(0, min(5, int(summaries))) if summarize_at_query else 0
        workers = min(5, max(1, int(summarize_workers)))
        top_k = max(1, int(top_k))

     
        results = engine.search(query, chunks, top_k=top_k)

        if summarize_n > 0 and results:
            # Summarize only the first N results 
            to_summarize_idx = []
            to_summarize_texts = []
            for i, res in enumerate(results[:summarize_n]):
                already = (res.detail or "").strip()
                if not already:
                    to_summarize_idx.append(i)
                    to_summarize_texts.append(res.text)

            if to_summarize_texts:
                # resolve alias to concrete model id
                resolved_model = SUMMARY_MODEL_MAP.get(summary_model_choice, summary_model_choice)
                print(f"[Summarizer] Model: {resolved_model} | items: {len(to_summarize_texts)} | workers: {min(workers, len(to_summarize_texts))}")

                summarizer = Summarizer(
                    model_name=resolved_model,  # pass resolved model id
                    workers=min(workers, len(to_summarize_texts)),
                    max_tokens=96,
                    temperature=0.0,
                )
                try:
                    summaries_out = summarizer.batch(to_summarize_texts)
                except Exception as e:
                    print(f"[Warn] Query-time summarization failed: {e}")
                    summaries_out = ["[Summary unavailable]"] * len(to_summarize_texts)

                # Install summaries into result.detail and cache on chunks
                for idx, summ in zip(to_summarize_idx, summaries_out):
                    s = (summ or "").strip() or "[Summary unavailable]"
                    results[idx].detail = s
                    if results[idx].chunk and not (getattr(results[idx].chunk, "summary", "") or "").strip():
                        results[idx].chunk.summary = s
    else:
        engine = ExactSearchEngine()
        results = engine.search(query, chunks, fuzzy=fuzzy)

       
        if summarize_at_query and results:
            summarize_n = max(0, min(5, int(summaries)))
            to_summarize_idx, to_summarize_texts = [], []
            for i, res in enumerate(results[:summarize_n]):
                to_summarize_idx.append(i)
                to_summarize_texts.append(res.text)
                #take chunks to summarize

            if to_summarize_texts:
                resolved_model = SUMMARY_MODEL_MAP.get(summary_model_choice, summary_model_choice)
                print(f"[Summarizer] Model: {resolved_model} | items: {len(to_summarize_texts)} | workers: {min(summarize_workers, len(to_summarize_texts))}")

                summarizer = Summarizer(
                    model_name=resolved_model,
                    workers=min(summarize_workers, len(to_summarize_texts)),
                    max_tokens=96,
                    temperature=0.0,
                )
                try:
                    summaries_out = summarizer.batch(to_summarize_texts)
                except Exception as e:
                    print(f"[Warn] Query-time summarization failed: {e}")
                    summaries_out = ["[Summary unavailable]"] * len(to_summarize_texts)

                for idx, summ in zip(to_summarize_idx, summaries_out):
                    s = (summ or "").strip() or "[Summary unavailable]"
                  
                    results[idx].detail = (results[idx].detail or "")
                    if results[idx].detail and not results[idx].detail.endswith("\n"):
                        results[idx].detail += "\n"
                    results[idx].detail += f"Summary: {s}"

                    
                    if getattr(results[idx], "chunk", None) and not (getattr(results[idx].chunk, "summary", "") or "").strip():
                        results[idx].chunk.summary = s


    if not results:
        print("\n[No results to display]")
        return

    if mode == "semantic":
        if summarize_at_query:
            enforce_n = max(0, min(5, int(summaries)))
            for i, res in enumerate(results):
                if i >= enforce_n:
                    res.detail = ""
        else:
            for res in results:
                res.detail = ""

    print("\nTop Results:")
    ResultFormatter.print_results(results)



def purge_vector_store():
    for file in ["vector_store_semantic.json", "vector_store_exact.json"]:
        if os.path.exists(file):
            os.remove(file)
            print(f"[INFO] {file} purged.")
        else:
            print(f"[INFO] {file} not found.")




def main():
    parser = argparse.ArgumentParser(description="Local Document Search Tool")
    parser.add_argument("--ingest", action="store_true", help="Run ingestion phase")
    parser.add_argument("--purge", action="store_true", help="Purge the chunk cache")
    parser.add_argument("--dir", help="Directory of .txt/.md files to ingest", default="./docs")
    parser.add_argument("--query", help="Text query to perform search")
    parser.add_argument("--mode", help="Choose search mode: 'semantic' or 'exact'", default="semantic")
    parser.add_argument("--append", action="store_true", help="Append new chunks to existing vector store")
    parser.add_argument("--fuzzy", action="store_true", help="Enable fuzzy matching for exact search")
    parser.add_argument("--topk", type=int, default=5, help="How many results to retrieve overall (default 5)")
    parser.add_argument(
        "--summarize-at-query", action="store_true",
        help="Generate summaries for the first N results at query time (semantic mode only)"
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
        choices=[ "phi3", "llama3"],
        default="llama3",
        help="Model to use for summaries: | phi3 | llama3",
    )

    args = parser.parse_args()

    if args.purge:
        purge_vector_store()
    elif args.ingest:
        ingest_phase(args.dir, args.mode, append=args.append)
    elif args.query:
        summarize_at_query = bool(getattr(args, "summarize_at_query", False))
        summaries_n = max(1, min(5, int(getattr(args, "summaries", 2)))) if summarize_at_query else 0
        workers = max(1, min(5, int(getattr(args, "summarize_workers", 5))))
        top_k = max(1, int(getattr(args, "topk", 5)))
        summary_model = getattr(args, "summary_model", "llama3")

        query_phase(
            args.query,
            args.mode,
            fuzzy=args.fuzzy,
            summarize_at_query=summarize_at_query,
            summarize_workers=workers,
            summaries=summaries_n,
            top_k=top_k,
            summary_model_choice=summary_model,  
        )
    else:
        print("[ERROR] Please provide --ingest, --query, or --purge")


if __name__ == "__main__":
    main()

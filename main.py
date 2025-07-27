# # # import argparse
# # # import os
# # # import pickle
# # # import hashlib

# # # from ingestion.directory_loader import DirectoryLoader
# # # from chunking.recursive_token_chunker import RecursiveTokenChunker
# # # from chunking.semantic_cluster_chunker import SemanticClusterChunker
# # # from models.data_models import Chunk, SearchResult
# # # from output.result_formatter import ResultFormatter
# # # from search.exact_search_engine import ExactSearchEngine
# # # from search.semantic_search_engine import SemanticSearchEngine
# # # from search.reranker import Reranker

# # # def get_cache_path(directory: str, mode: str) -> str:
# # #     key = hashlib.md5(f"{directory}_{mode}".encode()).hexdigest()
# # #     return f"chunk_cache_{key}.pkl"

# # # def ingest_phase(directory: str, mode: str, append: bool) -> list[Chunk]:
# # #     print("\n[Phase 1: Ingesting and Chunking Documents]\n")
# # #     loader = DirectoryLoader(directory)
# # #     docs = loader.load_documents()

# # #     chunker = SemanticClusterChunker() if mode == "semantic" else RecursiveTokenChunker()
# # #     new_chunks = []
# # #     for filepath, content in docs:
# # #         chunks = chunker.chunk(content)
# # #         for chunk in chunks:
# # #             new_chunks.append(Chunk(
# # #                 text=chunk,
# # #                 source_path=filepath,
# # #                 document_name=os.path.basename(filepath)
# # #             ))

# # #     cache_path = get_cache_path(directory, mode)
# # #     if append and os.path.exists(cache_path):
# # #         with open(cache_path, "rb") as f:
# # #             existing_chunks = pickle.load(f)
# # #         all_chunks = existing_chunks + new_chunks
# # #     else:
# # #         all_chunks = new_chunks

# # #     with open(cache_path, "wb") as f:
# # #         pickle.dump(all_chunks, f)
# # #     print(f"[Cached {len(all_chunks)} total chunks in {cache_path}]\n")
# # #     return all_chunks

# # # def retrieval_phase(query: str, search_type: str, rerank: bool, directory: str, mode: str) -> None:
# # #     print("\n[Phase 2: Performing Query/Retrieval]\n")
# # #     cache_path = get_cache_path(directory, mode)
# # #     if not os.path.exists(cache_path):
# # #         print(f"No cached chunks found for directory '{directory}'. Please run ingestion first.")
# # #         return

# # #     with open(cache_path, "rb") as f:
# # #         all_chunks = pickle.load(f)

# # #     # Filter out malformed chunks
# # #     valid_chunks = [
# # #         chunk for chunk in all_chunks
# # #         if hasattr(chunk, 'text') and isinstance(chunk.text, str)
# # #     ]

# # #     print(f"[INFO] Skipped {len(all_chunks) - len(valid_chunks)} malformed chunks out of {len(all_chunks)} total.\n")

# # #     results: list[SearchResult] = []
# # #     if search_type == "exact":
# # #         engine = ExactSearchEngine()
# # #         results = engine.search(query, valid_chunks)
# # #     elif search_type == "semantic":
# # #         engine = SemanticSearchEngine()
# # #         results = engine.search(query, valid_chunks)
# # #         if rerank:
# # #             reranker = Reranker()
# # #             results = reranker.rerank(query, results)
# # #             print("\n[Results have been reranked using cross-encoding similarity]\n")
# # #         else:
# # #             print("\n[Semantic results without reranking]\n")

# # #     ResultFormatter.print_results(results)

# # # def purge_cache(directory: str, mode: str) -> None:
# # #     cache_path = get_cache_path(directory, mode)
# # #     if os.path.exists(cache_path):
# # #         os.remove(cache_path)
# # #         print(f"[Cache file {cache_path} deleted]\n")
# # #     else:
# # #         print(f"[No cache file found for {directory} and mode {mode}]\n")

# # # def main():
# # #     parser = argparse.ArgumentParser(description="Two-Stage Document Search System")
# # #     parser.add_argument("--dir", required=True, help="Directory with .txt files")
# # #     parser.add_argument("--mode", choices=["recursive", "semantic"], default="recursive", help="Chunking strategy")
# # #     parser.add_argument("--query", help="Query string for searching")
# # #     parser.add_argument("--search", choices=["exact", "semantic"], help="Search type")
# # #     parser.add_argument("--rerank", action="store_true", help="Use cross-encoder reranker")
# # #     parser.add_argument("--ingest", action="store_true", help="Run ingestion/chunking phase")
# # #     parser.add_argument("--append", action="store_true", help="Append new chunks to existing cache")
# # #     parser.add_argument("--purge", action="store_true", help="Purge the chunk cache for the directory/mode")
# # #     args = parser.parse_args()

# # #     if args.purge:
# # #         purge_cache(args.dir, args.mode)
# # #     elif args.ingest:
# # #         ingest_phase(args.dir, args.mode, append=args.append)
# # #     elif args.query and args.search:
# # #         retrieval_phase(args.query, args.search, args.rerank, args.dir, args.mode)
# # #     else:
# # #         print("You must specify either --ingest, --purge, or both --query and --search for retrieval.")

# # # if __name__ == "__main__":
# # #     main()


# # # import argparse
# # # import os
# # # import pickle
# # # import hashlib

# # # from ingestion.directory_loader import DirectoryLoader
# # # from chunking.recursive_token_chunker import RecursiveTokenChunker
# # # from chunking.semantic_cluster_chunker import SemanticClusterChunker
# # # from models.data_models import Chunk, SearchResult
# # # from output.result_formatter import ResultFormatter
# # # from search.exact_search_engine import ExactSearchEngine
# # # from search.semantic_search_engine import SemanticSearchEngine
# # # from search.reranker import Reranker

# # # def get_cache_path(directory: str, mode: str) -> str:
# # #     key = hashlib.md5(f"{directory}_{mode}".encode()).hexdigest()
# # #     return f"chunk_cache_{key}.pkl"

# # # def ingest_phase(directory: str, mode: str, append: bool) -> list[Chunk]:
# # #     print("\n[Phase 1: Ingesting and Chunking Documents]\n")
# # #     loader = DirectoryLoader(directory)
# # #     docs = loader.load_documents()

# # #     chunker = SemanticClusterChunker() if mode == "semantic" else RecursiveTokenChunker()
# # #     new_chunks = []
# # #     for filepath, content in docs:
# # #         chunks = chunker.chunk(content, source_path=filepath, document_name=os.path.basename(filepath))
# # #         new_chunks.extend(chunks)

# # #     cache_path = get_cache_path(directory, mode)
# # #     if append and os.path.exists(cache_path):
# # #         with open(cache_path, "rb") as f:
# # #             existing_chunks = pickle.load(f)
# # #         all_chunks = existing_chunks + new_chunks
# # #     else:
# # #         all_chunks = new_chunks

# # #     with open(cache_path, "wb") as f:
# # #         pickle.dump(all_chunks, f)
# # #     print(f"[Cached {len(all_chunks)} total chunks in {cache_path}]\n")
# # #     return all_chunks

# # # def retrieval_phase(query: str, search_type: str, rerank: bool, directory: str, mode: str) -> None:
# # #     print("\n[Phase 2: Performing Query/Retrieval]\n")
# # #     cache_path = get_cache_path(directory, mode)
# # #     if not os.path.exists(cache_path):
# # #         print(f"No cached chunks found for directory '{directory}'. Please run ingestion first.")
# # #         return

# # #     with open(cache_path, "rb") as f:
# # #         all_chunks = pickle.load(f)

# # #     # Filter out malformed chunks
# # #     valid_chunks = [
# # #         chunk for chunk in all_chunks
# # #         if hasattr(chunk, 'text') and isinstance(chunk.text, str)
# # #     ]

# # #     print(f"[INFO] Skipped {len(all_chunks) - len(valid_chunks)} malformed chunks out of {len(all_chunks)} total.\n")

# # #     if not valid_chunks:
# # #         print("[ERROR] No valid chunks found. Please verify your chunking logic.")
# # #         return

# # #     results: list[SearchResult] = []
# # #     if search_type == "exact":
# # #         engine = ExactSearchEngine()
# # #         results = engine.search(query, valid_chunks)
# # #     elif search_type == "semantic":
# # #         engine = SemanticSearchEngine()
# # #         results = engine.search(query, valid_chunks)
# # #         if rerank:
# # #             reranker = Reranker()
# # #             results = reranker.rerank(query, results)
# # #             print("\n[Results have been reranked using cross-encoding similarity]\n")
# # #         else:
# # #             print("\n[Semantic results without reranking]\n")

# # #     ResultFormatter.print_results(results)

# # # def purge_cache(directory: str, mode: str) -> None:
# # #     cache_path = get_cache_path(directory, mode)
# # #     if os.path.exists(cache_path):
# # #         os.remove(cache_path)
# # #         print(f"[Cache file {cache_path} deleted]\n")
# # #     else:
# # #         print(f"[No cache file found for {directory} and mode {mode}]\n")

# # # def main():
# # #     parser = argparse.ArgumentParser(description="Two-Stage Document Search System")
# # #     parser.add_argument("--dir", required=True, help="Directory with .txt files")
# # #     parser.add_argument("--mode", choices=["recursive", "semantic"], default="recursive", help="Chunking strategy")
# # #     parser.add_argument("--query", help="Query string for searching")
# # #     parser.add_argument("--search", choices=["exact", "semantic"], help="Search type")
# # #     parser.add_argument("--rerank", action="store_true", help="Use cross-encoder reranker")
# # #     parser.add_argument("--ingest", action="store_true", help="Run ingestion/chunking phase")
# # #     parser.add_argument("--append", action="store_true", help="Append new chunks to existing cache")
# # #     parser.add_argument("--purge", action="store_true", help="Purge the chunk cache for the directory/mode")
# # #     args = parser.parse_args()

# # #     if args.purge:
# # #         purge_cache(args.dir, args.mode)
# # #     elif args.ingest:
# # #         ingest_phase(args.dir, args.mode, append=args.append)
# # #     elif args.query and args.search:
# # #         retrieval_phase(args.query, args.search, args.rerank, args.dir, args.mode)
# # #     else:
# # #         print("You must specify either --ingest, --purge, or both --query and --search for retrieval.")

# # # if __name__ == "__main__":
# # #     main()


# # import argparse
# # import os
# # import pickle
# # import hashlib

# # from ingestion.directory_loader import DirectoryLoader
# # from chunking.recursive_token_chunker import RecursiveTokenChunker
# # from chunking.semantic_cluster_chunker import SemanticClusterChunker
# # from models.data_models import Chunk, SearchResult
# # from output.result_formatter import ResultFormatter
# # from search.exact_search_engine import ExactSearchEngine
# # from search.semantic_search_engine import SemanticSearchEngine
# # from search.reranker import Reranker

# # def get_cache_path(directory: str, mode: str) -> str:
# #     key = hashlib.md5(f"{directory}_{mode}".encode()).hexdigest()
# #     return f"chunk_cache_{key}.pkl"

# # def ingest_phase(directory: str, mode: str, append: bool) -> list[Chunk]:
# #     print("\n[Phase 1: Ingesting and Chunking Documents]\n")
# #     loader = DirectoryLoader(directory)
# #     docs = loader.load_documents()

# #     chunker = SemanticClusterChunker() if mode == "semantic" else RecursiveTokenChunker()
# #     new_chunks = []

# #     for filepath, content in docs:
# #         # ✅ Updated: pass source_path and document_name
# #         chunks = chunker.chunk(content, source_path=filepath, document_name=os.path.basename(filepath))



# #         for result in chunks:
# #             if isinstance(result, tuple) and len(result) == 2:
# #                 chunk_text, context = result
# #             else:
# #                 chunk_text, context = result, None  # fallback for recursive or legacy modes

# #             if isinstance(chunk_text, str) and chunk_text.strip():
# #                 new_chunks.append(Chunk(
# #                     text=chunk_text.strip(),
# #                     context=context,
# #                     source_path=filepath,
# #                     document_name=os.path.basename(filepath)
# #                 ))





# #             new_chunks.append(Chunk(
# #                 text=chunk_text,
# #                 context=context,
# #                 source_path=filepath,
# #                 document_name=os.path.basename(filepath)
# #             ))

# #     cache_path = get_cache_path(directory, mode)
# #     if append and os.path.exists(cache_path):
# #         with open(cache_path, "rb") as f:
# #             existing_chunks = pickle.load(f)
# #         all_chunks = existing_chunks + new_chunks
# #     else:
# #         all_chunks = new_chunks

# #     with open(cache_path, "wb") as f:
# #         pickle.dump(all_chunks, f)
# #     print(f"[Cached {len(all_chunks)} total chunks in {cache_path}]\n")
# #     return all_chunks

# # def retrieval_phase(query: str, search_type: str, rerank: bool, directory: str, mode: str) -> None:
# #     print("\n[Phase 2: Performing Query/Retrieval]\n")
# #     cache_path = get_cache_path(directory, mode)
# #     if not os.path.exists(cache_path):
# #         print(f"No cached chunks found for directory '{directory}'. Please run ingestion first.")
# #         return

# #     with open(cache_path, "rb") as f:
# #         all_chunks = pickle.load(f)

# #     valid_chunks = [
# #         chunk for chunk in all_chunks
# #         if hasattr(chunk, 'text') and isinstance(chunk.text, str)
# #     ]

# #     print(f"[INFO] Skipped {len(all_chunks) - len(valid_chunks)} malformed chunks out of {len(all_chunks)} total.\n")

# #     if not valid_chunks:
# #         print("[ERROR] No valid chunks found. Please verify your chunking logic.")
# #         return

# #     results: list[SearchResult] = []
# #     if search_type == "exact":
# #         engine = ExactSearchEngine()
# #         results = engine.search(query, valid_chunks)
# #     elif search_type == "semantic":
# #         engine = SemanticSearchEngine()
# #         results = engine.search(query, valid_chunks)
# #         if rerank:
# #             reranker = Reranker()
# #             results = reranker.rerank(query, results)
# #             print("\n[Results have been reranked using cross-encoding similarity]\n")
# #         else:
# #             print("\n[Semantic results without reranking]\n")

# #     ResultFormatter.print_results(results)

# # def purge_cache(directory: str, mode: str) -> None:
# #     cache_path = get_cache_path(directory, mode)
# #     if os.path.exists(cache_path):
# #         os.remove(cache_path)
# #         print(f"[Cache file {cache_path} deleted]\n")
# #     else:
# #         print(f"[No cache file found for {directory} and mode {mode}]\n")

# # def main():
# #     parser = argparse.ArgumentParser(description="Two-Stage Document Search System")
# #     parser.add_argument("--dir", required=True, help="Directory with .txt files")
# #     parser.add_argument("--mode", choices=["recursive", "semantic"], default="recursive", help="Chunking strategy")
# #     parser.add_argument("--query", help="Query string for searching")
# #     parser.add_argument("--search", choices=["exact", "semantic"], help="Search type")
# #     parser.add_argument("--rerank", action="store_true", help="Use cross-encoder reranker")
# #     parser.add_argument("--ingest", action="store_true", help="Run ingestion/chunking phase")
# #     parser.add_argument("--append", action="store_true", help="Append new chunks to existing cache")
# #     parser.add_argument("--purge", action="store_true", help="Purge the chunk cache for the directory/mode")
# #     args = parser.parse_args()

# #     if args.purge:
# #         purge_cache(args.dir, args.mode)
# #     elif args.ingest:
# #         ingest_phase(args.dir, args.mode, append=args.append)
# #     elif args.query and args.search:
# #         retrieval_phase(args.query, args.search, args.rerank, args.dir, args.mode)
# #     else:
# #         print("You must specify either --ingest, --purge, or both --query and --search for retrieval.")

# # if __name__ == "__main__":
# #     main()
# # import argparse
# # import os
# # import pickle
# # import hashlib
# # import time

# # from ingestion.directory_loader import DirectoryLoader
# # from chunking.recursive_token_chunker import RecursiveTokenChunker
# # from chunking.semantic_cluster_chunker import SemanticClusterChunker
# # from models.data_models import Chunk, SearchResult
# # from output.result_formatter import ResultFormatter
# # from search.exact_search_engine import ExactSearchEngine
# # from search.semantic_search_engine import SemanticSearchEngine
# # from search.reranker import Reranker

# # def get_cache_path(directory: str, mode: str) -> str:
# #     key = hashlib.md5(f"{directory}_{mode}".encode()).hexdigest()
# #     return f"chunk_cache_{key}.pkl"

# # def ingest_phase(directory: str, mode: str, append: bool) -> list[Chunk]:
# #     print("\n[Phase 1: Ingesting and Chunking Documents]\n")
# #     loader = DirectoryLoader(directory)
# #     docs = loader.load_documents()

# #     chunker = SemanticClusterChunker() if mode == "semantic" else RecursiveTokenChunker()
# #     new_chunks = []

# #     for filepath, content in docs:
# #         chunks = chunker.chunk(content, source_path=filepath, document_name=os.path.basename(filepath))

# #         for idx, result in enumerate(chunks):
# #             if isinstance(result, tuple) and len(result) == 2:
# #                 chunk_text, context = result
# #             else:
# #                 chunk_text, context = result, None  # fallback for recursive or legacy modes

# #             if isinstance(chunk_text, str) and chunk_text.strip():
# #                 start_time = time.time()

# #                 try:
# #                     embed_start = time.time()
# #                     _ = chunker.embedder.get_embedding(chunk_text)
# #                     embed_duration = time.time() - embed_start
# #                 except Exception as e:
# #                     embed_duration = -1
# #                     print(f"[CHUNK {idx}] ⚠️ Embedding failed: {e}")

# #                 try:
# #                     token_count = len(chunker.tokenizer.encode(chunk_text, add_special_tokens=False))
# #                 except Exception:
# #                     token_count = -1

# #                 print(f"\n📄 File: {os.path.basename(filepath)}")
# #                 print(f"  ↳ Chunk #{idx + 1}")
# #                 print(f"  ↳ Tokens: {token_count}")
# #                 print(f"  ↳ Embedding time: {embed_duration:.3f}s")
# #                 print(f"  ↳ Preview: {chunk_text[:100]}")

# #                 new_chunks.append(Chunk(
# #                     text=chunk_text.strip(),
# #                     context=context,
# #                     source_path=filepath,
# #                     document_name=os.path.basename(filepath)
# #                 ))

# #     cache_path = get_cache_path(directory, mode)
# #     if append and os.path.exists(cache_path):
# #         with open(cache_path, "rb") as f:
# #             existing_chunks = pickle.load(f)
# #         all_chunks = existing_chunks + new_chunks
# #     else:
# #         all_chunks = new_chunks

# #     with open(cache_path, "wb") as f:
# #         pickle.dump(all_chunks, f)
# #     print(f"[Cached {len(all_chunks)} total chunks in {cache_path}]\n")
# #     return all_chunks

# # def retrieval_phase(query: str, search_type: str, rerank: bool, directory: str, mode: str) -> None:
# #     print("\n[Phase 2: Performing Query/Retrieval]\n")
# #     cache_path = get_cache_path(directory, mode)
# #     if not os.path.exists(cache_path):
# #         print(f"No cached chunks found for directory '{directory}'. Please run ingestion first.")
# #         return

# #     with open(cache_path, "rb") as f:
# #         all_chunks = pickle.load(f)

# #     valid_chunks = [
# #         chunk for chunk in all_chunks
# #         if hasattr(chunk, 'text') and isinstance(chunk.text, str)
# #     ]

# #     print(f"[INFO] Skipped {len(all_chunks) - len(valid_chunks)} malformed chunks out of {len(all_chunks)} total.\n")

# #     if not valid_chunks:
# #         print("[ERROR] No valid chunks found. Please verify your chunking logic.")
# #         return

# #     results: list[SearchResult] = []
# #     if search_type == "exact":
# #         engine = ExactSearchEngine()
# #         results = engine.search(query, valid_chunks)
# #     elif search_type == "semantic":
# #         engine = SemanticSearchEngine()
# #         results = engine.search(query, valid_chunks)
# #         if rerank:
# #             reranker = Reranker()
# #             results = reranker.rerank(query, results)
# #             print("\n[Results have been reranked using cross-encoding similarity]\n")
# #         else:
# #             print("\n[Semantic results without reranking]\n")

# #     ResultFormatter.print_results(results)

# # def purge_cache(directory: str, mode: str) -> None:
# #     cache_path = get_cache_path(directory, mode)
# #     if os.path.exists(cache_path):
# #         os.remove(cache_path)
# #         print(f"[Cache file {cache_path} deleted]\n")
# #     else:
# #         print(f"[No cache file found for {directory} and mode {mode}]\n")

# # def main():
# #     parser = argparse.ArgumentParser(description="Two-Stage Document Search System")
# #     parser.add_argument("--dir", required=True, help="Directory with .txt files")
# #     parser.add_argument("--mode", choices=["recursive", "semantic"], default="recursive", help="Chunking strategy")
# #     parser.add_argument("--query", help="Query string for searching")
# #     parser.add_argument("--search", choices=["exact", "semantic"], help="Search type")
# #     parser.add_argument("--rerank", action="store_true", help="Use cross-encoder reranker")
# #     parser.add_argument("--ingest", action="store_true", help="Run ingestion/chunking phase")
# #     parser.add_argument("--append", action="store_true", help="Append new chunks to existing cache")
# #     parser.add_argument("--purge", action="store_true", help="Purge the chunk cache for the directory/mode")
# #     args = parser.parse_args()

# #     if args.purge:
# #         purge_cache(args.dir, args.mode)
# #     elif args.ingest:
# #         ingest_phase(args.dir, args.mode, append=args.append)
# #     elif args.query and args.search:
# #         retrieval_phase(args.query, args.search, args.rerank, args.dir, args.mode)
# #     else:
# #         print("You must specify either --ingest, --purge, or both --query and --search for retrieval.")

# # if __name__ == "__main__":
# #     main()

# import argparse
# import os
# import pickle
# import hashlib
# import time

# from ingestion.directory_loader import DirectoryLoader
# from chunking.recursive_token_chunker import RecursiveTokenChunker
# from chunking.semantic_cluster_chunker import SemanticClusterChunker
# from models.data_models import Chunk, SearchResult
# from output.result_formatter import ResultFormatter
# from search.exact_search_engine import ExactSearchEngine
# from search.semantic_search_engine import SemanticSearchEngine
# from search.reranker import Reranker

# def get_cache_path(directory: str, mode: str) -> str:
#     key = hashlib.md5(f"{directory}_{mode}".encode()).hexdigest()
#     return f"chunk_cache_{key}.pkl"

# def ingest_phase(directory: str, mode: str, append: bool) -> list[Chunk]:
#     print("\n[Phase 1: Ingesting and Chunking Documents]\n")
#     loader = DirectoryLoader(directory)
#     docs = loader.load_documents()

#     chunker = SemanticClusterChunker() if mode == "semantic" else RecursiveTokenChunker()
#     new_chunks = []

#     for filepath, content in docs:
#         chunks = chunker.chunk(content, source_path=filepath, document_name=os.path.basename(filepath))

#         for idx, result in enumerate(chunks):
#             if isinstance(result, tuple) and len(result) == 2:
#                 chunk_text, context = result
#             else:
#                 chunk_text, context = result, None  # fallback for recursive or legacy modes

#             if isinstance(chunk_text, str) and chunk_text.strip():
#                 start_time = time.time()

#                 try:
#                     embed_start = time.time()
#                     _ = chunker.embedder.get_embedding(chunk_text)
#                     embed_duration = time.time() - embed_start
#                 except Exception as e:
#                     embed_duration = -1
#                     print(f"[CHUNK {idx}] ⚠️ Embedding failed: {e}")

#                 try:
#                     token_count = len(chunker.tokenizer.encode(chunk_text, add_special_tokens=False))
#                 except Exception:
#                     token_count = -1

#                 print(f"\n📄 File: {os.path.basename(filepath)}")
#                 print(f"  ↳ Chunk #{idx + 1}")
#                 print(f"  ↳ Tokens: {token_count}")
#                 print(f"  ↳ Embedding time: {embed_duration:.3f}s")
#                 print(f"  ↳ Preview: {chunk_text[:100]}")

#                 new_chunks.append(Chunk(
#                     text=chunk_text.strip(),
#                     # context=context,  # ⛔ Commented out for minimal embedding
#                     source_path=filepath,
#                     document_name=os.path.basename(filepath)
#                 ))
#                 # Uncomment below to include context again if needed
#                 # new_chunks.append(Chunk(
#                 #     text=chunk_text.strip(),
#                 #     context=context,
#                 #     source_path=filepath,
#                 #     document_name=os.path.basename(filepath)
#                 # ))

#     cache_path = get_cache_path(directory, mode)
#     if append and os.path.exists(cache_path):
#         with open(cache_path, "rb") as f:
#             existing_chunks = pickle.load(f)
#         all_chunks = existing_chunks + new_chunks
#     else:
#         all_chunks = new_chunks

#     with open(cache_path, "wb") as f:
#         pickle.dump(all_chunks, f)
#     print(f"[Cached {len(all_chunks)} total chunks in {cache_path}]\n")
#     return all_chunks

# def retrieval_phase(query: str, search_type: str, rerank: bool, directory: str, mode: str) -> None:
#     print("\n[Phase 2: Performing Query/Retrieval]\n")
#     cache_path = get_cache_path(directory, mode)
#     if not os.path.exists(cache_path):
#         print(f"No cached chunks found for directory '{directory}'. Please run ingestion first.")
#         return

#     with open(cache_path, "rb") as f:
#         all_chunks = pickle.load(f)

#     valid_chunks = [
#         chunk for chunk in all_chunks
#         if hasattr(chunk, 'text') and isinstance(chunk.text, str)
#     ]

#     print(f"[INFO] Skipped {len(all_chunks) - len(valid_chunks)} malformed chunks out of {len(all_chunks)} total.\n")

#     if not valid_chunks:
#         print("[ERROR] No valid chunks found. Please verify your chunking logic.")
#         return

#     results: list[SearchResult] = []
#     if search_type == "exact":
#         engine = ExactSearchEngine()
#         results = engine.search(query, valid_chunks)
#     elif search_type == "semantic":
#         engine = SemanticSearchEngine()
#         results = engine.search(query, valid_chunks)
#         if rerank:
#             reranker = Reranker()
#             results = reranker.rerank(query, results)
#             print("\n[Results have been reranked using cross-encoding similarity]\n")
#         else:
#             print("\n[Semantic results without reranking]\n")

#     ResultFormatter.print_results(results)

# def purge_cache(directory: str, mode: str) -> None:
#     cache_path = get_cache_path(directory, mode)
#     if os.path.exists(cache_path):
#         os.remove(cache_path)
#         print(f"[Cache file {cache_path} deleted]\n")
#     else:
#         print(f"[No cache file found for {directory} and mode {mode}]\n")

# def main():
#     parser = argparse.ArgumentParser(description="Two-Stage Document Search System")
#     parser.add_argument("--dir", required=True, help="Directory with .txt files")
#     parser.add_argument("--mode", choices=["recursive", "semantic"], default="recursive", help="Chunking strategy")
#     parser.add_argument("--query", help="Query string for searching")
#     parser.add_argument("--search", choices=["exact", "semantic"], help="Search type")
#     parser.add_argument("--rerank", action="store_true", help="Use cross-encoder reranker")
#     parser.add_argument("--ingest", action="store_true", help="Run ingestion/chunking phase")
#     parser.add_argument("--append", action="store_true", help="Append new chunks to existing cache")
#     parser.add_argument("--purge", action="store_true", help="Purge the chunk cache for the directory/mode")
#     args = parser.parse_args()

#     if args.purge:
#         purge_cache(args.dir, args.mode)
#     elif args.ingest:
#         ingest_phase(args.dir, args.mode, append=args.append)
#     elif args.query and args.search:
#         retrieval_phase(args.query, args.search, args.rerank, args.dir, args.mode)
#     else:
#         print("You must specify either --ingest, --purge, or both --query and --search for retrieval.")

# if __name__ == "__main__":
#     main()

# import argparse
# import os
# import pickle
# import hashlib
# import time

# from ingestion.directory_loader import DirectoryLoader
# from chunking.recursive_token_chunker import RecursiveTokenChunker
# from models.data_models import Chunk, SearchResult
# from output.result_formatter import ResultFormatter
# from search.exact_search_engine import ExactSearchEngine
# from search.semantic_search_engine import SemanticSearchEngine
# from search.reranker import Reranker
# from embedding.ollama_embedder import OllamaEmbedder

# def get_cache_path(directory: str, mode: str) -> str:
#     key = hashlib.md5(f"{directory}_{mode}".encode()).hexdigest()
#     return f"chunk_cache_{key}.pkl"

# def ingest_phase(directory: str, mode: str, append: bool) -> list[Chunk]:
#     print("\n[Phase 1: Ingesting, Chunking, and Embedding Documents]\n")
#     loader = DirectoryLoader(directory)
#     docs = loader.load_documents()

#     chunker = RecursiveTokenChunker()
#     embedder = OllamaEmbedder()
#     new_chunks = []

#     for filepath, content in docs:
#         print(f"\n📄 Processing file: {os.path.basename(filepath)}")
#         chunks = chunker.chunk(content, source_path=filepath, document_name=os.path.basename(filepath))

#         for idx, chunk in enumerate(chunks):
#             if isinstance(chunk, Chunk) and chunk.text.strip():
#                 try:
#                     embedding = embedder.get_embedding(chunk.text)
#                     chunk.embedding = embedding
#                     print(f"  ↳ Cached Chunk #{idx + 1} — Tokens: {len(chunker.tokenizer.encode(chunk.text, add_special_tokens=False))}")
#                     print(f"  ↳ Preview: {chunk.text[:100].replace('\n', ' ')}...\n")
#                     new_chunks.append(chunk)
#                 except Exception as e:
#                     print(f"  ⚠️ Embedding failed for chunk #{idx + 1}: {e}")

#     cache_path = get_cache_path(directory, mode)
#     if append and os.path.exists(cache_path):
#         with open(cache_path, "rb") as f:
#             existing_chunks = pickle.load(f)
#         all_chunks = existing_chunks + new_chunks
#     else:
#         all_chunks = new_chunks

#     with open(cache_path, "wb") as f:
#         pickle.dump(all_chunks, f)

#     print(f"[Cached {len(all_chunks)} total chunks in {cache_path}]\n")
#     return all_chunks

# def retrieval_phase(query: str, search_type: str, rerank: bool, directory: str, mode: str) -> None:
#     print("\n[Phase 2: Performing Query/Retrieval]\n")
#     cache_path = get_cache_path(directory, mode)
#     if not os.path.exists(cache_path):
#         print(f"No cached chunks found for directory '{directory}'. Please run ingestion first.")
#         return

#     with open(cache_path, "rb") as f:
#         all_chunks = pickle.load(f)

#     valid_chunks = [
#         chunk for chunk in all_chunks
#         if hasattr(chunk, 'text') and isinstance(chunk.text, str)
#     ]

#     print(f"[INFO] Skipped {len(all_chunks) - len(valid_chunks)} malformed chunks out of {len(all_chunks)} total.\n")

#     if not valid_chunks:
#         print("[ERROR] No valid chunks found. Please verify your chunking logic.")
#         return

#     results: list[SearchResult] = []
#     if search_type == "exact":
#         engine = ExactSearchEngine()
#         results = engine.search(query, valid_chunks)
#     elif search_type == "semantic":
#         engine = SemanticSearchEngine()
#         results = engine.search(query, valid_chunks)
#         if rerank:
#             reranker = Reranker()
#             results = reranker.rerank(query, results)
#             print("\n[Results have been reranked using cross-encoding similarity]\n")
#         else:
#             print("\n[Semantic results without reranking]\n")

#     ResultFormatter.print_results(results)

# def purge_cache(directory: str, mode: str) -> None:
#     cache_path = get_cache_path(directory, mode)
#     if os.path.exists(cache_path):
#         os.remove(cache_path)
#         print(f"[Cache file {cache_path} deleted]\n")
#     else:
#         print(f"[No cache file found for {directory} and mode {mode}]\n")

# def main():
#     parser = argparse.ArgumentParser(description="Two-Stage Document Search System")
#     parser.add_argument("--dir", required=True, help="Directory with .txt files")
#     parser.add_argument("--mode", choices=["recursive"], default="recursive", help="Chunking strategy")
#     parser.add_argument("--query", help="Query string for searching")
#     parser.add_argument("--search", choices=["exact", "semantic"], help="Search type")
#     parser.add_argument("--rerank", action="store_true", help="Use cross-encoder reranker")
#     parser.add_argument("--ingest", action="store_true", help="Run ingestion/chunking phase")
#     parser.add_argument("--append", action="store_true", help="Append new chunks to existing cache")
#     parser.add_argument("--purge", action="store_true", help="Purge the chunk cache for the directory/mode")
#     args = parser.parse_args()

#     if args.purge:
#         purge_cache(args.dir, args.mode)
#     elif args.ingest:
#         ingest_phase(args.dir, args.mode, append=args.append)
#     elif args.query and args.search:
#         retrieval_phase(args.query, args.search, args.rerank, args.dir, args.mode)
#     else:
#         print("You must specify either --ingest, --purge, or both --query and --search for retrieval.")

# if __name__ == "__main__":
#     main()


# import argparse
# import os
# import pickle
# import hashlib
# import time

# from ingestion.directory_loader import DirectoryLoader
# from chunking.recursive_token_chunker import RecursiveTokenChunker
# from models.data_models import Chunk, SearchResult
# from output.result_formatter import ResultFormatter
# from search.exact_search_engine import ExactSearchEngine
# from search.semantic_search_engine import SemanticSearchEngine
# from search.reranker import Reranker
# from embedding.ollama_embedder import OllamaEmbedder
# from concurrent.futures import ThreadPoolExecutor, as_completed

# def get_cache_path(directory: str, mode: str) -> str:
#     key = hashlib.md5(f"{directory}_{mode}".encode()).hexdigest()
#     return f"chunk_cache_{key}.pkl"

# # ...

# def ingest_phase(directory: str, mode: str, append: bool) -> list[Chunk]:
#     print("\n[Phase 1: Ingesting, Chunking, and Embedding Documents]\n")
#     loader = DirectoryLoader(directory)
#     docs = loader.load_documents()

#     chunker = RecursiveTokenChunker()
#     embedder = OllamaEmbedder()
#     new_chunks = []

#     def embed_chunk(chunk: Chunk, idx: int):
#         try:
#             embedding = embedder.get_embedding(chunk.text)
#             chunk.embedding = embedding
#             token_count = len(chunker.tokenizer.encode(chunk.text, add_special_tokens=False))
#             preview = chunk.text[:100].replace("\n", " ")
#             return (chunk, idx, token_count, preview, None)
#         except Exception as e:
#             return (None, idx, None, None, str(e))

#     all_raw_chunks = []
#     for filepath, content in docs:
#         print(f"\n📄 Processing file: {os.path.basename(filepath)}")
#         chunks = chunker.chunk(content, source_path=filepath, document_name=os.path.basename(filepath))
#         all_raw_chunks.extend(chunks)

#     print(f"\n[INFO] Embedding {len(all_raw_chunks)} chunks using parallel threads...")

#     with ThreadPoolExecutor(max_workers=8) as executor:  # Adjust to match your CPU capability
#         futures = {executor.submit(embed_chunk, chunk, idx): (chunk, idx)
#                    for idx, chunk in enumerate(all_raw_chunks)}

#         for future in as_completed(futures):
#             chunk, idx, token_count, preview, error = future.result()
#             if error:
#                 print(f"  ⚠️ Embedding failed for chunk #{idx + 1}: {error}")
#             elif chunk:
#                 print(f"  ↳ Cached Chunk #{idx + 1} — Tokens: {token_count}")
#                 print(f"    ↳ Preview: {preview}...\n")
#                 new_chunks.append(chunk)

#     # Cache logic (unchanged)
#     cache_path = get_cache_path(directory, mode)
#     if append and os.path.exists(cache_path):
#         with open(cache_path, "rb") as f:
#             existing_chunks = pickle.load(f)
#         all_chunks = existing_chunks + new_chunks
#     else:
#         all_chunks = new_chunks

#     with open(cache_path, "wb") as f:
#         pickle.dump(all_chunks, f)

#     print(f"[Cached {len(all_chunks)} total chunks in {cache_path}]\n")
#     return all_chunks

# def retrieval_phase(query: str, search_type: str, rerank: bool, directory: str, mode: str) -> None:
#     print("\n[Phase 2: Performing Query/Retrieval]\n")
#     cache_path = get_cache_path(directory, mode)
#     if not os.path.exists(cache_path):
#         print(f"No cached chunks found for directory '{directory}'. Please run ingestion first.")
#         return

#     with open(cache_path, "rb") as f:
#         all_chunks = pickle.load(f)

#     valid_chunks = [
#         chunk for chunk in all_chunks
#         if hasattr(chunk, 'text') and isinstance(chunk.text, str)
#     ]

#     print(f"[INFO] Skipped {len(all_chunks) - len(valid_chunks)} malformed chunks out of {len(all_chunks)} total.\n")

#     if not valid_chunks:
#         print("[ERROR] No valid chunks found. Please verify your chunking logic.")
#         return

#     results: list[SearchResult] = []
#     if search_type == "exact":
#         engine = ExactSearchEngine()
#         results = engine.search(query, valid_chunks)
#     elif search_type == "semantic":
#         engine = SemanticSearchEngine()
#         results = engine.search(query, valid_chunks)
#         if rerank:
#             reranker = Reranker()
#             results = reranker.rerank(query, results)
#             print("\n[Results have been reranked using cross-encoding similarity]\n")
#         else:
#             print("\n[Semantic results without reranking]\n")

#     ResultFormatter.print_results(results)

# def purge_cache(directory: str, mode: str) -> None:
#     cache_path = get_cache_path(directory, mode)
#     if os.path.exists(cache_path):
#         os.remove(cache_path)
#         print(f"[Cache file {cache_path} deleted]\n")
#     else:
#         print(f"[No cache file found for {directory} and mode {mode}]\n")

# def main():
#     parser = argparse.ArgumentParser(description="Two-Stage Document Search System")
#     parser.add_argument("--dir", required=True, help="Directory with .txt files")
#     parser.add_argument("--mode", choices=["recursive"], default="recursive", help="Chunking strategy")
#     parser.add_argument("--query", help="Query string for searching")
#     parser.add_argument("--search", choices=["exact", "semantic"], help="Search type")
#     parser.add_argument("--rerank", action="store_true", help="Use cross-encoder reranker")
#     parser.add_argument("--ingest", action="store_true", help="Run ingestion/chunking phase")
#     parser.add_argument("--append", action="store_true", help="Append new chunks to existing cache")
#     parser.add_argument("--purge", action="store_true", help="Purge the chunk cache for the directory/mode")
#     args = parser.parse_args()

#     if args.purge:
#         purge_cache(args.dir, args.mode)
#     elif args.ingest:
#         ingest_phase(args.dir, args.mode, append=args.append)
#     elif args.query and args.search:
#         retrieval_phase(args.query, args.search, args.rerank, args.dir, args.mode)
#     else:
#         print("You must specify either --ingest, --purge, or both --query and --search for retrieval.")

# if __name__ == "__main__":
#     main()

# import argparse
# import os
# import pickle
# import hashlib
# import time

# from ingestion.directory_loader import DirectoryLoader
# from chunking.recursive_token_chunker import RecursiveTokenChunker
# from models.data_models import Chunk, SearchResult
# from output.result_formatter import ResultFormatter
# from search.exact_search_engine import ExactSearchEngine
# from search.semantic_search_engine import SemanticSearchEngine
# from search.reranker import Reranker
# from embedding.ollama_embedder import OllamaEmbedder
# from concurrent.futures import ThreadPoolExecutor, as_completed
# import numpy as np

# def get_cache_path(directory: str, mode: str) -> str:
#     key = hashlib.md5(f"{directory}_{mode}".encode()).hexdigest()
#     return f"chunk_cache_{key}.pkl"

# def ingest_phase(directory: str, mode: str, append: bool) -> list[Chunk]:
#     print("\n[Phase 1: Ingesting, Chunking, and Embedding Documents]\n")
#     loader = DirectoryLoader(directory)
#     docs = loader.load_documents()

#     chunker = RecursiveTokenChunker()
#     embedder = OllamaEmbedder()
#     new_chunks = []

#     def embed_chunk(chunk: Chunk, idx: int):
#         try:
#             embedding = embedder.get_embedding(chunk.text)
#             if isinstance(embedding, list) and isinstance(embedding[0], list):
#                 embedding = embedding[0]  # unwrap outer list
#             chunk.embedding = embedding
#             token_count = len(chunker.tokenizer.encode(chunk.text, add_special_tokens=False))
#             preview = chunk.text[:100].replace("\n", " ")
#             return (chunk, idx, token_count, preview, None)
#         except Exception as e:
#             return (None, idx, None, None, str(e))

#     all_raw_chunks = []
#     for filepath, content in docs:
#         print(f"\n📄 Processing file: {os.path.basename(filepath)}")
#         chunks = chunker.chunk(content, source_path=filepath, document_name=os.path.basename(filepath))
#         all_raw_chunks.extend(chunks)

#     print(f"\n[INFO] Embedding {len(all_raw_chunks)} chunks using parallel threads...")

#     with ThreadPoolExecutor(max_workers=8) as executor:
#         futures = {executor.submit(embed_chunk, chunk, idx): (chunk, idx)
#                    for idx, chunk in enumerate(all_raw_chunks)}

#         for future in as_completed(futures):
#             chunk, idx, token_count, preview, error = future.result()
#             if error:
#                 print(f"  ⚠️ Embedding failed for chunk #{idx + 1}: {error}")
#             elif chunk:
#                 print(f"  ↳ Cached Chunk #{idx + 1} — Tokens: {token_count}")
#                 print(f"    ↳ Preview: {preview}...\n")
#                 new_chunks.append(chunk)

#     cache_path = get_cache_path(directory, mode)
#     if append and os.path.exists(cache_path):
#         with open(cache_path, "rb") as f:
#             existing_chunks = pickle.load(f)
#         all_chunks = existing_chunks + new_chunks
#     else:
#         all_chunks = new_chunks

#     with open(cache_path, "wb") as f:
#         pickle.dump(all_chunks, f)

#     print(f"[Cached {len(all_chunks)} total chunks in {cache_path}]\n")
#     return all_chunks

# def retrieval_phase(query: str, search_type: str, rerank: bool, directory: str, mode: str) -> None:
#     print("\n[Phase 2: Performing Query/Retrieval]\n")
#     cache_path = get_cache_path(directory, mode)
#     if not os.path.exists(cache_path):
#         print(f"No cached chunks found for directory '{directory}'. Please run ingestion first.")
#         return

#     with open(cache_path, "rb") as f:
#         all_chunks = pickle.load(f)

#     valid_chunks = [
#         chunk for chunk in all_chunks
#         if hasattr(chunk, 'text') and isinstance(chunk.text, str) and hasattr(chunk, 'embedding')
#            and isinstance(chunk.embedding, list) and np.array(chunk.embedding).ndim == 1
#     ]

#     print(f"[INFO] Skipped {len(all_chunks) - len(valid_chunks)} malformed or unembedded chunks out of {len(all_chunks)} total.\n")

#     if not valid_chunks:
#         print("[ERROR] No valid chunks found. Please verify your chunking and embedding logic.")
#         return

#     results: list[SearchResult] = []
#     if search_type == "exact":
#         engine = ExactSearchEngine()
#         results = engine.search(query, valid_chunks)
#     elif search_type == "semantic":
#         engine = SemanticSearchEngine()
#         engine.embedder = OllamaEmbedder(model_name="mxbai-embed-large")
#         try:
#             engine.set_query_embedding(query)  # NEW: handles embedding and safety checks
#         except Exception as e:
#             print(f"[Error] Failed to embed query: {e}")
#             return

#         results = engine.search(query, valid_chunks)

#         if rerank:
#             reranker = Reranker()
#             results = reranker.rerank(query, results)
#             print("\n[Results have been reranked using cross-encoding similarity]\n")
#         else:
#             print("\n[Semantic results without reranking]\n")

#     ResultFormatter.print_results(results)

# def purge_cache(directory: str, mode: str) -> None:
#     cache_path = get_cache_path(directory, mode)
#     if os.path.exists(cache_path):
#         os.remove(cache_path)
#         print(f"[Cache file {cache_path} deleted]\n")
#     else:
#         print(f"[No cache file found for {directory} and mode {mode}]\n")

# def main():
#     parser = argparse.ArgumentParser(description="Two-Stage Document Search System")
#     parser.add_argument("--dir", required=True, help="Directory with .txt files")
#     parser.add_argument("--mode", choices=["recursive"], default="recursive", help="Chunking strategy")
#     parser.add_argument("--query", help="Query string for searching")
#     parser.add_argument("--search", choices=["exact", "semantic"], help="Search type")
#     parser.add_argument("--rerank", action="store_true", help="Use cross-encoder reranker")
#     parser.add_argument("--ingest", action="store_true", help="Run ingestion/chunking phase")
#     parser.add_argument("--append", action="store_true", help="Append new chunks to existing cache")
#     parser.add_argument("--purge", action="store_true", help="Purge the chunk cache for the directory/mode")
#     args = parser.parse_args()

#     if args.purge:
#         purge_cache(args.dir, args.mode)
#     elif args.ingest:
#         ingest_phase(args.dir, args.mode, append=args.append)
#     elif args.query and args.search:
#         retrieval_phase(args.query, args.search, args.rerank, args.dir, args.mode)
#     else:
#         print("You must specify either --ingest, --purge, or both --query and --search for retrieval.")

# if __name__ == "__main__":
#     main()

# from vector_store.vector_store import VectorStoreManager
# from models.data_models import Chunk, SearchResult
# from chunking.recursive_token_chunker import RecursiveTokenChunker
# from search.semantic_search_engine import SemanticSearchEngine
# from embedding.embedding_manager import EmbeddingManager
# from sklearn.preprocessing import normalize
# import numpy as np
# import argparse
# import os
# import json

# # === Phase 1: Ingesting, Chunking, and Saving ===
# def ingest_phase(directory: str, mode: str = "semantic", append: bool = False):
#     print("[Phase 1: Ingesting, Chunking, and Embedding Documents]")

#     chunker = RecursiveTokenChunker()
#     all_chunks = []

#     filepaths = [
#         os.path.join(directory, f)
#         for f in os.listdir(directory)
#         if f.endswith(".txt") or f.endswith(".md")
#     ]

#     for filepath in filepaths:
#         print(f"Processing file: {os.path.basename(filepath)}")
#         with open(filepath, encoding='utf-8', errors='ignore') as f:
#             text = f.read()
#         chunks = chunker.chunk(text, source_path=filepath, document_name=os.path.basename(filepath))
#         all_chunks.extend(chunks)

#     embedder = EmbeddingManager.get_instance()
#     texts = [chunk.text for chunk in all_chunks if chunk.text and isinstance(chunk.text, str)]
#     text_chunk_map = [chunk for chunk in all_chunks if chunk.text and isinstance(chunk.text, str)]

#     if texts:
#         try:
#             embeddings = embedder.get_embedding(texts)
#             if not isinstance(embeddings, list) or not all(isinstance(e, list) for e in embeddings):
#                 raise ValueError("Invalid embedding format")
#             if len(embeddings) != len(text_chunk_map):
#                 raise ValueError("Mismatch between embeddings and chunks")
#             for chunk, embedding in zip(text_chunk_map, embeddings):
#                 chunk.embedding = embedding
#         except Exception as e:
#             print(f"[Error] Batch embedding failed: {e}")

#     existing_chunks = []
#     if append and os.path.exists("vector_store.json"):
#         with open("vector_store.json", "r", encoding="utf-8") as f:
#             existing_chunks = [Chunk.from_dict(data) for data in json.load(f)]

#     combined_chunks = existing_chunks + all_chunks

#     # Save chunks to disk with embeddings
#     with open("vector_store.json", "w", encoding="utf-8") as f:
#         json.dump([chunk.to_dict() for chunk in combined_chunks], f, ensure_ascii=False, indent=2)

#     print(f"[INFO] Saved {len(combined_chunks)} total chunks with embeddings to vector store")

# # === Phase 2: Query Phase ===
# # def query_phase(query: str, top_k: int = 5):
# #     print("\n[Phase 2: Performing Query/Retrieval]")

# #     # Load all chunks (with or without embeddings)
# #     with open("vector_store.json", "r", encoding="utf-8") as f:
# #         chunks_data = json.load(f)
# #     chunks = [Chunk.from_dict(data) for data in chunks_data]

# #     # Embed the query only
# #     search_engine = SemanticSearchEngine()
# #     results = search_engine.search(query, chunks, top_k=top_k)

# #     if not results:
# #         print("\n[No results to display]")
# #         return

# #     print("\nTop Results:")
# #     for res in results:
# #         print(f"\nRank {res.rank} | Score: {res.score:.4f} | File: {res.document_name}")
# #         print(f"→ {res.text.strip()[:300]}...")
# #         print(f"Explanation: {res.detail}")


# def query_phase(query: str, top_k: int = 5):
#     print("\n[Phase 2: Performing Query/Retrieval]")

#     # Load all chunks with stored embeddings
#     with open("vector_store.json", "r", encoding="utf-8") as f:
#         chunks_data = json.load(f)
#     chunks = [Chunk.from_dict(data) for data in chunks_data if data.get("embedding")]

#     if not chunks:
#         print("[Error] No chunks with embeddings available.")
#         return

#     # Embed the query once using the same model
#     embedder = EmbeddingManager.get_instance()
#     query_vec = np.array(embedder.get_single_embedding(query))

#     # Score chunks using cosine similarity
#     scored = []
#     for chunk in chunks:
#         chunk_vec = np.array(chunk.embedding)
#         score = float(np.dot(query_vec, chunk_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(chunk_vec)))
#         chunk.score = score
#         scored.append((chunk, score))

#     # Sort and get top_k results
#     top_chunks = sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]

#     if not top_chunks:
#         print("\n[No results to display]")
#         return

#     print("\nTop Results:")
#     for i, (chunk, score) in enumerate(top_chunks, start=1):
#         print(f"\nRank {i} | Score: {score:.4f} | File: {chunk.document_name}")
#         print(f"→ {chunk.text}")




# # === Purge Phase ===
# def purge_vector_store():
#     if os.path.exists("vector_store.json"):
#         os.remove("vector_store.json")
#         print("[INFO] Vector store cache purged.")
#     else:
#         print("[INFO] No vector store cache found to purge.")

# # === Entry Point ===
# def main():
#     parser = argparse.ArgumentParser(description="Local Document Search Tool")
#     parser.add_argument("--ingest", action="store_true", help="Run ingestion phase")
#     parser.add_argument("--purge", action="store_true", help="Purge the chunk cache")
#     parser.add_argument("--dir", help="Directory of .txt/.md files to ingest", default="./docs")
#     parser.add_argument("--query", help="Text query to perform semantic search")
#     parser.add_argument("--mode", help="Chunking mode", default="semantic")
#     parser.add_argument("--append", action="store_true", help="Append new chunks to existing vector store")
#     args = parser.parse_args()

#     if args.purge:
#         purge_vector_store()
#     elif args.ingest:
#         ingest_phase(args.dir, args.mode, append=args.append)
#     elif args.query:
#         query_phase(args.query)
#     else:
#         print("[ERROR] Please provide --ingest, --query, or --purge")

# if __name__ == "__main__":
#     main()


# import os
# import json
# import argparse
# import numpy as np
# from chunking.recursive_token_chunker import RecursiveTokenChunker
# from models.data_models import Chunk
# from embedding.embedding_manager import EmbeddingManager
# from search.semantic_search_engine import SemanticSearchEngine
# from search.exact_search_engine import ExactSearchEngine

# # === Utility ===
# def get_vector_store_path(mode):
#     if mode == "semantic":
#         return "vector_store_semantic.json"
#     elif mode == "exact":
#         return "vector_store_exact.json"
#     else:
#         raise ValueError(f"Unsupported mode: {mode}")

# # === Phase 1: Ingestion ===
# def ingest_phase(directory: str, mode: str = "semantic", append: bool = False):
#     print(f"[Phase 1: Ingesting in '{mode}' mode]")
#     chunker = RecursiveTokenChunker()
#     all_chunks = []

#     filepaths = [
#         os.path.join(directory, f)
#         for f in os.listdir(directory)
#         if f.endswith(".txt") or f.endswith(".md")
#     ]

#     for filepath in filepaths:
#         print(f"↪️ Processing file: {os.path.basename(filepath)}")
#         with open(filepath, encoding='utf-8', errors='ignore') as f:
#             text = f.read()
#         chunks = chunker.chunk(text, source_path=filepath, document_name=os.path.basename(filepath))
#         all_chunks.extend(chunks)

#     if mode == "semantic":
#         embedder = EmbeddingManager.get_instance()
#         texts = [chunk.text for chunk in all_chunks if chunk.text]
#         valid_chunks = [chunk for chunk in all_chunks if chunk.text]
#         try:
#             embeddings = embedder.get_embedding(texts)
#             if len(embeddings) != len(valid_chunks):
#                 raise ValueError("Mismatch between embeddings and chunks")
#             for chunk, emb in zip(valid_chunks, embeddings):
#                 chunk.embedding = emb
#         except Exception as e:
#             print(f"[Error] Embedding failed: {e}")

#     path = get_vector_store_path(mode)
#     existing_chunks = []
#     if append and os.path.exists(path):
#         with open(path, "r", encoding="utf-8") as f:
#             existing_chunks = [Chunk.from_dict(data) for data in json.load(f)]

#     with open(path, "w", encoding="utf-8") as f:
#         json.dump([chunk.to_dict() for chunk in (existing_chunks + all_chunks)], f, indent=2)

#     print(f"[INFO] Saved {len(existing_chunks + all_chunks)} chunks to {path}")

# # === Phase 2: Query ===
# def query_phase(query: str, mode: str = "semantic", top_k: int = 5):
#     print(f"\n[Phase 2: Querying in '{mode}' mode]")
#     path = get_vector_store_path(mode)

#     if not os.path.exists(path):
#         print(f"[Error] No vector store found at: {path}")
#         return

#     with open(path, "r", encoding="utf-8") as f:
#         chunks_data = json.load(f)
#     chunks = [Chunk.from_dict(data) for data in chunks_data]

#     if mode == "semantic":
#         embedder = EmbeddingManager.get_instance()
#         query_vec = np.array(embedder.get_single_embedding(query))
#         scored = []
#         for chunk in chunks:
#             if chunk.embedding:
#                 chunk_vec = np.array(chunk.embedding)
#                 sim = float(np.dot(query_vec, chunk_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(chunk_vec)))
#                 chunk.score = sim
#                 scored.append((chunk, sim))
#         results = sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]

#     elif mode == "exact":
#         engine = ExactSearchEngine()
#         results = engine.search(query, chunks, top_k=top_k)
#         results = [(r, r.score) for r in results]

#     else:
#         print(f"[Error] Unsupported mode: {mode}")
#         return

#     print("\nTop Results:")
#     for i, (chunk, score) in enumerate(results, start=1):
#         print(f"\nRank {i} | Score: {score:.4f} | File: {chunk.document_name}")
#         print(f"→ {chunk.text.strip()[:300]}")

# # === Purge Cache ===
# def purge_vector_store():
#     for mode in ["semantic", "exact"]:
#         path = get_vector_store_path(mode)
#         if os.path.exists(path):
#             os.remove(path)
#             print(f"[INFO] Purged: {path}")
#         else:
#             print(f"[INFO] No vector store to purge for {mode}")

# # === Entry Point ===
# def main():
#     parser = argparse.ArgumentParser(description="Document Search App")
#     parser.add_argument("--ingest", action="store_true")
#     parser.add_argument("--query", help="Text to query")
#     parser.add_argument("--mode", choices=["semantic", "exact"], default="semantic")
#     parser.add_argument("--dir", help="Directory of input files", default="./data")
#     parser.add_argument("--append", action="store_true")
#     parser.add_argument("--purge", action="store_true")
#     args = parser.parse_args()

#     if args.purge:
#         purge_vector_store()
#     elif args.ingest:
#         ingest_phase(args.dir, args.mode, append=args.append)
#     elif args.query:
#         query_phase(args.query, args.mode)
#     else:
#         print("[ERROR] Provide --ingest, --query, or --purge")

# if __name__ == "__main__":
#     main()
# import argparse
# import os
# import json
# import numpy as np

# from models.data_models import Chunk
# from chunking.recursive_token_chunker import RecursiveTokenChunker
# #from chunking.semantic_cluster_chunker import SemanticClusterChunker
# from embedding.embedding_manager import EmbeddingManager
# from search.semantic_search_engine import SemanticSearchEngine
# from search.exact_search_engine import ExactSearchEngine

# # === Phase 1: Ingesting, Chunking, and Saving ===
# def ingest_phase(directory: str, mode: str = "semantic", append: bool = False):
#     print(f"[Phase 1: Ingesting in {mode} mode]")

#     chunker = RecursiveTokenChunker() if mode == "exact" else RecursiveTokenChunker()
#     all_chunks = []

#     filepaths = [
#         os.path.join(directory, f)
#         for f in os.listdir(directory)
#         if f.endswith(".txt") or f.endswith(".md")
#     ]

#     for filepath in filepaths:
#         print(f"Processing file: {os.path.basename(filepath)}")
#         with open(filepath, encoding='utf-8', errors='ignore') as f:
#             text = f.read()
#         chunks = chunker.chunk(text, source_path=filepath, document_name=os.path.basename(filepath))
#         all_chunks.extend(chunks)

#     if mode == "semantic":
#         embedder = EmbeddingManager.get_instance()
#         texts = [chunk.text for chunk in all_chunks if chunk.text and isinstance(chunk.text, str)]
#         text_chunk_map = [chunk for chunk in all_chunks if chunk.text and isinstance(chunk.text, str)]

#         if texts:
#             try:
#                 embeddings = embedder.get_embedding(texts)
#                 if not isinstance(embeddings, list) or not all(isinstance(e, list) for e in embeddings):
#                     raise ValueError("Invalid embedding format")
#                 if len(embeddings) != len(text_chunk_map):
#                     raise ValueError("Mismatch between embeddings and chunks")
#                 for chunk, embedding in zip(text_chunk_map, embeddings):
#                     chunk.embedding = embedding
#             except Exception as e:
#                 print(f"[Error] Batch embedding failed: {e}")

#     json_path = f"vector_store_{mode}.json"

#     existing_chunks = []
#     if append and os.path.exists(json_path):
#         with open(json_path, "r", encoding="utf-8") as f:
#             existing_chunks = [Chunk.from_dict(data) for data in json.load(f)]

#     combined_chunks = existing_chunks + all_chunks

#     with open(json_path, "w", encoding="utf-8") as f:
#         json.dump([chunk.to_dict() for chunk in combined_chunks], f, ensure_ascii=False, indent=2)

#     print(f"[INFO] Saved {len(combined_chunks)} chunks to {json_path}")

# # === Phase 2: Query Phase ===
# def query_phase(query: str, mode: str = "semantic"):
#     print(f"\n[Phase 2: Performing {mode.capitalize()} Query]")
#     json_path = f"vector_store_{mode}.json"

#     if not os.path.exists(json_path):
#         print(f"[Error] No vector store found for mode: {mode}")
#         return

#     with open(json_path, "r", encoding="utf-8") as f:
#         chunks_data = json.load(f)
#     chunks = [Chunk.from_dict(data) for data in chunks_data if (mode == "exact" or data.get("embedding"))]

#     if not chunks:
#         print("[Error] No chunks found for query")
#         return

#     top_k = 5

#     if mode == "semantic":
#         embedder = EmbeddingManager.get_instance()
#         query_vec = np.array(embedder.get_single_embedding(query))

#         scored = []
#         for chunk in chunks:
#             chunk_vec = np.array(chunk.embedding)
#             score = float(np.dot(query_vec, chunk_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(chunk_vec)))
#             chunk.score = score
#             scored.append((chunk, score))

#         top_chunks = sorted(scored, key=lambda x: x[1], reverse=True)[:top_k]

#         if not top_chunks:
#             print("\n[No results to display]")
#             return

#         print("\nTop Results:")
#         for i, (chunk, score) in enumerate(top_chunks, start=1):
#             print(f"\nRank {i} | Score: {score:.4f} | File: {chunk.document_name}")
#             print(f"→ {chunk.text}")
#             print(f"{chunk.detail}")

#     elif mode == "exact":
#         engine = ExactSearchEngine()
#         results = engine.search(query, chunks)

#         if not results:
#             print("\n[No results to display]")
#             return

#         print("\nTop Results:")
#         for res in results:
#             print(f"\nRank {res.rank} | Score: {res.score:.4f} | File: {res.document_name}")
#             print(f"→ {res.text.strip()[:300]}...")

# # === Purge Phase ===
# def purge_vector_store(mode: str = "semantic"):
#     json_path = f"vector_store_{mode}.json"
#     if os.path.exists(json_path):
#         os.remove(json_path)
#         print(f"[INFO] {json_path} purged.")
#     else:
#         print(f"[INFO] No vector store found at {json_path} to purge.")

# # === Entry Point ===
# def main():
#     parser = argparse.ArgumentParser(description="Local Document Search Tool")
#     parser.add_argument("--ingest", action="store_true", help="Run ingestion phase")
#     parser.add_argument("--purge", action="store_true", help="Purge the chunk cache")
#     parser.add_argument("--dir", help="Directory of .txt/.md files to ingest", default="./docs")
#     parser.add_argument("--query", help="Text query to perform search")
#     parser.add_argument("--mode", help="Choose 'semantic' or 'exact'", default="semantic")
#     parser.add_argument("--append", action="store_true", help="Append new chunks to existing vector store")
#     args = parser.parse_args()

#     if args.purge:
#         purge_vector_store(args.mode)
#     elif args.ingest:
#         ingest_phase(args.dir, args.mode, append=args.append)
#     elif args.query:
#         query_phase(args.query, args.mode)
#     else:
#         print("[ERROR] Please provide --ingest, --query, or --purge")

# if __name__ == "__main__":
#     main()


from vector_store.vector_store import VectorStoreManager
from models.data_models import Chunk, SearchResult
from chunking.recursive_token_chunker import RecursiveTokenChunker
from search.semantic_search_engine import SemanticSearchEngine
from search.exact_search_engine import ExactSearchEngine
from embedding.embedding_manager import EmbeddingManager
import numpy as np
import argparse
import os
import json

# === Phase 1: Ingesting, Chunking, and Saving ===
def ingest_phase(directory: str, mode: str = "semantic", append: bool = False):
    print("[Phase 1: Ingesting, Chunking, and Embedding Documents]")

    chunker = RecursiveTokenChunker()
    all_chunks = []

    filepaths = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith(".txt") or f.endswith(".md")
    ]

    for filepath in filepaths:
        print(f"Processing file: {os.path.basename(filepath)}")
        with open(filepath, encoding='utf-8', errors='ignore') as f:
            text = f.read()
        chunks = chunker.chunk(text, source_path=filepath, document_name=os.path.basename(filepath))
        all_chunks.extend(chunks)

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

    existing_chunks = []
    path = "vector_store_semantic.json" if mode == "semantic" else "vector_store_exact.json"
    if append and os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            existing_chunks = [Chunk.from_dict(data) for data in json.load(f)]

    combined_chunks = existing_chunks + all_chunks

    # Save chunks to disk with embeddings
    with open(path, "w", encoding="utf-8") as f:
        json.dump([chunk.to_dict() for chunk in combined_chunks], f, ensure_ascii=False, indent=2)

    print(f"[INFO] Saved {len(combined_chunks)} total chunks with embeddings to {path}")

# === Phase 2: Query Phase ===
def query_phase(query: str, mode: str = "semantic"):
    print("\n[Phase 2: Performing Query/Retrieval]")

    path = "vector_store_semantic.json" if mode == "semantic" else "vector_store_exact.json"
    with open(path, "r", encoding="utf-8") as f:
        chunks_data = json.load(f)
    chunks = [Chunk.from_dict(data) for data in chunks_data if mode == "exact" or data.get("embedding")]

    if not chunks:
        print("[Error] No chunks available.")
        return

    if mode == "semantic":
        engine = SemanticSearchEngine()
        results = engine.search(query, chunks, top_k=5)
    else:
        engine = ExactSearchEngine()
        results = engine.search(query, chunks)

    if not results:
        print("\n[No results to display]")
        return

    print("\nTop Results:")
    for res in results:
        print(f"\nRank {res.rank} | Score: {res.score:.4f} | File: {res.document_name}")
        print(f"→ {res.text.strip()[:300]}...")
        if res.detail:
            print(f"Explanation: {res.detail}")

# === Purge Phase ===
def purge_vector_store():
    for file in ["vector_store_semantic.json", "vector_store_exact.json"]:
        if os.path.exists(file):
            os.remove(file)
            print(f"[INFO] {file} purged.")
        else:
            print(f"[INFO] {file} not found.")

# === Entry Point ===
def main():
    parser = argparse.ArgumentParser(description="Local Document Search Tool")
    parser.add_argument("--ingest", action="store_true", help="Run ingestion phase")
    parser.add_argument("--purge", action="store_true", help="Purge the chunk cache")
    parser.add_argument("--dir", help="Directory of .txt/.md files to ingest", default="./docs")
    parser.add_argument("--query", help="Text query to perform search")
    parser.add_argument("--mode", help="Choose search mode: 'semantic' or 'exact'", default="semantic")
    parser.add_argument("--append", action="store_true", help="Append new chunks to existing vector store")
    args = parser.parse_args()

    if args.purge:
        purge_vector_store()
    elif args.ingest:
        ingest_phase(args.dir, args.mode, append=args.append)
    elif args.query:
        query_phase(args.query, args.mode)
    else:
        print("[ERROR] Please provide --ingest, --query, or --purge")

if __name__ == "__main__":
    main()

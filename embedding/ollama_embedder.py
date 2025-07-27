# # # # ollama_embedder.py

# # # from typing import List
# # # from transformers import AutoTokenizer
# # # import ollama

# # # MAX_TOKENS = 512  # Most MiniLM-style models have a 512-token limit

# # # class OllamaEmbedder:
# # #     def __init__(self, model_name: str = "mxbai-embed-large"):
# # #         self.model_name = model_name
# # #         self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2", local_files_only=True)

# # #     def get_embedding(self, text: str) -> List[float]:
# # #         inputs = self.tokenizer(
# # #             text,
# # #             max_length=MAX_TOKENS,
# # #             truncation=True,
# # #             return_tensors="pt",
# # #             return_attention_mask=False
# # #         )
# # #         input_ids = inputs["input_ids"][0]
# # #         if input_ids.shape[0] >= MAX_TOKENS:
# # #             print(f"[Truncated input to {MAX_TOKENS} tokens for embedding]")
# # #         truncated_text = self.tokenizer.decode(input_ids, skip_special_tokens=True)
# # #         response = ollama.embeddings(model=self.model_name, prompt=truncated_text)
# # #         return response["embedding"]



# # from transformers import AutoTokenizer
# # import numpy as np
# # import ollama
# # from typing import List

# # MAX_TOKENS = 512

# # class OllamaEmbedder:
# #     def __init__(self, model_name: str = "mxbai-embed-large"):
# #         self.model_name = model_name
# #         self.tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2", local_files_only=True)

# #     def get_embedding(self, text: str) -> List[float]:
# #         tokens = self.tokenizer.encode(text, add_special_tokens=False)
        
# #         # If small enough, just embed directly
# #         if len(tokens) <= MAX_TOKENS:
# #             truncated_text = self.tokenizer.decode(tokens, skip_special_tokens=True)
# #             return ollama.embeddings(model=self.model_name, prompt=truncated_text)["embedding"]

# #         # Otherwise process in batches
# #         print(f"[Batching] Input is {len(tokens)} tokens. Splitting into batches of {MAX_TOKENS}...")
# #         sub_embeddings = []

# #         for i in range(0, len(tokens), MAX_TOKENS):
# #             token_slice = tokens[i:i+MAX_TOKENS]
# #             prompt = self.tokenizer.decode(token_slice, skip_special_tokens=True)
# #             result = ollama.embeddings(model=self.model_name, prompt=prompt)
# #             embedding = result.get("embedding")
# #             if embedding:
# #                 sub_embeddings.append(np.array(embedding))

# #         if not sub_embeddings:
# #             raise ValueError("Failed to get any valid embeddings from Ollama.")

# #         # Average all the slices
# #         averaged = np.mean(sub_embeddings, axis=0)
# #         return averaged.tolist()

# # ollama_embedder.py
# # ollama_embedder.py

# # from transformers import AutoTokenizer
# # import numpy as np
# # import ollama
# # from typing import List

# # MAX_TOKENS = 512

# # class OllamaEmbedder:
# #     def __init__(self, model_name: str = "mxbai-embed-large"):
# #         self.model_name = model_name
# #         self.tokenizer = AutoTokenizer.from_pretrained(
# #             "sentence-transformers/all-MiniLM-L6-v2",
# #             local_files_only=True
# #         )
# #     def get_embedding(self, texts: List[str]) -> List[List[float]]:
# #         embeddings = []

# #         for text in texts:
# #             # You can still tokenize if needed
# #             _ = self.tokenizer.encode(text, add_special_tokens=False)

# #             response = ollama.embeddings(model=self.model_name, prompt=text)
# #             if "embedding" in response:
# #                 embeddings.append(response["embedding"])
# #             else:
# #                 print(f"[Warning] Embedding failed for text: {text[:60]}...")
# #                 embeddings.append([0.0] * 384)  # fallback to zero vector of fixed dim

# #         return embeddings


# from transformers import AutoTokenizer
# import numpy as np
# import ollama
# from typing import List, Union

# MAX_TOKENS = 512

# class OllamaEmbedder:
#     def __init__(self, model_name: str = "mxbai-embed-large"):
#         self.model_name = model_name
#         self.tokenizer = AutoTokenizer.from_pretrained(
#             "sentence-transformers/all-MiniLM-L6-v2",
#             local_files_only=True
#         )

#     def get_embedding(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
#         if isinstance(texts, str):
#             texts = [texts]

#         try:
#             response = ollama.embeddings(model=self.model_name, prompt=texts)
#             embeddings = response.get("embeddings")

#             if embeddings is None or not isinstance(embeddings, list):
#                 raise ValueError("Embeddings response was invalid.")

#             return embeddings

#         except Exception as e:
#             print(f"[Error] Batch embedding failed: {e}")
#             return [[0.0] * 384 for _ in texts]  # fallback to dummy embeddings



# from transformers import AutoTokenizer
# import numpy as np
# import ollama
# from typing import List, Union

# MAX_TOKENS = 512

# class OllamaEmbedder:
#     def __init__(self, model_name: str = "nomic-embed-text"):
#         self.model_name = model_name
#         self.tokenizer = AutoTokenizer.from_pretrained(
#             "sentence-transformers/all-MiniLM-L6-v2",
#             local_files_only=True
#         )

#     def get_embedding(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
#         if isinstance(texts, str):
#             texts = [texts]

#         results = []
#         for text in texts:
#             try:
#                 response = ollama.embeddings(model=self.model_name, prompt=text)
#                 emb = response.get("embedding")
#                 if emb is None:
#                     raise ValueError("No embedding returned")
#                 results.append(emb)
#             except Exception as e:
#                 print(f"[Error] Embedding failed for text: {text[:30]}... → {e}")
#                 results.append([0.0] * 1024)  # or 384 depending on model
#         return results

#     def get_single_embedding(self, text: str) -> List[float]:
#         if not isinstance(text, str):
#             raise ValueError("get_single_embedding expects a string input")
#         try:
#             response = ollama.embeddings(model=self.model_name, prompt=text)
#             emb = response["embedding"]
#             print(f"[OllamaEmbedder] Query embedding shape: {len(emb)}")
#             return emb
#         except Exception as e:
#             print(f"[Error] Single embedding failed: {e}")
#             return [0.0] * 1024  # fallback should match expected dimension


# from transformers import AutoTokenizer
# import numpy as np
# import ollama
# from typing import List, Union
# from concurrent.futures import ThreadPoolExecutor, as_completed

# MAX_TOKENS = 512
# THREAD_BATCH_SIZE = 8

# class OllamaEmbedder:
#     def __init__(self, model_name: str = "nomic-embed-text"):
#         self.model_name = model_name
#         self.tokenizer = AutoTokenizer.from_pretrained(
#             "sentence-transformers/all-MiniLM-L6-v2",
#             local_files_only=True
#         )

#     def get_embedding(self, texts: Union[str, List[str]]) -> Union[List[float], List[List[float]]]:
#         if isinstance(texts, str):
#             texts = [texts]

#         def embed_text(text: str):
#             try:
#                 response = ollama.embeddings(model=self.model_name, prompt=text)
#                 emb = response.get("embedding")
#                 if emb is None:
#                     raise ValueError("No embedding returned")
#                 return emb
#             except Exception as e:
#                 print(f"[Error] Embedding failed for text: {text[:30]}... → {e}")
#                 return [0.0] * 1024  # fallback vector

#         embeddings = []
#         with ThreadPoolExecutor(max_workers=THREAD_BATCH_SIZE) as executor:
#             future_to_text = {executor.submit(embed_text, text): text for text in texts}
#             for future in as_completed(future_to_text):
#                 embeddings.append(future.result())

#         return embeddings

#     def get_single_embedding(self, text: str) -> List[float]:
#         if not isinstance(text, str):
#             raise ValueError("get_single_embedding expects a string input")
#         try:
#             response = ollama.embeddings(model=self.model_name, prompt=text)
#             emb = response["embedding"]
#             print(f"[OllamaEmbedder] Query embedding shape: {len(emb)}")
#             return emb
#         except Exception as e:
#             print(f"[Error] Single embedding failed: {e}")
#             return [0.0] * 1024


import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from transformers import AutoTokenizer
import numpy as np
import ollama
from typing import List, Union

MAX_TOKENS = 512
THREAD_BATCH_SIZE = 8

class OllamaEmbedder:
    def __init__(self, model_name: str = "all-minilm"):
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

        print(f"[✅ Complete] {count} embeddings in {total_time:.2f}s → {count / total_time:.2f} embeds/sec")
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

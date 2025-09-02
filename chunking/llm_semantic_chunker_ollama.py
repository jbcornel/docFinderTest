
from __future__ import annotations
from typing import List, Optional, Tuple
import re
import time

import ollama

from .base_chunker import BaseChunker
from chunking.recursive_token_chunker import RecursiveTokenChunker
from models.data_models import Chunk


def _approx_token_count(s: str) -> int:
    
    if not isinstance(s, str):
        return 0
    return max(1, len(s) // 4)


def _build_system_prompt() -> str:
    return (
        "You are an assistant specialized in splitting text into thematically consistent sections.\n"
        "The text is provided as short chunks, each delimited by <|start_chunk_X|> and <|end_chunk_X|>, "
        "where X is the chunk number (1-based).\n"
        "Your task: identify split points where a new topic begins, keeping consecutive chunks of similar themes together.\n"
        "Respond ONLY with a single line starting with 'split_after:' followed by a comma-separated list of chunk IDs.\n"
        "Example: split_after: 3, 5\n"
        "Rules:\n"
        " - IDs MUST be in ascending order\n"
        " - IDs MUST be >= the provided lower bound\n"
        " - Provide AT LEAST ONE split if possible\n"
    )


def _build_user_prompt(chunked_input: str, lower_bound: int, invalid_hint: Optional[List[int]] = None) -> str:
    hint = (
        f"\nThe previous response {invalid_hint} violated the rules. "
        "DO NOT REPEAT that sequence. Respond correctly this time."
        if invalid_hint else ""
    )
    return (
        "CHUNKED_TEXT:\n"
        f"{chunked_input}\n\n"
        f"Respond ONLY with the IDs in the form 'split_after: a, b, c'. "
        f"IDs MUST be in ascending order and each MUST be >= {lower_bound}.{hint}"
    )


class LLMSemanticChunkerOllama(BaseChunker):
    """
    LLM-guided semantic chunker using Ollama models.

    Flow:
      1) Use small base chunks via RecursiveTokenChunker (configurable).
      2) Send a window of these base chunks to Ollama with instructions to propose split points.
      3) Validate + collect split indices; merge base chunks accordingly.
      4) Return List[Chunk] (text, source_path, document_name), with 'ordinal' set for stability.
    """

    def __init__(
        self,
        model_name: str = "phi3:mini",
        temperature: float = 0.2,
        base_chunk_size: int = 50,
        window_token_budget: int = 800,
        max_retries: int = 3,
        base_tokenizer_model: Optional[str] = None,
        base_chunk_overlap: Optional[int] = None, 
        
        debug: bool = False,
        log_every: int = 1,
        preview_chars: int = 120,
    ):
        self.model_name = model_name
        self.temperature = float(temperature)
        self.window_token_budget = int(window_token_budget)
        self.max_retries = int(max_retries)

        self.debug = bool(debug)
        self.log_every = max(1, int(log_every))
        self.preview_chars = max(40, int(preview_chars))

       
        if base_tokenizer_model:
            self.splitter = RecursiveTokenChunker(
                model_name=base_tokenizer_model,
                chunk_size=base_chunk_size,
            )
        else:
            self.splitter = RecursiveTokenChunker(
                chunk_size=base_chunk_size,
            )

    
    def _dbg(self, msg: str) -> None:
        if self.debug:
            print(f"[LLM-SEM-CHUNK] {msg}")

   
    def _llm_split_suggestion(
        self,
        chunked_input: str,
        lower_bound: int,
        invalid_hint: Optional[List[int]] = None,
        window_index: int = 0,
        span: Optional[Tuple[int, int]] = None,
    ) -> List[int]:
        """
        Ask the model for 'split_after: a, b, c' line; parse and validate.
        Retries up to self.max_retries times on invalid format or exceptions.
        """
        sys = _build_system_prompt()
        user = _build_user_prompt(chunked_input, lower_bound, invalid_hint)

        invalid_prev: Optional[List[int]] = invalid_hint
        last_err: Optional[Exception] = None

        for attempt in range(self.max_retries):
            t0 = time.time()
            try:
                resp = ollama.chat(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": sys},
                        {"role": "user", "content": user},
                    ],
                    options={
                        "temperature": self.temperature,
                        "num_predict": 200,
                    },
                )
                dur = time.time() - t0

           
                content = ""
                if isinstance(resp, dict):
                    msg = resp.get("message") or {}
                    content = (msg.get("content") or "").strip()

               
                line = ""
                for ln in (content.splitlines() if content else []):
                    if "split_after:" in ln.lower():
                        line = ln
                        break
                if not line:
                    line = content

                nums = re.findall(r"\d+", line or "")
                ints = list(map(int, nums)) if nums else []

                if self.debug:
                    span_txt = f"{span[0]}..{span[1]}" if span else "?"
                    preview = (chunked_input[:self.preview_chars] + "…") if len(chunked_input) > self.preview_chars else chunked_input
                    preview = preview.replace("\n", " ") 

                    self._dbg(
                        f"window#{window_index} attempt {attempt+1}/{self.max_retries} "
                        f"span[{span_txt}] dur={dur:.2f}s -> raw='{line.strip()}' parsed={ints} "
                        f"| input.preview='{preview}'"
                    )


             
                if ints and ints == sorted(ints) and all(n >= lower_bound for n in ints):
                    return ints

                
                invalid_prev = ints or invalid_prev
                user = _build_user_prompt(chunked_input, lower_bound, invalid_prev)

            except Exception as e:
                last_err = e
                dur = time.time() - t0
                self._dbg(f"window#{window_index} attempt {attempt+1}/{self.max_retries} EXC after {dur:.2f}s: {e!r}")
                time.sleep(0.5 * (attempt + 1))

       
        if self.debug:
            self._dbg(
                f"window#{window_index} giving fallback split at lower_bound={lower_bound}"
                + (f" (last error: {last_err!r})" if last_err else "")
            )
        return [max(lower_bound, 1)]

   
    def chunk(self, text: str, source_path: str, document_name: str) -> List[Chunk]:
        """
        Produce final chunk texts based on LLM-proposed boundaries.
        Returns Chunk objects compatible with your pipeline.
        """
        start_total = time.time()


        base_chunk_objs: List[Chunk] = self.splitter.chunk(text, source_path=source_path, document_name=document_name)
        base_chunks: List[str] = [c.text for c in base_chunk_objs if isinstance(c.text, str) and c.text.strip()]
        n_base = len(base_chunks)
        self._dbg(f"file='{document_name}' base_chunks={n_base} (model={self.model_name}, window_budget~{self.window_token_budget})")

        if not base_chunks:
            self._dbg("No base chunks produced; returning empty.")
            return []

        split_indices: List[int] = []  
        current_idx = 0                
        window_index = 0
        windows_made = 0

        while True:
            if current_idx >= n_base - 1:
                break

            token_count = 0
            chunked_input = ""
            span_start = current_idx
           
            for i in range(current_idx, n_base):
                token_count += _approx_token_count(base_chunks[i])
            
                chunked_input += f"<|start_chunk_{i+1}|>{base_chunks[i]}<|end_chunk_{i+1}|>"
                if token_count > self.window_token_budget:
                    break
            span_end = i  
            windows_made += 1
            window_index += 1

            
            if self.debug and (window_index == 1 or (window_index % self.log_every == 0)):
                self._dbg(
                    f"window#{window_index} covering base[{span_start+1}..{span_end+1}] "
                    f"~tokens={token_count}, lower_bound={current_idx+1}"
                )

         
            t0 = time.time()
            suggested = self._llm_split_suggestion(
                chunked_input=chunked_input,
                lower_bound=current_idx + 1,
                invalid_hint=None,
                window_index=window_index,
                span=(span_start + 1, span_end + 1),  
            )
            t_call = time.time() - t0

            if suggested:
                split_indices.extend(suggested)
                current_idx = suggested[-1] 
                if self.debug:
                    self._dbg(
                        f"window#{window_index} accepted splits={suggested} "
                        f"(advance current_idx -> {current_idx}, call={t_call:.2f}s)"
                    )
            else:
                current_idx += 1
                if self.debug:
                    self._dbg(
                        f"window#{window_index} no parseable splits, advance current_idx -> {current_idx} "
                        f"(call={t_call:.2f}s)"
                    )

        
        split_after_zero_based = [i - 1 for i in split_indices if i - 1 >= 0]

        # Merge base chunks according to split points
        docs: List[str] = []
        buf = ""
        for i, ch_text in enumerate(base_chunks):
            buf += (ch_text + " ")
            if i in split_after_zero_based:
                docs.append(buf.strip())
                buf = ""
        if buf.strip():
            docs.append(buf.strip())

        # Convert to final Chunk objects and assign ordinals
        out: List[Chunk] = []
        for idx, body in enumerate(docs):
            out.append(Chunk(
                text=body,
                source_path=source_path,
                document_name=document_name,
            ))
            setattr(out[-1], "ordinal", idx)

        total_dur = time.time() - start_total
        if self.debug:
            self._dbg(
                f"completed file='{document_name}' windows={windows_made}, "
                f"final_chunks={len(out)}, elapsed={total_dur:.2f}s"
            )

        return out

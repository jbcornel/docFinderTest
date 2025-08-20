import os
import re
import hashlib
from typing import List, Optional
from concurrent.futures import ThreadPoolExecutor, as_completed

import ollama


def _resolve_model(alias_or_name: Optional[str]) -> str:
    """
    Resolve a friendly alias or a concrete Ollama model id.
    Default is llama3:latest.
    """
    default_id = "llama3:latest"
    if not alias_or_name:
        return os.getenv("OLLAMA_SUMMARY_MODEL", default_id)

    name = alias_or_name.strip().lower()
    if name in ("llama3", "llama-3", "llama3:latest"):
        return "llama3:latest"
    if name in ("phi3", "phi-3", "phi3:mini", "phi-3:mini"):
        return "phi3:mini"
    return alias_or_name



def _key(text: str, model_id: str) -> str:
    """Cache key includes model id so switching models changes outputs."""
    payload = f"{model_id}||{text}".encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def _clean_summary(s: str) -> str:
   
    if not isinstance(s, str):
        return ""
    s = s.strip()

    # Strip common prefixes & instruction echoes
    patterns = [
        r'^\s*(summary|tl;dr)\s*:\s*',
        r'^\s*\d+\.\s*(summarize|avoid|stick to)\b.*',
        r'\b(in|using)\s+1[-–]?\s*2\s+sentences\b.*',
        r'^(task|instruction)s?\s*:\s*.*', 
        r'^output\s+only.*',
    ]
    for p in patterns:
        s = re.sub(p, '', s, flags=re.IGNORECASE | re.MULTILINE).strip()

    # Drop a first line if it still looks like an instruction
    lines = [ln.strip() for ln in s.splitlines() if ln.strip()]
    if lines and (
        "summarize the passage" in lines[0].lower()
        or "avoid speculation" in lines[0].lower()
        or lines[0].lower().startswith("1.")
    ):
        lines = lines[1:]

    return "\n".join(lines).strip()



def _llm_call_chat(model_id: str, passage: str, max_tokens: int, temperature: float) -> str:

    messages = [
        {
            "role": "system",
            "content": "You are a concise summarizer. Output only the summary text—no preamble.",
        },
        {
            "role": "user",
            "content": f"Summarize the passage in 1–2 sentences:\n\n{passage}",
        },
    ]
    resp = ollama.chat(
        model=model_id,
        messages=messages,
        options={
            "temperature": float(temperature),
            "num_predict": int(max_tokens),
            "stop": ["Summary:", "PASSAGE:", "\n\nSummary", "\nSummary:", "\nInstructions:"],
        },
    )
    return (resp.get("message") or {}).get("content", "").strip()


class Summarizer:
    def __init__(
        self,
        model_name: Optional[str] = None,
        max_tokens: int = 80,
        temperature: float = 0.0,
        workers: int = 4,
        disable_cache: bool = False,
    ):
        
        self.model_id = _resolve_model(model_name)
        self.max_tokens = int(max_tokens)
        self.temperature = float(temperature)
        self.workers = max(1, int(workers))
        self.disable_cache = bool(disable_cache)
        self.cache = {}

    def summarize(self, text: str) -> str:
        if not isinstance(text, str) or not text.strip():
            return ""

        ck = _key(text, self.model_id)
        if not self.disable_cache and ck in self.cache:
            return self.cache[ck]

        raw = _llm_call_chat(self.model_id, text, self.max_tokens, self.temperature)
        summary = _clean_summary(raw) or "[Summary unavailable]"

        if not self.disable_cache:
            self.cache[ck] = summary
        return summary

    def batch(self, texts: List[str]) -> List[str]:
        """
        Summarize many texts in parallel using ThreadPoolExecutor.
       
        """
        if not texts:
            return []

        results: List[str] = ["" for _ in texts]
        to_submit = []

        # fill cache hits; queue misses
        for i, t in enumerate(texts):
            if not isinstance(t, str) or not t.strip():
                results[i] = ""
                continue
            ck = _key(t, self.model_id)
            if not self.disable_cache and ck in self.cache:
                results[i] = self.cache[ck]
            else:
                to_submit.append(i)

        if not to_submit:
            return results

        with ThreadPoolExecutor(max_workers=self.workers) as ex:
            futs = {ex.submit(self.summarize, texts[i]): i for i in to_submit}
            for fut in as_completed(futs):
                idx = futs[fut]
                try:
                    results[idx] = fut.result()
                except Exception as e:
                    results[idx] = f"[Summary unavailable: {e}]"
        return results

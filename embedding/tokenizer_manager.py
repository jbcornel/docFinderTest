# embedding/tokenizer_manager.py

from transformers import AutoTokenizer

class TokenizerManager:
    _tokenizer = None

    @staticmethod
    def get_tokenizer(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
        if TokenizerManager._tokenizer is None:
            TokenizerManager._tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                local_files_only=True  # Enforces local-only use
            )
        return TokenizerManager._tokenizer


import os
from typing import List, Tuple

class DirectoryLoader:
    def __init__(self, directory: str, extensions: Tuple[str, ...] = (".txt",)):
        self.directory = directory
        self.extensions = extensions

    def load_documents(self) -> List[Tuple[str, str]]:
        
        documents = []
        for root, _, files in os.walk(self.directory):
            for filename in files:
                if filename.endswith(self.extensions):
                    path = os.path.join(root, filename)
                    with open(path, 'r', encoding='utf-8') as file:
                        content = file.read()
                        documents.append((path, content))
        return documents

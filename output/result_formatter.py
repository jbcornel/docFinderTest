
from models.data_models import SearchResult
from typing import List

class ResultFormatter:
    @staticmethod
    def format_results(results: List[SearchResult]) -> str:
        lines = []
        for res in results:
            chunk = res.chunk
            lines.append(f"#{res.rank or '[?]'} - {res.document_name or (chunk.document_name if chunk else '[Unknown]')}")
            lines.append(f"Path: {res.source_path or (chunk.source_path if chunk else '[Unknown path]')}")

            if res.mode == 'exact':
                lines.append("Full Text with Highlights:")
                lines.append(res.text.strip())
                lines.append(f"BM25 Score: {res.detail}")
            elif res.mode == 'semantic':
                summary = getattr(chunk, 'summary', '') or res.text[:200]
                lines.append(f"Summary: {summary.strip()}...")
                lines.append("Full Text with Highlights:")
                lines.append(res.text.strip())
                lines.append(f"Why it's relevant:")
                lines.append(res.detail.strip())
            else:
                lines.append("[Unknown search mode]")
                lines.append(res.text.strip())

            lines.append("-" * 60)
        return "\n".join(lines)

    @staticmethod
    def print_results(results: List[SearchResult]) -> None:
        output = ResultFormatter.format_results(results)
        if output.strip():
            print(output)
        else:
            print("[No results to display]")

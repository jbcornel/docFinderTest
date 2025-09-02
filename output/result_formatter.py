# from models.data_models import SearchResult
# from typing import List, Optional

# EXACT_BONUS_PER_MATCH = 10.0
# PARTIAL_BONUS_PER_MATCH = 2.0

# # Optional, just to keep output tidy
# _MAX_SUMMARY_CHARS = 400
# _MAX_TEXT_CHARS = 1200

# def _truncate(s: str, n: int) -> str:
#     if not isinstance(s, str):
#         return ""
#     s = s.strip()
#     return (s[:n] + "...") if len(s) > n else s

# class ResultFormatter:
#     @staticmethod
#     def _format_exact_breakdown(res: SearchResult) -> List[str]:
#         lines = []
#         meta = getattr(res, "meta", {}) or {}


#         bm25 = float(meta.get("bm25_score", 0.0))
#         exact_cnt = int(meta.get("exact_match_count", 0) or 0)
#         partial_cnt = int(meta.get("partial_match_count", 0) or 0)

#        #legacy option still here but should be removed in future
#         fuzzy_bonus = meta.get("fuzzy_bonus", None)
#         fuzzy_cnt = meta.get("fuzzy_match_count", None)
#         fuzzy_threshold = meta.get("fuzzy_threshold", None)

#         exact_bonus = EXACT_BONUS_PER_MATCH * exact_cnt
#         partial_bonus = PARTIAL_BONUS_PER_MATCH * partial_cnt

#         if isinstance(fuzzy_bonus, (int, float)):
#             fuzzy_component = float(fuzzy_bonus)
#         else:
            
#             fuzzy_component = 0.0

#         computed_total = bm25 + exact_bonus + partial_bonus + fuzzy_component
#         reported_total = float(getattr(res, "score", computed_total))

    
#         return lines

#     @staticmethod
#     def format_results(results: List[SearchResult]) -> str:
#         lines = []
#         for res in results:
#             chunk = res.chunk
#             title = res.document_name or (chunk.document_name if chunk else '[Unknown]')
#             path = res.source_path or (chunk.source_path if chunk else '[Unknown path]')
#             lines.append(f"#{res.rank or '[?]'} - {title}")
#             lines.append(f"Path: {path}")

        
#             chunk_summary = (getattr(chunk, 'summary', '') if chunk else '') or ''

#             if res.mode in ('exact', 'exact+fuzzy'):
               
#                 lines.append("Full Text with Highlights:")
#                 lines.append((res.text or "").strip())

                
#                 if chunk_summary:
#                     lines.append("Summary:")
#                     lines.append(chunk_summary)
#                 else:
                   
#                     summary_from_detail = ""
#                     if getattr(res, "detail", None):
#                         for line in res.detail.splitlines():
#                             if line.strip().lower().startswith("summary:"):
                               
#                                 summary_from_detail = line.split(":", 1)[-1].strip()
#                                 break
#                     if summary_from_detail:
#                         lines.append("Summary:")
#                         lines.append((summary_from_detail or "").strip())
#                #optional inclusion
#                 #lines.extend(ResultFormatter._format_exact_breakdown(res))

#             elif res.mode == 'semantic':
               
#                 lines.append("Full Text with Highlights:")
#                 lines.append(res.text)

#                 lines.append("Summary:")
#                 if chunk_summary:
               
#                     lines.append(chunk_summary)
#                 else:
#                     detail = (res.detail)
#                     lines.append(detail if detail else "[no explanation provided]")

#             else:
#                 lines.append("[Unknown search mode]")
#                 lines.append((res.text or "").strip())

#             lines.append("-" * 60)
#         return "\n".join(lines)

#     @staticmethod
#     def print_results(results: List[SearchResult]) -> None:
#         output = ResultFormatter.format_results(results)
#         print(output.strip() if output.strip() else "[No results to display]")



# from models.data_models import SearchResult
# from typing import List, Optional

# EXACT_BONUS_PER_MATCH = 10.0
# PARTIAL_BONUS_PER_MATCH = 2.0

# # Optional, just to keep output tidy
# _MAX_SUMMARY_CHARS = 400
# _MAX_TEXT_CHARS = 1200

# def _truncate(s: str, n: int) -> str:
#     if not isinstance(s, str):
#         return ""
#     s = s.strip()
#     return (s[:n] + "...") if len(s) > n else s

# class ResultFormatter:
#     @staticmethod
#     def _format_exact_breakdown(res: SearchResult) -> List[str]:
#         lines = []
#         meta = getattr(res, "meta", {}) or {}

#         bm25 = float(meta.get("bm25_score", 0.0))
#         exact_cnt = int(meta.get("exact_match_count", 0) or 0)
#         partial_cnt = int(meta.get("partial_match_count", 0) or 0)

#        #legacy option still here but should be removed in future
#         fuzzy_bonus = meta.get("fuzzy_bonus", None)
#         fuzzy_cnt = meta.get("fuzzy_match_count", None)
#         fuzzy_threshold = meta.get("fuzzy_threshold", None)

#         exact_bonus = EXACT_BONUS_PER_MATCH * exact_cnt
#         partial_bonus = PARTIAL_BONUS_PER_MATCH * partial_cnt

#         if isinstance(fuzzy_bonus, (int, float)):
#             fuzzy_component = float(fuzzy_bonus)
#         else:
#             # keep zero if not present
#             fuzzy_component = 0.0

#         computed_total = bm25 + exact_bonus + partial_bonus + fuzzy_component
#         reported_total = float(getattr(res, "score", computed_total))

#         # (intentionally returning empty lines; extend here if you want verbose scoring)
#         return lines

#     @staticmethod
#     def format_results(results: List[SearchResult]) -> str:
#         lines = []
#         for res in results:
#             chunk = res.chunk
#             title = res.document_name or (chunk.document_name if chunk else '[Unknown]')
#             path = res.source_path or (chunk.source_path if chunk else '[Unknown path]')
#             lines.append(f"#{res.rank or '[?]'} - {title}")
#             lines.append(f"Path: {path}")

#             # prefer chunk.summary if present (persisted for semantic); otherwise fall back later
#             chunk_summary = (getattr(chunk, 'summary', '') if chunk else '') or ''

#             if res.mode in ('exact', 'exact+fuzzy'):
#                 lines.append("Full Text with Highlights:")
#                 lines.append((res.text or "").strip())

#                 # Always show a Summary block in exact mode
#                 summary_text = ""
#                 if chunk_summary:
#                     summary_text = chunk_summary
#                 else:
#                     d = (getattr(res, "detail", "") or "").strip()
#                     if d.lower().startswith("summary:"):
#                         summary_text = d.split(":", 1)[-1].strip()
#                     else:
#                         summary_text = d or "[No explanation provided]"

#                 lines.append("Summary:")
#                 lines.append(_truncate(summary_text, _MAX_SUMMARY_CHARS))
#                #optional inclusion
#                 #lines.extend(ResultFormatter._format_exact_breakdown(res))

#             elif res.mode == 'semantic':
#                 lines.append("Full Text with Highlights:")
#                 lines.append(res.text)

#                 lines.append("Summary:")
#                 if chunk_summary:
#                     lines.append(chunk_summary)
#                 # else:
#                 #     detail = (res.detail)
#                 #     lines.append(detail if detail else "[no explanation provided]")

#             else:
#                 lines.append("[Unknown search mode]")
#                 lines.append((res.text or "").strip())

#             lines.append("-" * 60)
#         return "\n".join(lines)

#     @staticmethod
#     def print_results(results: List[SearchResult]) -> None:
#         output = ResultFormatter.format_results(results)
#         print(output.strip() if output.strip() else "[No results to display]")


from models.data_models import SearchResult
from typing import List, Optional

EXACT_BONUS_PER_MATCH = 10.0
PARTIAL_BONUS_PER_MATCH = 2.0

# Optional, just to keep output tidy
_MAX_SUMMARY_CHARS = 400
_MAX_TEXT_CHARS = 1200

def _truncate(s: str, n: int) -> str:
    if not isinstance(s, str):
        return ""
    s = s.strip()
    return (s[:n] + "...") if len(s) > n else s

class ResultFormatter:
    @staticmethod
    def _format_exact_breakdown(res: SearchResult) -> List[str]:
        lines = []
        meta = getattr(res, "meta", {}) or {}

        bm25 = float(meta.get("bm25_score", 0.0))
        exact_cnt = int(meta.get("exact_match_count", 0) or 0)
        partial_cnt = int(meta.get("partial_match_count", 0) or 0)

       #legacy option still here but should be removed in future
        fuzzy_bonus = meta.get("fuzzy_bonus", None)
        fuzzy_cnt = meta.get("fuzzy_match_count", None)
        fuzzy_threshold = meta.get("fuzzy_threshold", None)

        exact_bonus = EXACT_BONUS_PER_MATCH * exact_cnt
        partial_bonus = PARTIAL_BONUS_PER_MATCH * partial_cnt

        if isinstance(fuzzy_bonus, (int, float)):
            fuzzy_component = float(fuzzy_bonus)
        else:
            
            fuzzy_component = 0.0

        computed_total = bm25 + exact_bonus + partial_bonus + fuzzy_component
        reported_total = float(getattr(res, "score", computed_total))

        
        return lines

    @staticmethod
    def format_results(results: List[SearchResult]) -> str:
        lines = []
        for res in results:
            chunk = res.chunk
            title = res.document_name or (chunk.document_name if chunk else '[Unknown]')
            path = res.source_path or (chunk.source_path if chunk else '[Unknown path]')
            lines.append(f"#{res.rank or '[?]'} - {title}")
            lines.append(f"Path: {path}")

            chunk_summary = (getattr(chunk, 'summary', '') if chunk else '') or ''

            if res.mode in ('exact', 'exact+fuzzy'):
                lines.append("Full Text with Highlights:")
                lines.append((res.text or "").strip())

                
                lines.append("Summary:")
                if chunk_summary:
                    lines.append(_truncate(chunk_summary, _MAX_SUMMARY_CHARS))
                else:
                    lines.append("[No explanation provided]")

               #optional inclusion
                #lines.extend(ResultFormatter._format_exact_breakdown(res))

            elif res.mode == 'semantic':
                lines.append("Full Text with Highlights:")
                lines.append(res.text)

                lines.append("Summary:")
                if chunk_summary:
                    lines.append(chunk_summary)
                else:
                    detail = (res.detail)
                    lines.append(detail if detail else "[no explanation provided]")

            else:
                lines.append("[Unknown search mode]")
                lines.append((res.text or "").strip())

            lines.append("-" * 60)
        return "\n".join(lines)

    @staticmethod
    def print_results(results: List[SearchResult]) -> None:
        output = ResultFormatter.format_results(results)
        print(output.strip() if output.strip() else "[No results to display]")

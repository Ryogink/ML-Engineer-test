"""Fuzzy-match extracted names against the requested name pairs.

Full-string similarity, case-insensitive, accent-insensitive. Only reports
matches at or above a configurable threshold (default 90%). Scores 0..1.
"""

import unicodedata

from thefuzz import fuzz


def _normalize(text: str) -> str:
    """NFKD normalize, casefold, strip accents."""
    nfkd = unicodedata.normalize("NFKD", text.casefold())
    return "".join(c for c in nfkd if not unicodedata.combining(c))


def fuzzy_match_names(
    extracted_names: list[str], query_names: list[dict], threshold: int = 90
) -> list[dict]:
    """Return matches as ``{"extracted_name", "matched_name", "score"}``."""
    results = []
    for extracted in extracted_names:
        for query in query_names:
            query_full = f"{query.get('first_name', '')} {query.get('last_name', '')}".strip()
            if not query_full:
                continue
            score = fuzz.ratio(_normalize(extracted), _normalize(query_full))
            if score >= threshold:
                results.append({
                    "extracted_name": extracted,
                    "matched_name": query_full,
                    "score": score / 100.0,
                })
    return results

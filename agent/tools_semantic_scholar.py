"""Semantic Scholar ingestion tool.

This module talks to the public Semantic Scholar API using `requests`.

- It is used to find scientific papers across many domains, not just PubMed.
- It returns a normalised structure:
      {title, snippet, url, source="semantic-scholar", paperId}

TGRM can combine these with web + PubMed results to build a unified
evidence set for a research cycle.
"""

from __future__ import annotations

from typing import Any, Dict, List

import requests


class SemanticScholarTool:
    """Minimal Semantic Scholar client."""

    SEARCH_URL = "https://api.semanticscholar.org/graph/v1/paper/search"

    def __init__(self) -> None:
        # No API key is required for basic usage; rate limits apply.
        self.fields = "title,abstract,url,citationCount,year"

    def search(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """Search Semantic Scholar for papers related to `query`."""
        if not query:
            return []

        try:
            params = {
                "query": query,
                "limit": max_results,
                "fields": self.fields,
            }
            resp = requests.get(self.SEARCH_URL, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            papers = data.get("data", []) or []

            results: List[Dict[str, str]] = []
            for p in papers:
                title = p.get("title", "No title")
                snippet = p.get("abstract", "") or ""
                url = p.get("url", "") or ""
                paper_id = p.get("paperId", "") or ""
                results.append(
                    {
                        "title": title,
                        "snippet": snippet,
                        "url": url,
                        "source": "semantic-scholar",
                        "paperId": paper_id,
                    }
                )
            return results

        except Exception as e:
            return [
                {
                    "title": f"[STUB] Semantic Scholar error for query='{query}'",
                    "snippet": f"Semantic Scholar request failed: {e}",
                    "url": "",
                    "source": "semantic-scholar",
                    "paperId": "",
                }
            ]

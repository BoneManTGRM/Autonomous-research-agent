"""PubMed ingestion tool.

This module provides a thin wrapper around NCBI PubMed E-utilities using
the `requests` library. It is purposely simple and defensive:

- On network or API failure, it returns stubbed results instead of crashing.
- It exposes a unified interface:
      search(query) -> list of {title, snippet, url, source, pmid}

TGRM can call this during the Repair phase when:
- source_controls["pubmed"] is True
- the goal or notes suggest biomedical or aging topics

Design goals:
- Safe: never crash the agent on network or API errors.
- Normalized: return a citation friendly structure consistent with other tools.
- Configurable: allow an optional NCBI API key and email for better rate limits.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import re

import requests


class PubMedTool:
    """Minimal PubMed client using NCBI E-utilities."""

    ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    DB = "pubmed"

    def __init__(
        self,
        email: str = "example@example.com",
        api_key: Optional[str] = None,
    ) -> None:
        """Create a PubMedTool.

        Args:
            email:
                Contact email string. NCBI recommends including one.
            api_key:
                Optional NCBI API key. If not provided, the tool will look
                for NCBI_API_KEY in the environment. This can improve rate
                limits on heavy autonomous runs.
        """
        self.email = email
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = os.getenv("NCBI_API_KEY", None)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _request_json(self, url: str, params: Dict[str, Any], timeout: int = 10) -> Dict[str, Any]:
        """Perform a GET request and return JSON, with defensive error handling."""
        # Always include email and optional api_key
        params = dict(params)
        params.setdefault("email", self.email)
        if self.api_key:
            params.setdefault("api_key", self.api_key)

        resp = requests.get(url, params=params, timeout=timeout)
        resp.raise_for_status()
        return resp.json()

    # ------------------------------------------------------------------
    # Public search API
    # ------------------------------------------------------------------
    def search(self, query: str, max_results: int = 5) -> List[Dict[str, str]]:
        """Search PubMed and return a list of structured results.

        This method sanitizes the incoming query to avoid overly long or malformed
        search terms being sent to the NCBI API. It also guards against network
        errors by returning a stubbed result instead of raising exceptions.

        Each result is a dict with the following keys:

            - ``title``: Title of the PubMed article (``"No title"`` if missing).
            - ``snippet``: A short snippet derived from the first author or source journal.
            - ``url``: Direct link to the PubMed abstract.
            - ``source``: Constant string ``"pubmed"``.
            - ``pmid``: PubMed identifier for the article.

        Args:
            query: User supplied query string. Newlines and excessive whitespace
                will be collapsed and the string will be truncated to 200 characters.
            max_results: Maximum number of PubMed records to return.

        Returns:
            A list of result dictionaries. If the search fails, a single stub
            record will be returned containing error details.
        """
        # Defensive: return empty list on falsy query
        if not query:
            return []

        # ------------------------------------------------------------------
        # Sanitize the query
        # ------------------------------------------------------------------
        # Collapse newlines and excessive whitespace
        sanitized = " ".join(str(query).strip().split())
        # Extract main query from structured prompts (e.g., TOPIC: ...)
        for key in ("TOPIC:", "Topic:", "QUERY:", "Query:", "GOAL:", "Goal:"):
            idx = sanitized.lower().find(key.lower())
            if idx != -1:
                sanitized = sanitized[idx + len(key):].strip()
                break
        # Truncate to a reasonable length to avoid hitting URL limits or
        # triggering PubMed errors (200 chars is often sufficient)
        sanitized = sanitized[:200]

        try:
            # Step 1: esearch to get PMIDs
            params = {
                "db": self.DB,
                "term": sanitized,
                "retmax": max_results,
                "retmode": "json",
            }
            data = self._request_json(self.ESEARCH_URL, params=params, timeout=10)
            id_list = data.get("esearchresult", {}).get("idlist", [])
            if not id_list:
                return []

            # Step 2: esummary to get titles and snippets
            params2 = {
                "db": self.DB,
                "id": ",".join(id_list),
                "retmode": "json",
            }
            data2 = self._request_json(self.ESUMMARY_URL, params=params2, timeout=10)
            result_dict = data2.get("result", {})

            results: List[Dict[str, str]] = []

            for pmid in id_list:
                rec: Dict[str, Any] = result_dict.get(str(pmid), {})
                # Title (fall back to placeholder if missing)
                title = rec.get("title", "") or "No title"

                # Extract publication metadata
                authors_val = rec.get("sortfirstauthor") or ""
                # Fallback: if rec.get("authors") is a list of dicts with 'name'
                if not authors_val:
                    auths = rec.get("authors")
                    try:
                        if isinstance(auths, list) and auths:
                            first_author = auths[0]
                            if isinstance(first_author, dict):
                                authors_val = first_author.get("name", "") or ""
                    except Exception:
                        pass
                venue_val = rec.get("fulljournalname") or rec.get("source") or ""
                pubdate = rec.get("pubdate", "") or ""
                year_val = ""
                if pubdate:
                    # Extract first 4-digit year from pubdate
                    m = re.search(r"(\d{4})", str(pubdate))
                    if m:
                        year_val = m.group(1)

                # Build a richer snippet: include author, year, venue
                snippet_parts: List[str] = []
                if authors_val:
                    snippet_parts.append(str(authors_val))
                if year_val:
                    snippet_parts.append(str(year_val))
                if venue_val:
                    snippet_parts.append(str(venue_val))
                snippet = " | ".join(snippet_parts)

                url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

                results.append(
                    {
                        "title": str(title),
                        "snippet": str(snippet),
                        "url": str(url),
                        "source": "pubmed",
                        "pmid": str(pmid),
                        "year": year_val,
                        "venue": venue_val,
                        "authors": authors_val,
                    }
                )

            return results

        except Exception as e:
            # Safe fallback: stubbed result so TGRM does not break
            return [
                {
                    "title": f"[STUB] PubMed error for query='{sanitized}'",
                    "snippet": f"PubMed request failed: {e}",
                    "url": "",
                    "source": "pubmed",
                    "pmid": "",
                }
            ]

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

        Each result is a dict:
            {
              "title":   ...,
              "snippet": ...,
              "url":     ...,
              "source":  "pubmed",
              "pmid":    ...,
            }
        """
        if not query:
            return []

        try:
            # Step 1: esearch to get PMIDs
            params = {
                "db": self.DB,
                "term": query,
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
                rec: Dict[str, Any] = result_dict.get(pmid, {})
                title = rec.get("title", "") or "No title"

                # Use a compact snippet field: first author or source journal
                snippet = rec.get("sortfirstauthor", "") or rec.get("source", "") or ""
                url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"

                results.append(
                    {
                        "title": str(title),
                        "snippet": str(snippet),
                        "url": str(url),
                        "source": "pubmed",
                        "pmid": str(pmid),
                    }
                )

            return results

        except Exception as e:
            # Safe fallback: stubbed result so TGRM does not break
            return [
                {
                    "title": f"[STUB] PubMed error for query='{query}'",
                    "snippet": f"PubMed request failed: {e}",
                    "url": "",
                    "source": "pubmed",
                    "pmid": "",
                }
            ]

"""PubMed ingestion tool.

This module provides a thin wrapper around NCBI PubMed E-utilities using
the `requests` library. It is purposely simple and defensive:

- On network or API failure, it returns stubbed results instead of crashing.
- It exposes a unified interface:
      search(query) -> list of {title, snippet, url, source, pmid}

TGRM can call this during the Repair phase when:
- source_controls["pubmed"] is True
- the goal or notes suggest biomedical / aging topics
"""

from __future__ import annotations

from typing import Any, Dict, List

import requests


class PubMedTool:
    """Minimal PubMed client using NCBI E-utilities."""

    ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    ESUMMARY_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    DB = "pubmed"

    def __init__(self, email: str = "example@example.com") -> None:
        # NCBI recommends including an email, even if generic
        self.email = email

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
                "email": self.email,
            }
            resp = requests.get(self. ESEARCH_URL, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            id_list = data.get("esearchresult", {}).get("idlist", [])
            if not id_list:
                return []

            # Step 2: esummary to get titles and snippets
            params2 = {
                "db": self.DB,
                "id": ",".join(id_list),
                "retmode": "json",
                "email": self.email,
            }
            resp2 = requests.get(self.ESUMMARY_URL, params=params2, timeout=10)
            resp2.raise_for_status()
            data2 = resp2.json()
            result_dict = data2.get("result", {})

            results: List[Dict[str, str]] = []
            for pmid in id_list:
                rec: Dict[str, Any] = result_dict.get(pmid, {})
                title = rec.get("title", "") or "No title"
                snippet = rec.get("sortfirstauthor", "") or rec.get("source", "")
                url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
                results.append(
                    {
                        "title": title,
                        "snippet": snippet,
                        "url": url,
                        "source": "pubmed",
                        "pmid": pmid,
                    }
                )

            return results

        except Exception as e:
            # Safe fallback: stubbed result
            return [
                {
                    "title": f"[STUB] PubMed error for query='{query}'",
                    "snippet": f"PubMed request failed: {e}",
                    "url": "",
                    "source": "pubmed",
                    "pmid": "",
                }
            ]

"""Real web research tool using Tavily Search API.

This module powers the "web" part of the autonomous research agent.

Features
--------
- REAL internet search via Tavily when TAVILY_API_KEY is available.
- Safe stubbed search when no key / client is available (agent never crashes).
- Optional page fetch + HTML cleanup for deeper analysis.
- Helper to convert search results into structured citation objects.
- Normalised citation structure for use across PubMed / Semantic / Web.

Reparodynamics / TGRM:
    The Repair phase calls this tool to bring in new information from
    the environment. The quality and efficiency of these calls are
    reflected in ΔR (issues resolved, contradictions clarified) and E
    (energy cost) which together define RYE = ΔR / E.
"""

from __future__ import annotations

import os
from typing import List, Dict, Any, Optional
from urllib.parse import urlparse

import requests

try:
    # High level Tavily client (recommended)
    from tavily import TavilyClient
except ImportError:
    TavilyClient = None  # Streamlit Cloud installs this from requirements.txt

try:
    # For optional HTML cleanup of fetched pages
    from bs4 import BeautifulSoup  # type: ignore
except ImportError:
    BeautifulSoup = None  # Optional – tool still works without this


class WebResearchTool:
    """Web research tool powered by Tavily Search API."""

    def __init__(self) -> None:
        # Streamlit Secrets inject TAVILY_API_KEY into environment
        self.api_key = os.getenv("TAVILY_API_KEY", None)
        self.client: Optional[TavilyClient] = None  # type: ignore[type-arg]

        if self.api_key and TavilyClient is not None:
            try:
                self.client = TavilyClient(api_key=self.api_key)
            except Exception:
                # If anything goes wrong, fall back to stub mode
                self.client = None

    # ------------------------------------------------------------------
    # REAL INTERNET SEARCH
    # ------------------------------------------------------------------
    def search(self, query: str) -> List[Dict[str, str]]:
        """Perform a REAL web search using Tavily, if available.

        If Tavily is not configured, return a stub result so the
        agent can still complete cycles gracefully.
        """
        if not self.client:
            # STUB FALLBACK MODE
            return [
                {
                    "title": f"[STUB] No Tavily key: query='{query}'",
                    "snippet": (
                        "Tavily API key missing or Tavily client not initialised. "
                        "Using placeholder result instead of real web search."
                    ),
                    "url": "",
                }
            ]

        # REAL Tavily API call
        try:
            response = self.client.search(
                query=query,
                max_results=5,
                topic="general",
                search_depth="advanced",
            )

            results: List[Dict[str, str]] = []
            for item in response.get("results", []):
                results.append(
                    {
                        "title": item.get("title", "No title"),
                        # Prefer snippet, fall back to content if needed
                        "snippet": item.get("snippet", item.get("content", "")),
                        "url": item.get("url", ""),
                    }
                )

            if results:
                return results

            # If API returns empty, fall back safely
            return [
                {
                    "title": "No results found",
                    "snippet": "Tavily returned no results for this query.",
                    "url": "",
                }
            ]

        except Exception as e:
            # Safe fallback on API errors
            return [
                {
                    "title": "Tavily Search Error",
                    "snippet": str(e),
                    "url": "",
                }
            ]

    # ------------------------------------------------------------------
    # SUMMARISATION
    # ------------------------------------------------------------------
    def summarize_results(self, results: List[Dict[str, str]]) -> str:
        """Create a short summary from search results.

        Used by the TGRM Repair phase to store compact notes that still
        retain a high RYE (information gain per unit of effort).
        """
        if not results:
            return "No results found."

        titles = [r.get("title", "") for r in results[:3]]
        return "; ".join(titles)

    # ------------------------------------------------------------------
    # OPTIONAL PAGE FETCH + CLEANUP
    # ------------------------------------------------------------------
    def fetch_page_text(self, url: str, max_chars: int = 4000) -> str:
        """Download a web page and return cleaned text (best-effort).

        This is optional but useful when the agent wants more context
        from a specific source discovered via Tavily.

        If BeautifulSoup is installed, we strip HTML and keep visible
        text. Otherwise we return raw HTML.
        """
        if not url:
            return "[No URL provided]"

        try:
            resp = requests.get(url, timeout=15)
            resp.raise_for_status()
        except Exception as e:
            return f"[Error fetching page: {e}]"

        content = resp.text

        if BeautifulSoup is not None:
            try:
                soup = BeautifulSoup(content, "lxml")
                # Simple strategy: join all paragraph text
                paragraphs = [p.get_text(strip=True) for p in soup.find_all("p")]
                text = "\n".join(p for p in paragraphs if p)
            except Exception:
                text = content
        else:
            # No HTML parser available
            text = content

        # Truncate to avoid huge logs
        if len(text) > max_chars:
            return text[:max_chars].rstrip() + " ..."
        return text.strip()

    # ------------------------------------------------------------------
    # INTERNAL HELPERS FOR CITATIONS
    # ------------------------------------------------------------------
    @staticmethod
    def _infer_source_label(url: str) -> str:
        """Infer a short source label from a URL domain."""
        if not url:
            return "web"
        try:
            parsed = urlparse(url)
            host = parsed.netloc.lower()
            if "pubmed" in host or "ncbi.nlm.nih.gov" in host:
                return "pubmed"
            if "semanticscholar.org" in host:
                return "semantic-scholar"
            if "arxiv.org" in host:
                return "arxiv"
            if "doi.org" in host:
                return "doi"
            # Fallback: use the hostname
            return host or "web"
        except Exception:
            return "web"

    @staticmethod
    def normalize_result(result: Dict[str, Any]) -> Dict[str, str]:
        """Normalize a generic search result into a citation-like structure.

        This makes it easier to merge Tavily, PubMed, and Semantic Scholar
        into a single citation format in the TGRM loop.
        """
        title = result.get("title", "") or result.get("name", "") or "Untitled"
        url = result.get("url", "") or result.get("link", "")
        snippet = (
            result.get("snippet")
            or result.get("content")
            or result.get("abstract")
            or ""
        )

        source = result.get("source") or WebResearchTool._infer_source_label(url)

        return {
            "source": str(source),
            "title": str(title),
            "url": str(url),
            "snippet": str(snippet),
        }

    # ------------------------------------------------------------------
    # CITATION EXTRACTION
    # ------------------------------------------------------------------
    def to_citations(self, results: List[Dict[str, Any]]) -> List[Dict[str, str]]:
        """Convert search results into structured citation objects.

        Each citation is a simple dict:
            { "source": "web", "title": ..., "url": ..., "snippet": ... }

        TGRM can log these as part of the cycle data so RYE can be
        evaluated not just at the text level but also at the source level.
        """
        citations: List[Dict[str, str]] = []
        for r in results:
            citations.append(self.normalize_result(r))
        return citations

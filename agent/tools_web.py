"""Real web research tool using Tavily Search API.

This module powers the "web" part of the autonomous research agent.

Features
--------
- REAL internet search via Tavily when TAVILY_API_KEY is available.
- Safe stubbed search when no key or client is available (agent never crashes).
- Optional page fetch plus HTML cleanup for deeper analysis.
- Helper to convert search results into structured citation objects.
- Normalised citation structure for use across PubMed, Semantic Scholar, and web.
- Multi-level search depth (Level 1 / 2 / 3) for cost vs depth control.
- In-memory caching to avoid repeated API calls in long autonomous runs.

Reparodynamics / TGRM:
    The Repair phase calls this tool to bring in new information from
    the environment. The quality and efficiency of these calls are
    reflected in ΔR (issues resolved, contradictions clarified) and E
    (energy cost) which together define RYE = ΔR / E.

API key model:
    - This file never hardcodes a key.
    - It prefers a key passed in to WebResearchTool(api_key=...).
    - If none is passed, it falls back to the TAVILY_API_KEY environment
      variable (set by your UI, for example Streamlit).

Search levels (for 50x value):
    Level 1 (cheap scan):
        - Low max_results
        - Basic Tavily depth
        - No page fetch

    Level 2 (standard research):
        - Default mode for the agent
        - Advanced Tavily search_depth
        - Good balance of depth and cost

    Level 3 (deep dive):
        - Advanced search_depth
        - Optional page fetch and cleaned text
        - Best when you really care about one question

Environment safety controls:
    - WEB_TOOL_MAX_LEVEL  : cap the maximum allowed level (default 3)
    - WEB_TOOL_CACHE_SIZE : max cached queries before eviction (default 256)
"""

from __future__ import annotations

import os
from typing import List, Dict, Any, Optional, Tuple
from urllib.parse import urlparse

import requests

try:
    # High level Tavily client (recommended)
    from tavily import TavilyClient
except ImportError:
    TavilyClient = None  # Installed from requirements.txt if available

try:
    # For optional HTML cleanup of fetched pages
    from bs4 import BeautifulSoup  # type: ignore
except ImportError:
    BeautifulSoup = None  # Optional - tool still works without this


class WebResearchTool:
    """Web research tool powered by Tavily Search API with multi-level depth."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        """Create a WebResearchTool.

        Args:
            api_key:
                Optional Tavily API key. If not provided, the tool will
                look for TAVILY_API_KEY in the environment. This lets
                each user supply their own key at runtime through the UI.
        """
        # Prefer explicit key if given, otherwise read from environment
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = os.getenv("TAVILY_API_KEY", None)

        self.client: Optional[TavilyClient] = None  # type: ignore[type-arg]

        if self.api_key and TavilyClient is not None:
            try:
                self.client = TavilyClient(api_key=self.api_key)
            except Exception:
                # If anything goes wrong, fall back to stub mode
                self.client = None

        # Safety controls for deep search
        try:
            self.max_level: int = max(
                1, min(int(os.getenv("WEB_TOOL_MAX_LEVEL", "3")), 3)
            )
        except ValueError:
            self.max_level = 3

        # Simple in-memory cache: maps (query, level, max_results, topic) -> results
        self._cache: Dict[Tuple[str, int, int, str], List[Dict[str, Any]]] = {}
        try:
            self._cache_size_limit: int = max(
                16, min(int(os.getenv("WEB_TOOL_CACHE_SIZE", "256")), 2048)
            )
        except ValueError:
            self._cache_size_limit = 256

    # ------------------------------------------------------------------
    # INTERNAL: CACHE HELPERS
    # ------------------------------------------------------------------
    def _make_cache_key(
        self,
        query: str,
        level: int,
        max_results: int,
        topic: str,
    ) -> Tuple[str, int, int, str]:
        return (query.strip(), int(level), int(max_results), topic.strip().lower())

    def _get_from_cache(
        self,
        key: Tuple[str, int, int, str],
    ) -> Optional[List[Dict[str, Any]]]:
        return self._cache.get(key)

    def _store_in_cache(
        self,
        key: Tuple[str, int, int, str],
        results: List[Dict[str, Any]],
    ) -> None:
        # Basic LRU-ish behavior: trim when too large
        if len(self._cache) >= self._cache_size_limit:
            # Drop an arbitrary item (Python 3.7+ dict preserves insertion order)
            oldest_key = next(iter(self._cache))
            self._cache.pop(oldest_key, None)
        # Store a shallow copy to avoid accidental external mutation
        self._cache[key] = [dict(r) for r in results]

    # ------------------------------------------------------------------
    # REAL INTERNET SEARCH (WITH LEVELS)
    # ------------------------------------------------------------------
    def search(
        self,
        query: str,
        level: int = 2,
        max_results: int = 5,
        topic: str = "general",
    ) -> List[Dict[str, Any]]:
        """Perform a REAL web search using Tavily, if available.

        Args:
            query:
                Text query to send to Tavily.
            level:
                Depth level:
                    1 = cheap/fast scan
                    2 = standard research (default)
                    3 = deep dive + optional page fetch
            max_results:
                Maximum results to request from Tavily (upper bound).
            topic:
                Tavily topic hint, e.g. "general", "science", etc.

        Returns:
            List of dicts with keys: title, snippet, url, and optionally
            page_text for level 3 deep results.

        If Tavily is not configured, return a stub result so the
        agent can still complete cycles gracefully.

        Backwards compatibility:
            Existing calls like search("query") still work and map to
            level=2, max_results=5, topic="general".
        """
        # Enforce safety level cap
        safe_level = max(1, min(level, self.max_level))

        # Adjust behavior per level
        if safe_level == 1:
            tavily_depth = "basic"
            effective_max_results = max(1, min(max_results, 3))
            deep_fetch = False
        elif safe_level == 2:
            tavily_depth = "advanced"
            effective_max_results = max(1, max_results)
            deep_fetch = False
        else:  # safe_level == 3
            tavily_depth = "advanced"
            effective_max_results = max(1, max_results)
            deep_fetch = True

        cache_key = self._make_cache_key(query, safe_level, effective_max_results, topic)
        cached = self._get_from_cache(cache_key)
        if cached is not None:
            # Return a copy to avoid mutation from callers
            return [dict(r) for r in cached]

        # STUB FALLBACK MODE
        if not self.client:
            stub = [
                {
                    "title": f"[STUB] No Tavily key: query='{query}'",
                    "snippet": (
                        "Tavily API key missing or Tavily client not initialised. "
                        f"Using placeholder result at level={safe_level} instead of real web search."
                    ),
                    "url": "",
                }
            ]
            self._store_in_cache(cache_key, stub)
            return stub

        # REAL Tavily API call
        try:
            response = self.client.search(
                query=query,
                max_results=effective_max_results,
                topic=topic,
                search_depth=tavily_depth,
            )

            results: List[Dict[str, Any]] = []
            for item in response.get("results", []):
                res: Dict[str, Any] = {
                    "title": item.get("title", "No title"),
                    # Prefer snippet, fall back to content if needed
                    "snippet": item.get("snippet", item.get("content", "")),
                    "url": item.get("url", ""),
                }

                # Level 3: optionally pull the page text for richer context
                if deep_fetch and res["url"]:
                    res["page_text"] = self.fetch_page_text(res["url"], max_chars=4000)

                results.append(res)

            if not results:
                results = [
                    {
                        "title": "No results found",
                        "snippet": "Tavily returned no results for this query.",
                        "url": "",
                    }
                ]

            self._store_in_cache(cache_key, results)
            return results

        except Exception as e:
            # Safe fallback on API errors
            err = [
                {
                    "title": "Tavily Search Error",
                    "snippet": str(e),
                    "url": "",
                }
            ]
            self._store_in_cache(cache_key, err)
            return err

    # ------------------------------------------------------------------
    # SUMMARISATION
    # ------------------------------------------------------------------
    def summarize_results(self, results: List[Dict[str, Any]]) -> str:
        """Create a short summary from search results.

        Used by the TGRM Repair phase to store compact notes that still
        retain a high RYE (information gain per unit of effort).

        If page_text is present (deep mode), this still only uses titles
        so summaries stay short and cheap.
        """
        if not results:
            return "No results found."

        titles = [r.get("title", "") for r in results[:3]]
        return "; ".join(titles)

    # ------------------------------------------------------------------
    # OPTIONAL PAGE FETCH PLUS CLEANUP
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
